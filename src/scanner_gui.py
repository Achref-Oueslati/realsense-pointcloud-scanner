#!/usr/bin/env python3
"""
RealSense D435i -> Open3D CUDA dense RGB-D scanner.

Features
--------
- Live RGB view
- Live aligned depth view
- Live colored point cloud
- Persistent CUDA dense-SLAM / TSDF map
- Camera tracking with frame-to-model RGB-D odometry
- Colored fused map extraction
- Export current-frame PLY
- Export fused-map PLY
- Export fused-map mesh PLY
- Reset map
- Pause / resume

Designed for Open3D 0.20.x + pyrealsense2 2.58.x.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pyrealsense2 as rs


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WIDTH = 848
HEIGHT = 480
FPS = 30

DEPTH_MIN_M = 0.12
DEPTH_MAX_M = 3.5

# Fused map resolution. 6 mm is a good starting point for a D435i.
VOXEL_SIZE_M = 0.006
TRUNC_VOXEL_MULTIPLIER = 8.0
BLOCK_RESOLUTION = 16
BLOCK_COUNT = 40_000

# Let Open3D determine the required output point-cloud size.
# Using a fixed estimate that is too small causes repeated reallocations.
ESTIMATED_MAP_POINTS = -1

# Open3D's documented default is 3.0, which keeps surfaces supported by
# multiple observations and generally gives a cleaner fused map.
SURFACE_WEIGHT_THRESHOLD = 3.0

# Extract/update the displayed fused map every N processed frames.
MAP_UPDATE_INTERVAL = 10
DISPLAY_EVERY = 2

# Current-frame point cloud uses every pixel with valid depth.
CURRENT_PCD_STRIDE = 1

# Viewer point size.
POINT_SIZE = 2.0

# RealSense filtering. Kept conservative because aggressive temporal filtering
# can smear moving edges and can affect tracking during handheld motion.
USE_DEPTH_FILTERS = True

OUTPUT_ROOT = Path.home() / "realsense3d" / "recordings"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = o3c.Device("CUDA:0")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def require_cuda() -> None:
    if not o3d.core.cuda.is_available():
        raise RuntimeError("Open3D CUDA is not available.")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_t_image(
    array: np.ndarray,
    dtype: o3c.Dtype,
    device: o3c.Device,
) -> o3d.t.geometry.Image:
    tensor = o3c.Tensor(array, dtype=dtype, device=device)
    return o3d.t.geometry.Image(tensor)


def create_camera_intrinsic_tensor(
    intr: rs.intrinsics,
    device: o3c.Device,
) -> o3c.Tensor:
    matrix = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Open3D 0.20 Tensor geometry requires Float64 intrinsics.
    return o3c.Tensor(
        matrix,
        dtype=o3c.float64,
        device=device,
    )


def colorize_depth(
    depth_image: o3d.t.geometry.Image,
    depth_scale: float,
) -> o3d.t.geometry.Image:
    return depth_image.colorize_depth(
        depth_scale,
        DEPTH_MIN_M,
        DEPTH_MAX_M,
    )


def get_legacy_empty_image(
    width: int,
    height: int,
    channels: int,
) -> o3d.geometry.Image:
    if channels == 1:
        arr = np.zeros(
            (height, width),
            dtype=np.uint8,
        )
    else:
        arr = np.zeros(
            (height, width, channels),
            dtype=np.uint8,
        )

    return o3d.geometry.Image(arr)


# -----------------------------------------------------------------------------
# RealSense capture
# -----------------------------------------------------------------------------


class RealSenseCapture:
    def __init__(self) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Depth is aligned into the color camera coordinate system.
        self.align = rs.align(rs.stream.color)

        self.profile = None
        self.depth_scale = None
        self.color_intrinsics = None

        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        self.config.enable_stream(
            rs.stream.depth,
            WIDTH,
            HEIGHT,
            rs.format.z16,
            FPS,
        )

        self.config.enable_stream(
            rs.stream.color,
            WIDTH,
            HEIGHT,
            rs.format.rgb8,
            FPS,
        )

    def start(self) -> tuple[float, rs.intrinsics]:
        self.profile = self.pipeline.start(self.config)

        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()

        # RealSense returns meters per raw depth unit.
        #
        # Example:
        #     0.001 m/unit
        #
        # Open3D expects:
        #
        #     meters = raw_depth / depth_scale
        #
        # therefore:
        #
        #     Open3D depth_scale ~= 1000
        #
        self.rs_depth_scale = float(
            depth_sensor.get_depth_scale()
        )

        self.depth_scale = 1.0 / self.rs_depth_scale

        self.color_intrinsics = (
            self.profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        name = device.get_info(
            rs.camera_info.name
        )

        serial = device.get_info(
            rs.camera_info.serial_number
        )

        firmware = device.get_info(
            rs.camera_info.firmware_version
        )

        print("========== REALSENSE ==========")
        print("Camera:", name)
        print("Serial:", serial)
        print("Firmware:", firmware)

        print(
            "RealSense depth scale (m/unit):",
            self.rs_depth_scale,
        )

        print(
            "Open3D depth scale (unit/m):",
            self.depth_scale,
        )

        print(
            "Color:",
            WIDTH,
            "x",
            HEIGHT,
            "@",
            FPS,
        )

        print()

        return (
            self.depth_scale,
            self.color_intrinsics,
        )

    def get_frame(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:

        frames = self.pipeline.wait_for_frames()

        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError(
                "Missing depth or color frame."
            )

        if USE_DEPTH_FILTERS:
            depth_frame = self.spatial.process(
                depth_frame
            )

            depth_frame = self.temporal.process(
                depth_frame
            )

            depth_frame = self.hole_filling.process(
                depth_frame
            )

        depth = np.asanyarray(
            depth_frame.get_data()
        )

        color = np.asanyarray(
            color_frame.get_data()
        )

        return depth, color

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# CUDA dense SLAM model
# -----------------------------------------------------------------------------
#
# This follows the Open3D 0.20 dense-SLAM structure:
#
#   input frame
#       ↓
#   track against raycast model
#       ↓
#   accumulate T_frame_to_model
#       ↓
#   update frame pose
#       ↓
#   integrate into TSDF
#       ↓
#   synthesize/raycast next model frame
#
# We deliberately do NOT integrate frames when tracking fails.
#
# -----------------------------------------------------------------------------


class DenseScanner:
    def __init__(
        self,
        depth_scale: float,
        color_intrinsics: rs.intrinsics,
    ) -> None:

        self.depth_scale = depth_scale
        self.device = DEVICE
        self.frame_id = 0

        # T_frame_to_model / world pose.
        #
        # Keep the accumulated transform as Float64 on CPU,
        # matching the Open3D dense-SLAM example.
        self.pose = o3c.Tensor(
            np.eye(4, dtype=np.float64),
            dtype=o3c.float64,
            device=o3c.Device("CPU:0"),
        )

        self.intrinsic = (
            create_camera_intrinsic_tensor(
                color_intrinsics,
                self.device,
            )
        )

        self.model = self._create_model()

        # Open3D maintains two frame objects:
        #
        # 1. input_frame:
        #       actual D435i RGB-D input
        #
        # 2. model_frame:
        #       raycasted RGB-D representation of the
        #       current TSDF model
        #
        self.input_frame = o3d.t.pipelines.slam.Frame(
            HEIGHT,
            WIDTH,
            self.intrinsic,
            self.device,
        )

        self.model_frame = o3d.t.pipelines.slam.Frame(
            HEIGHT,
            WIDTH,
            self.intrinsic,
            self.device,
        )

        self.initialized = False

        self.last_tracking_success = True
        self.last_tracking_mode = "INIT"

    def _create_model(self):
        return o3d.t.pipelines.slam.Model(
            VOXEL_SIZE_M,
            BLOCK_RESOLUTION,
            BLOCK_COUNT,
            o3c.Tensor(
                np.eye(4, dtype=np.float64),
                dtype=o3c.float64,
            ),
            self.device,
        )

    def reset(self) -> None:
        self.model = self._create_model()

        self.input_frame = o3d.t.pipelines.slam.Frame(
            HEIGHT,
            WIDTH,
            self.intrinsic,
            self.device,
        )

        self.model_frame = o3d.t.pipelines.slam.Frame(
            HEIGHT,
            WIDTH,
            self.intrinsic,
            self.device,
        )

        self.pose = o3c.Tensor(
            np.eye(4, dtype=np.float64),
            dtype=o3c.float64,
            device=o3c.Device("CPU:0"),
        )

        self.frame_id = 0
        self.initialized = False

        self.last_tracking_success = True
        self.last_tracking_mode = "RESET"

    def _set_input_frame(
        self,
        depth_np: np.ndarray,
        color_np: np.ndarray,
    ) -> None:

        depth = make_t_image(
            depth_np,
            o3c.uint16,
            self.device,
        )

        color = make_t_image(
            color_np,
            o3c.uint8,
            self.device,
        )

        self.input_frame.set_data_from_image(
            "depth",
            depth,
        )

        self.input_frame.set_data_from_image(
            "color",
            color,
        )

    def _integrate(self) -> None:
        self.model.integrate(
            self.input_frame,
            self.depth_scale,
            DEPTH_MAX_M,
            TRUNC_VOXEL_MULTIPLIER,
        )

    def _synthesize_model_frame(self) -> None:
        # Render a new RGB-D view from the current TSDF volume.
        self.model.synthesize_model_frame(
            self.model_frame,
            self.depth_scale,
            DEPTH_MIN_M,
            DEPTH_MAX_M,
            TRUNC_VOXEL_MULTIPLIER,
            True,
            SURFACE_WEIGHT_THRESHOLD,
        )

    def process(
        self,
        depth_np: np.ndarray,
        color_np: np.ndarray,
    ) -> dict:

        self._set_input_frame(
            depth_np,
            color_np,
        )

        # ------------------------------------------------------------------
        # First frame
        # ------------------------------------------------------------------

        if not self.initialized:
            self.model.update_frame_pose(
                self.frame_id,
                self.pose,
            )

            self._integrate()

            self._synthesize_model_frame()

            self.initialized = True

            self.last_tracking_success = True
            self.last_tracking_mode = "INIT"

            self.frame_id += 1

        # ------------------------------------------------------------------
        # Subsequent frames
        # ------------------------------------------------------------------

        else:

            try:
                # Use Open3D's documented frame-to-model tracking.
                #
                # PointToPlane is the default tracking method in
                # Open3D's Model.track_frame_to_model().
                result = (
                    self.model.track_frame_to_model(
                        self.input_frame,
                        self.model_frame,
                        self.depth_scale,
                        DEPTH_MAX_M,
                        0.07,
                    )
                )

                # Accumulate:
                #
                # T_frame_to_model =
                #     T_previous_frame_to_model
                #     @
                #     T_current_frame_relative
                #
                self.pose = (
                    self.pose
                    @ result.transformation.to(
                        o3c.Device("CPU:0")
                    )
                )

                # Register this frame pose with the model.
                self.model.update_frame_pose(
                    self.frame_id,
                    self.pose,
                )

                # Only integrate after successful tracking.
                self._integrate()

                # Generate a new raycast target for the next frame.
                self._synthesize_model_frame()

                self.last_tracking_success = True
                self.last_tracking_mode = (
                    "MODEL-POINT"
                )

            except RuntimeError as exc:

                # Tracking failure must NOT corrupt the map.
                #
                # Keep the previous valid pose and TSDF state.
                self.last_tracking_success = False
                self.last_tracking_mode = "LOST"

                print(
                    "\n[Tracking] Frame "
                    f"{self.frame_id} failed:"
                )

                print(
                    f"[Tracking] {exc}"
                )

                # We intentionally do not integrate this frame.
                #
                # The existing model_frame remains valid because
                # the TSDF itself was not modified.

            self.frame_id += 1

        # ------------------------------------------------------------------
        # Build the current-camera colored point cloud.
        # ------------------------------------------------------------------

        rgbd = o3d.t.geometry.RGBDImage(
            make_t_image(
                color_np,
                o3c.uint8,
                self.device,
            ),
            make_t_image(
                depth_np,
                o3c.uint16,
                self.device,
            ),
        )

        current_pcd = (
            o3d.t.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.intrinsic,
                o3c.Tensor.eye(
                    4,
                    o3c.float64,
                    self.device,
                ),
                self.depth_scale,
                DEPTH_MAX_M,
                CURRENT_PCD_STRIDE,
                False,
            )
        )

        return {
            "frame": self.input_frame,
            "model_frame": self.model_frame,
            "current_pcd": current_pcd,
            "pose": self.pose,
            "tracking_success": self.last_tracking_success,
            "tracking_mode": self.last_tracking_mode,
        }

    def extract_map(
        self,
    ) -> o3d.t.geometry.PointCloud:

        return self.model.extract_pointcloud(
            SURFACE_WEIGHT_THRESHOLD,
            ESTIMATED_MAP_POINTS,
        )

    def extract_mesh(
        self,
    ) -> o3d.t.geometry.TriangleMesh:

        return self.model.extract_trianglemesh(
            SURFACE_WEIGHT_THRESHOLD,
            ESTIMATED_MAP_POINTS,
        )


# -----------------------------------------------------------------------------
# GUI application
# -----------------------------------------------------------------------------


class ScannerWindow:
    def __init__(self) -> None:
        require_cuda()

        self.window = gui.Application.instance.create_window(
            "RealSense D435i - Open3D CUDA 3D Scanner",
            1500,
            900,
        )

        self.window.set_on_layout(
            self.on_layout
        )

        self.window.set_on_close(
            self.on_close
        )

        self.capture = RealSenseCapture()

        (
            self.depth_scale,
            self.color_intrinsics,
        ) = self.capture.start()

        self.scanner = DenseScanner(
            self.depth_scale,
            self.color_intrinsics,
        )

        self.running = True
        self.stopping = False

        self.frame_counter = 0
        self.display_counter = 0

        self.last_map = None
        self.last_current_pcd = None
        self.last_rgb = None
        self.last_depth = None
        self.last_frame_elements = None
        self.last_displayed_map_id = None
        self.last_error = ""

        self.fps_times = []
        self.processing_ms = 0.0
        self.tracking_success = True
        self.pose = np.eye(4)

        self._build_ui()
        self._configure_scene()

        self.worker = threading.Thread(
            target=self._capture_loop,
            name="RealSenseScanner",
            daemon=True,
        )

        self.worker.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        self.panel = gui.Vert(
            8,
            gui.Margins(
                10,
                10,
                10,
                10,
            ),
        )

        title = gui.Label(
            "D435i CUDA 3D Scanner"
        )

        title.text_color = gui.Color(
            0.9,
            0.9,
            0.9,
        )

        self.panel.add_child(title)

        info = gui.Label(
            f"{WIDTH}x{HEIGHT} @ {FPS} FPS | "
            f"voxel {VOXEL_SIZE_M * 1000:.1f} mm | "
            f"CUDA:0"
        )

        self.panel.add_child(info)

        controls = gui.Horiz(6)

        self.pause_button = gui.Button(
            "Pause"
        )

        self.pause_button.set_on_clicked(
            self.toggle_running
        )

        controls.add_child(
            self.pause_button
        )

        reset_button = gui.Button(
            "Reset Map"
        )

        reset_button.set_on_clicked(
            self.reset_map
        )

        controls.add_child(
            reset_button
        )

        self.panel.add_child(
            controls
        )

        export_row = gui.Horiz(6)

        save_frame_button = gui.Button(
            "Save Frame PLY"
        )

        save_frame_button.set_on_clicked(
            self.save_current_pcd
        )

        export_row.add_child(
            save_frame_button
        )

        save_map_button = gui.Button(
            "Save Map PLY"
        )

        save_map_button.set_on_clicked(
            self.save_map
        )

        export_row.add_child(
            save_map_button
        )

        save_mesh_button = gui.Button(
            "Save Mesh PLY"
        )

        save_mesh_button.set_on_clicked(
            self.save_mesh
        )

        export_row.add_child(
            save_mesh_button
        )

        self.panel.add_child(
            export_row
        )

        self.filter_checkbox = gui.Checkbox(
            "RealSense depth filters"
        )

        self.filter_checkbox.checked = (
            USE_DEPTH_FILTERS
        )

        self.filter_checkbox.set_on_checked(
            self.toggle_filters
        )

        self.panel.add_child(
            self.filter_checkbox
        )

        self.status = gui.Label(
            "Starting camera..."
        )

        self.status.text_color = gui.Color(
            0.8,
            0.8,
            0.8,
        )

        self.panel.add_child(
            self.status
        )

        self.rgb_label = gui.Label(
            "RGB"
        )

        self.panel.add_child(
            self.rgb_label
        )

        self.rgb_widget = gui.ImageWidget(
            get_legacy_empty_image(
                WIDTH,
                HEIGHT,
                3,
            )
        )

        self.panel.add_child(
            self.rgb_widget
        )

        self.depth_label = gui.Label(
            "Aligned depth"
        )

        self.panel.add_child(
            self.depth_label
        )

        self.depth_widget = gui.ImageWidget(
            get_legacy_empty_image(
                WIDTH,
                HEIGHT,
                3,
            )
        )

        self.panel.add_child(
            self.depth_widget
        )

        self.window.add_child(
            self.panel
        )

        self.scene_widget = gui.SceneWidget()

        self.scene_widget.scene = (
            rendering.Open3DScene(
                self.window.renderer
            )
        )

        self.scene_widget.scene.set_background(
            [
                0.04,
                0.04,
                0.05,
                1.0,
            ]
        )

        self.scene_widget.scene.show_axes(
            True
        )

        self.window.add_child(
            self.scene_widget
        )

        self.stats = gui.Label(
            "FPS: 0.0\n"
            "Points: 0\n"
            "Map: 0\n"
            "Tracking: OK"
        )

        self.stats.text_color = gui.Color(
            0.9,
            0.9,
            0.9,
        )

        self.window.add_child(
            self.stats
        )

        self.pcd_material = (
            rendering.MaterialRecord()
        )

        self.pcd_material.shader = (
            "defaultUnlit"
        )

        self.pcd_material.point_size = (
            POINT_SIZE * self.window.scaling
        )

        self.pcd_material.sRGB_color = True

        self.map_material = (
            rendering.MaterialRecord()
        )

        self.map_material.shader = (
            "defaultUnlit"
        )

        self.map_material.point_size = (
            2.5 * self.window.scaling
        )

        self.map_material.sRGB_color = True

    def _configure_scene(self) -> None:
        bbox = (
            o3d.geometry.AxisAlignedBoundingBox(
                [-1.5, -1.5, -1.5],
                [1.5, 1.5, 4.0],
            )
        )

        self.scene_widget.set_view_controls(
            gui.SceneWidget.Controls.ROTATE_CAMERA
        )

        self.scene_widget.setup_camera(
            60.0,
            bbox,
            [0.0, 0.0, 1.0],
        )

        self.scene_widget.look_at(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -2.0],
            [0.0, -1.0, 0.0],
        )

    def on_layout(self, ctx) -> None:
        rect = self.window.content_rect

        panel_width = int(
            330
            * ctx.theme.font_size
            / 16.0
        )

        stats_width = int(
            180
            * ctx.theme.font_size
            / 16.0
        )

        panel_width = max(
            panel_width,
            300,
        )

        stats_width = max(
            stats_width,
            160,
        )

        self.panel.frame = gui.Rect(
            rect.x,
            rect.y,
            panel_width,
            rect.height,
        )

        self.scene_widget.frame = gui.Rect(
            rect.x + panel_width,
            rect.y,
            rect.width - panel_width,
            rect.height,
        )

        self.stats.frame = gui.Rect(
            rect.get_right()
            - stats_width
            - 15,
            rect.y + 15,
            stats_width,
            100,
        )

    # -------------------------------------------------------------- callbacks

    def toggle_running(
        self,
        _=None,
    ) -> None:

        self.running = not self.running

        self.pause_button.text = (
            "Pause"
            if self.running
            else "Resume"
        )

        self.set_status(
            "Scanning"
            if self.running
            else "Paused"
        )

    def toggle_filters(
        self,
        checked: bool,
    ) -> None:

        global USE_DEPTH_FILTERS

        USE_DEPTH_FILTERS = bool(
            checked
        )

    def reset_map(self) -> None:
        self.scanner.reset()

        self.last_map = None

        if self.scene_widget.scene.has_geometry(
            "map"
        ):
            self.scene_widget.scene.remove_geometry(
                "map"
            )

        self.last_displayed_map_id = None

        self.set_status(
            "Map reset. "
            "Move slowly to initialize "
            "the new map."
        )

    def set_status(
        self,
        message: str,
    ) -> None:

        self.status.text = message

    def save_current_pcd(self) -> None:
        pcd = self.last_current_pcd

        if pcd is None:
            self.set_status(
                "No current point cloud yet."
            )
            return

        stamp = now_stamp()

        path = (
            OUTPUT_ROOT
            / f"current_frame_{stamp}.ply"
        )

        pcd_cpu = pcd.cpu()

        ok = o3d.t.io.write_point_cloud(
            str(path),
            pcd_cpu,
            False,
            False,
            False,
        )

        self.set_status(
            f"Frame PLY: {path}"
            if ok
            else "Frame PLY save failed."
        )

        print(
            f"Saved current frame: {path}"
        )

    def save_map(self) -> None:
        if self.last_map is None:
            self.set_status(
                "Map is not available yet."
            )
            return

        stamp = now_stamp()

        path = (
            OUTPUT_ROOT
            / f"fused_map_{stamp}.ply"
        )

        ok = o3d.t.io.write_point_cloud(
            str(path),
            self.last_map.cpu(),
            False,
            False,
            False,
        )

        self.set_status(
            f"Map PLY: {path}"
            if ok
            else "Map PLY save failed."
        )

        print(
            f"Saved fused map: {path}"
        )

    def save_mesh(self) -> None:
        if self.stopping:
            return

        self.set_status(
            "Extracting mesh on CUDA..."
        )

        def worker() -> None:
            try:
                mesh = (
                    self.scanner.extract_mesh()
                )

                mesh_cpu = mesh.cpu()

                stamp = now_stamp()

                path = (
                    OUTPUT_ROOT
                    / f"mesh_{stamp}.ply"
                )

                ok = (
                    o3d.t.io.write_triangle_mesh(
                        str(path),
                        mesh_cpu,
                        False,
                        False,
                        False,
                    )
                )

                message = (
                    f"Mesh PLY: {path}"
                    if ok
                    else "Mesh save failed."
                )

                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda: self.set_status(
                        message
                    ),
                )

                print(
                    f"Saved mesh: {path}"
                )

            except Exception as exc:

                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda: self.set_status(
                        f"Mesh error: {exc}"
                    ),
                )

        threading.Thread(
            target=worker,
            name="MeshExport",
            daemon=True,
        ).start()

    def on_close(self) -> bool:
        self.stopping = True
        self.running = False

        self.capture.stop()

        return True

    # -------------------------------------------------------------- processing

    def _capture_loop(self) -> None:
        while not self.stopping:

            if not self.running:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()

            try:
                depth_np, color_np = (
                    self.capture.get_frame()
                )

                result = (
                    self.scanner.process(
                        depth_np,
                        color_np,
                    )
                )

                current_world = (
                    result["current_pcd"]
                )

                current_world.transform(
                    self.scanner.pose.to(
                        DEVICE
                    )
                )

                self.last_current_pcd = (
                    current_world
                )

                self.pose = result["pose"]

                self.tracking_success = (
                    result[
                        "tracking_success"
                    ]
                )

                tracking_mode = result[
                    "tracking_mode"
                ]

                depth_img = make_t_image(
                    depth_np,
                    o3c.uint16,
                    DEVICE,
                )

                color_img = make_t_image(
                    color_np,
                    o3c.uint8,
                    DEVICE,
                )

                depth_colorized = (
                    colorize_depth(
                        depth_img,
                        self.scanner.depth_scale,
                    )
                )

                self.frame_counter += 1
                self.display_counter += 1

                elapsed = (
                    time.perf_counter()
                    - t0
                )

                self.processing_ms = (
                    elapsed * 1000.0
                )

                stamp = (
                    time.perf_counter()
                )

                self.fps_times.append(
                    stamp
                )

                cutoff = (
                    stamp - 2.0
                )

                self.fps_times = [
                    t
                    for t in self.fps_times
                    if t >= cutoff
                ]

                fps = (
                    len(self.fps_times)
                    / 2.0
                )

                # Extract the persistent fused
                # map periodically.
                if (
                    self.frame_counter
                    % MAP_UPDATE_INTERVAL
                    == 0
                ):
                    try:
                        self.last_map = (
                            self.scanner
                            .extract_map()
                            .cpu()
                        )

                    except RuntimeError as exc:
                        self.last_error = (
                            f"Map extraction: {exc}"
                        )

                # Update the GUI at a lower rate
                # than the camera processing loop
                # so rendering does not become the
                # bottleneck.
                if (
                    self.frame_counter
                    % DISPLAY_EVERY
                    == 0
                ):

                    current_cpu = (
                        self.last_current_pcd
                        .cpu()
                    )

                    rgb_cpu = (
                        color_img.to_legacy()
                    )

                    depth_cpu = (
                        depth_colorized
                        .to_legacy()
                    )

                    map_cpu = self.last_map

                    stats = {
                        "fps": fps,
                        "processing_ms": (
                            self.processing_ms
                        ),
                        "current_points": int(
                            current_cpu
                            .point
                            .positions
                            .shape[0]
                        ),
                        "map_points": (
                            0
                            if map_cpu is None
                            else int(
                                map_cpu
                                .point
                                .positions
                                .shape[0]
                            )
                        ),
                        "tracking": (
                            self.tracking_success
                        ),
                        "frame": (
                            self.frame_counter
                        ),
                        "tracking_mode": (
                            tracking_mode
                        ),
                    }

                    self._post_frame_update(
                        rgb_cpu,
                        depth_cpu,
                        current_cpu,
                        map_cpu,
                        stats,
                    )

            except Exception as exc:

                self.last_error = str(
                    exc
                )

                print(
                    f"\n[Scanner] {exc}"
                )

                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda msg=str(
                        exc
                    ): self.set_status(
                        f"ERROR: {msg}"
                    ),
                )

                time.sleep(0.1)

        self.capture.stop()

    def _post_frame_update(
        self,
        rgb: o3d.geometry.Image,
        depth: o3d.geometry.Image,
        current_pcd: o3d.t.geometry.PointCloud,
        map_pcd: (
            o3d.t.geometry.PointCloud
            | None
        ),
        stats: dict,
    ) -> None:

        def update() -> None:
            if self.stopping:
                return

            self.rgb_widget.update_image(
                rgb
            )

            self.depth_widget.update_image(
                depth
            )

            # Current camera cloud:
            # update/re-add at the display rate.
            if (
                self.scene_widget.scene.has_geometry(
                    "current"
                )
            ):
                self.scene_widget.scene.remove_geometry(
                    "current"
                )

            if (
                current_pcd is not None
                and not current_pcd.is_empty()
            ):
                self.scene_widget.scene.add_geometry(
                    "current",
                    current_pcd,
                    self.pcd_material,
                    True,
                )

            # Persistent fused map.
            # Only replace when a new extraction
            # occurred.
            if (
                map_pcd is not None
                and id(map_pcd)
                != self.last_displayed_map_id
            ):

                if (
                    self.scene_widget.scene.has_geometry(
                        "map"
                    )
                ):
                    self.scene_widget.scene.remove_geometry(
                        "map"
                    )

                if not map_pcd.is_empty():
                    self.scene_widget.scene.add_geometry(
                        "map",
                        map_pcd,
                        self.map_material,
                        True,
                    )

                self.last_displayed_map_id = (
                    id(map_pcd)
                )

            tracking_text = (
                "OK"
                if stats["tracking"]
                else "LOST"
            )

            self.stats.text = (
                f"FPS: {stats['fps']:.1f}\n"
                f"Processing: "
                f"{stats['processing_ms']:.1f} ms\n"
                f"Frame points: "
                f"{stats['current_points']:,}\n"
                f"Map points: "
                f"{stats['map_points']:,}\n"
                f"Tracking: "
                f"{tracking_text}\n"
                f"Mode: "
                f"{stats['tracking_mode']}"
            )

            self.status.text = (
                "Scanning"
                if self.running
                else "Paused"
            ) + (
                f" | frame {stats['frame']}"
            )

        gui.Application.instance.post_to_main_thread(
            self.window,
            update,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    print(
        "========== OPEN3D =========="
    )

    print(
        "Open3D:",
        o3d.__version__,
    )

    print(
        "CUDA available:",
        o3d.core.cuda.is_available(),
    )

    print(
        "CUDA devices:",
        o3d.core.cuda.device_count(),
    )

    print(
        "Device:",
        DEVICE,
    )

    print()

    app = gui.Application.instance

    app.initialize()

    ScannerWindow()

    app.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
