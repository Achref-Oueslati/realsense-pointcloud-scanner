#!/usr/bin/env python3

import time
import sys

import numpy as np
import open3d as o3d
import pyrealsense2 as rs


# ============================================================
# Configuration
# ============================================================

WIDTH = 848
HEIGHT = 480
FPS = 30

DEPTH_MAX_METERS = 5.0
VOXEL_SIZE = 0.01  # 1 cm

DEVICE = o3d.core.Device("CUDA:0")


# ============================================================
# Validation
# ============================================================

def validate_environment() -> None:
    print("========== ENVIRONMENT ==========")
    print("Python:", sys.version.split()[0])
    print("Open3D:", o3d.__version__)
    print("NumPy:", np.__version__)

    if not o3d.core.cuda.is_available():
        raise RuntimeError("Open3D CUDA is not available.")

    print("Open3D CUDA: available")
    print("CUDA devices:", o3d.core.cuda.device_count())
    print("Open3D device:", DEVICE)

    print()


# ============================================================
# RealSense setup
# ============================================================

def create_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        FPS,
    )

    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()

    depth_scale = depth_sensor.get_depth_scale()

    print("========== REALSENSE ==========")
    print(
        "Camera:",
        profile.get_device().get_info(rs.camera_info.name),
    )
    print(
        "Serial:",
        profile.get_device().get_info(rs.camera_info.serial_number),
    )
    print(
        "Firmware:",
        profile.get_device().get_info(rs.camera_info.firmware_version),
    )
    print("Depth scale:", depth_scale)

    return pipeline, profile, depth_scale


# ============================================================
# Intrinsics
# ============================================================

def get_intrinsic_tensor(profile) -> o3d.core.Tensor:

    depth_stream = profile.get_stream(
        rs.stream.depth
    ).as_video_stream_profile()

    intr = depth_stream.get_intrinsics()

    print("\n========== INTRINSICS ==========")
    print("Width :", intr.width)
    print("Height:", intr.height)
    print("fx    :", intr.fx)
    print("fy    :", intr.fy)
    print("ppx   :", intr.ppx)
    print("ppy   :", intr.ppy)

    return o3d.core.Tensor(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=o3d.core.Dtype.Float32,
        device=DEVICE,
    )


# ============================================================
# Point-cloud creation
# ============================================================

def create_cuda_pointcloud(
    depth_np: np.ndarray,
    color_np: np.ndarray,
    intrinsic: o3d.core.Tensor,
    depth_scale: float,
) -> o3d.t.geometry.PointCloud:

    depth = o3d.t.geometry.Image(
        o3d.core.Tensor(
            depth_np,
            dtype=o3d.core.Dtype.UInt16,
            device=DEVICE,
        )
    )

    color = o3d.t.geometry.Image(
        o3d.core.Tensor(
            color_np,
            dtype=o3d.core.Dtype.UInt8,
            device=DEVICE,
        )
    )

    rgbd = o3d.t.geometry.RGBDImage(
        color,
        depth,
    )

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic,
        depth_scale=1.0 / depth_scale,
        depth_max=DEPTH_MAX_METERS,
        stride=1,
    )

    return pcd


# ============================================================
# Main
# ============================================================

def main():

    validate_environment()

    pipeline = None

    try:
        pipeline, profile, depth_scale = create_pipeline()

        intrinsic = get_intrinsic_tensor(profile)

        # Warm-up.
        for _ in range(30):
            pipeline.wait_for_frames()

        print("\n========== STREAM ==========")
        print(f"Resolution: {WIDTH}x{HEIGHT}")
        print(f"FPS:        {FPS}")
        print(f"Max depth:  {DEPTH_MAX_METERS} m")
        print(f"Voxel size: {VOXEL_SIZE} m")
        print()

        frames_processed = 0
        total_processing_time = 0.0
        last_report = time.perf_counter()

        while True:

            frame_start = time.perf_counter()

            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # ------------------------------------------------
            # Point cloud creation on CUDA
            # ------------------------------------------------

            pcd = create_cuda_pointcloud(
                depth_image,
                color_image,
                intrinsic,
                depth_scale,
            )

            # ------------------------------------------------
            # CUDA voxel downsampling
            # ------------------------------------------------

            pcd = pcd.voxel_down_sample(VOXEL_SIZE)

            # Force synchronization so timing is meaningful.
            # Open3D CUDA operations may be asynchronous.
            if o3d.core.cuda.is_available():
                o3d.core.cuda.synchronize()

            processing_time = time.perf_counter() - frame_start

            total_processing_time += processing_time
            frames_processed += 1

            # ------------------------------------------------
            # Performance report
            # ------------------------------------------------

            now = time.perf_counter()

            if now - last_report >= 2.0:

                avg_processing_ms = (
                    total_processing_time
                    / frames_processed
                    * 1000.0
                )

                processing_fps = (
                    frames_processed
                    / total_processing_time
                    if total_processing_time > 0
                    else 0.0
                )

                print(
                    f"\r"
                    f"Frames: {frames_processed:6d} | "
                    f"Points: {len(pcd.point.positions):7d} | "
                    f"Processing: {avg_processing_ms:7.2f} ms | "
                    f"Processing FPS: {processing_fps:6.1f}",
                    end="",
                    flush=True,
                )

                last_report = now

    except KeyboardInterrupt:

        print("\n\nStopping...")

    finally:

        if pipeline is not None:
            pipeline.stop()

        print("RealSense pipeline stopped.")


if __name__ == "__main__":
    main()
