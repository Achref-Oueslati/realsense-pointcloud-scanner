# RealSense 3D Scanner

Utilities for acquiring depth/color frames from an Intel RealSense camera and
creating point clouds with Open3D. The project includes a GUI scanner and a
CUDA-backed real-time point-cloud example.

## Run

Create and activate a local Python environment, install the project's camera,
Open3D, and GUI dependencies, then run:

```bash
./run_scanner.sh
```

The script expects a `.venv` directory in this project. The CUDA point-cloud
example is `src/realtime_pointcloud.py`; it requires a compatible CUDA-enabled
Open3D installation and GPU.

Camera recordings are stored under `recordings/`, which is ignored by Git.
Review recordings and camera metadata before sharing them.