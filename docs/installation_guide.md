# Autonomous Vehicle Control System — Installation Guide

**Version:** 2.0.0
**Last Updated:** 2026-03-04
**Applies To:** AVCS v0.1.0+

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Quick Install (Docker)](#2-quick-install-docker)
3. [Native Installation](#3-native-installation)
4. [Conda Environment](#4-conda-environment)
5. [Building from Source](#5-building-from-source)
6. [Configuration](#6-configuration)
7. [Troubleshooting](#7-troubleshooting)
8. [Verification](#8-verification)

---

## 1. System Requirements

### 1.1 Hardware Requirements

The AVCS platform has demanding real-time and GPU compute requirements. The following table summarizes minimum and recommended hardware configurations:

| Component       | Minimum                              | Recommended                              |
|-----------------|--------------------------------------|------------------------------------------|
| **CPU**         | Intel i7-10700K / AMD Ryzen 7 5800X | Intel i9-13900K / AMD Ryzen 9 7950X     |
| **CPU Cores**   | 8 cores                              | 16+ cores                                |
| **GPU**         | NVIDIA RTX 3060 (12 GB VRAM)        | NVIDIA RTX 4090 (24 GB VRAM)            |
| **RAM**         | 16 GB DDR4                           | 32–64 GB DDR5                            |
| **Storage**     | 50 GB SSD                            | 500 GB NVMe SSD                          |
| **Network**     | Gigabit Ethernet                     | 10 GbE (for multi-sensor data streams)  |

**Notes:**

- An NVIDIA GPU with CUDA Compute Capability 7.0+ (Volta or newer) is required for the perception pipeline. The GPU must have at least 12 GB of VRAM to run the multi-camera detection and LiDAR processing networks simultaneously.
- The CARLA simulator requires significant GPU resources; for headless simulation with rendering, at least 8 GB VRAM is needed. For off-screen rendering (`-RenderOffScreen`), 6 GB may suffice.
- SSD storage is strongly recommended because the LiDAR and camera data streams generate 500 MB/s–1 GB/s of raw data during recording sessions.
- If you plan to run CARLA and AVCS on the same machine, 64 GB of RAM is recommended.

### 1.2 Software Requirements

| Software            | Minimum Version | Notes                                              |
|---------------------|-----------------|----------------------------------------------------|
| **Operating System**| Ubuntu 22.04 LTS| 20.04 may work but is not officially supported     |
| **GCC**             | 11.0+           | Required for C++17 support                         |
| **CMake**           | 3.20+           | Required for modern CMake features                 |
| **Python**          | 3.10+           | Required for type hint syntax and match/case       |
| **ROS2**            | Humble Hawksbill| Desktop or ROS-Base install                        |
| **CUDA**            | 11.8+           | Required for GPU-accelerated perception            |
| **cuDNN**           | 8.6+            | Required for neural network inference              |
| **Eigen3**          | 3.4+            | Linear algebra library for C++ core                |
| **Docker**          | 20.10+          | Required for containerized deployment              |
| **NVIDIA Container Toolkit** | 1.13+  | Required for GPU passthrough in Docker             |
| **Git**             | 2.34+           | For cloning the repository                         |
| **Conda**           | 23.0+           | Optional, for isolated Python environment          |

### 1.3 NVIDIA Driver Requirements

The CUDA 11.8 toolkit requires NVIDIA driver version **520.61.05** or later. Check your current driver version:

```bash
nvidia-smi
```

If you need to update the driver:

```bash
# Add the NVIDIA driver PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install the recommended driver (e.g., 535)
sudo apt install nvidia-driver-535

# Reboot to apply
sudo reboot
```

---

## 2. Quick Install (Docker)

Docker is the recommended installation method for users who want a fully reproducible environment without manually installing all system dependencies. The AVCS project provides a multi-stage Dockerfile that builds the C++ core and Python extensions in a builder stage and packages them into a minimal runtime image with CUDA support.

### 2.1 Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker Engine
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group (log out and back in to apply)
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure the Docker daemon to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2.2 Clone and Build the Docker Image

```bash
# Clone the repository
git clone https://github.com/avcs/autonomous-vehicle-control.git
cd autonomous-vehicle-control

# Build the Docker image (this takes 15-30 minutes depending on your hardware)
docker build -t avcs:latest .

# Verify the image was built
docker images | grep avcs
```

### 2.3 Run the Container with GPU Passthrough

```bash
# Run the AVCS core container with full GPU access
docker run -it --rm --gpus all \
    --network host \
    -v $(pwd)/config:/avcs/config:ro \
    -v $(pwd)/data:/avcs/data \
    -v $(pwd)/logs:/avcs/logs \
    avcs:latest

# Run with a specific GPU device (useful for multi-GPU systems)
docker run -it --rm --gpus '"device=0"' \
    --network host \
    avcs:latest

# Run with X11 forwarding for visualization (CARLA, RViz2)
xhost +local:docker
docker run -it --rm --gpus all \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    avcs:latest
```

### 2.4 Run with Docker Compose

The `docker-compose.yml` file orchestrates multiple containers: the AVCS core, the CARLA simulator, SUMO traffic simulator, Redis message bus, web dashboard, ROS2 bridge, and data recorder.

```bash
# Start all services
docker-compose up --build -d

# Check the status of all services
docker-compose ps

# View logs from the AVCS core
docker-compose logs -f avcs-core

# Stop all services and remove volumes
docker-compose down -v

# Start only specific services (e.g., AVCS core and Redis)
docker-compose up -d avcs-core redis
```

The exposed ports are:

| Port(s)       | Service             | Description                     |
|---------------|---------------------|---------------------------------|
| 2000–2002     | CARLA Simulator     | RPC, streaming, and world ports |
| 5555–5557     | V2X DSRC            | V2V/V2I communication           |
| 6379          | Redis               | Message bus                     |
| 8080          | Web Dashboard       | System monitoring dashboard     |
| 9090          | ROS2 Bridge         | ROS2 websocket bridge           |

---

## 3. Native Installation

If you prefer to install the AVCS natively (without Docker), follow these steps in order. This approach gives you more control over the build configuration but requires manual dependency management.

### 3.1 System Dependency Installation

```bash
# Update the package index
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    gcc-11 \
    g++-11 \
    cmake \
    git \
    make \
    ninja-build \
    pkg-config \
    curl \
    wget \
    libssl-dev

# Set GCC 11 as the default compiler
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Verify the compiler version
gcc --version   # Should report 11.x
g++ --version   # Should report 11.x

# Install additional libraries
sudo apt install -y \
    libeigen3-dev \
    libboost-all-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv
```

### 3.2 ROS2 Humble Installation

```bash
# Set up the ROS2 apt repository
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble (ROS-Base is sufficient; use ros-humble-desktop for RViz2)
sudo apt update
sudo apt install -y ros-humble-ros-base

# Install additional ROS2 packages used by AVCS
sudo apt install -y \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    ros-humble-ackermann-msgs \
    ros-humble-visualization-msgs \
    python3-colcon-common-extensions

# Source ROS2 in your shell
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source /opt/ros/humble/setup.bash
```

### 3.3 CUDA and cuDNN Installation

```bash
# Download and install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add CUDA to your PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version   # Should report 11.8

# Install cuDNN 8 (for Ubuntu 22.04)
# Option A: Via apt (requires NVIDIA developer account and apt repo setup)
sudo apt install -y cudnn8

# Option B: Manual download from NVIDIA
# Download cuDNN 8.6+ for CUDA 11.x from https://developer.nvidia.com/cudnn
# Then extract and copy:
# tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
# sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
# sudo cp cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
# sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 3.4 Eigen3 Installation

```bash
# Eigen3 is typically available via apt on Ubuntu 22.04
sudo apt install -y libeigen3-dev

# Verify the installed version
pkg-config --modversion eigen3   # Should report 3.4.x

# If you need a newer version than what apt provides:
# git clone https://gitlab.com/libeigen/eigen.git
# cd eigen && mkdir build && cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
# make -j$(nproc) && sudo make install
```

### 3.5 Clone and Build the Project

```bash
# Clone the repository
git clone https://github.com/avcs/autonomous-vehicle-control.git
cd autonomous-vehicle-control

# Create the build directory and configure CMake
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/avcs-install \
    -DUSE_CUDA=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTING=ON

# Build the project (use all available cores)
make -j$(nproc)

# Install the built artifacts
make install

# Return to the project root
cd ..
```

### 3.6 Environment Setup

```bash
# Set up the library path
echo "export LD_LIBRARY_PATH=$HOME/avcs-install/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Set up the Python path
echo "export PYTHONPATH=$(pwd)/src:\$PYTHONPATH" >> ~/.bashrc

# Set the AVCS configuration directory
echo "export AVCS_CONFIG_PATH=$(pwd)/config" >> ~/.bashrc

# Set the ROS domain ID (must match across all AVCS nodes)
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc

# Source ROS2 workspace (if using colcon build)
cd autonomous-vehicle-control
colcon build --symlink-install --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DBUILD_PYTHON_BINDINGS=ON

echo "source $(pwd)/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 4. Conda Environment

The AVCS project ships an `environment.yml` file that defines a complete Conda environment with all Python dependencies, C++ build tools, and ROS2 packages from the RoboStack channel.

### 4.1 Install Conda

```bash
# Download and install Miniforge (recommended over Anaconda for conda-forge compatibility)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# Initialize conda for your shell
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc
```

### 4.2 Create the AVCS Conda Environment

```bash
# Navigate to the project directory
cd autonomous-vehicle-control

# Create the conda environment from the environment.yml file
conda env create -f environment.yml

# This will create an environment named 'avcs' with:
# - Python 3.10+
# - GCC/G++ 11+ (Linux cross-compiler)
# - CMake 3.20+ and Ninja
# - NumPy, SciPy, Eigen, OpenCV
# - ROS2 Humble packages from robostack-humble
# - PyTorch, ONNX Runtime, Open3D
# - Testing and code quality tools

# Verify the environment was created
conda env list
```

### 4.3 Activate and Use the Environment

```bash
# Activate the AVCS environment
conda activate avcs

# Verify key dependencies
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Install pip-only dependencies (already included via environment.yml, but can be re-installed)
pip install -r requirements.txt

# Deactivate when done
conda deactivate
```

### 4.4 Updating the Environment

If the `environment.yml` file is updated (e.g., new dependencies are added), update your existing environment:

```bash
conda env update -f environment.yml --prune
```

The `--prune` flag removes packages that are no longer listed in the environment file.

---

## 5. Building from Source

This section details all CMake configuration options, build targets, and install targets available when building the AVCS from source.

### 5.1 CMake Configuration Options

The AVCS CMakeLists.txt exposes the following options that control which components are built:

| Option                     | Default | Description                                       |
|----------------------------|---------|---------------------------------------------------|
| `BUILD_TESTING`            | ON      | Build GTest-based C++ unit tests                  |
| `BUILD_PYTHON_BINDINGS`    | OFF     | Build pybind11 Python bindings for the C++ core   |
| `BUILD_BENCHMARKS`         | OFF     | Build Google Benchmark performance benchmarks     |
| `USE_CUDA`                 | OFF     | Enable CUDA support for perception algorithms     |
| `USE_TBB`                  | OFF     | Enable Intel TBB for parallel algorithms          |

When the AVCS is built within a ROS2 workspace (i.e., `ament_cmake` is found automatically), the following are also enabled:

- An `avcs_ros2_node` executable is built and linked against `rclcpp` and ROS2 message types.
- The `ament_package()` macro is invoked, generating the ROS2 package manifest.

### 5.2 Release vs. Debug Builds

```bash
# Release build (optimized, no debug symbols)
mkdir build-release && cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Debug build (with debug symbols, no optimization, all warnings)
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# RelWithDebInfo (optimized with debug symbols — best for profiling)
mkdir build-reldeb && cd build-reldeb
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)

# MinSizeRel (optimized for binary size — useful for embedded targets)
mkdir build-minsize && cd build-minsize
cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel
make -j$(nproc)
```

The CMakeLists.txt applies the following compiler flags automatically:

- **Release:** `-O3 -DNDEBUG`
- **Debug:** `-g -O0 -Wall -Wextra -Wpedantic`

### 5.3 Building With/Without CUDA

```bash
# With CUDA (requires CUDA 11.8+ installed and an NVIDIA GPU)
cmake .. -DUSE_CUDA=ON

# Without CUDA (CPU-only perception; some neural network features will be unavailable)
cmake .. -DUSE_CUDA=OFF
```

When CUDA is enabled, the CMakeLists.txt adds the `AVCS_USE_CUDA` compile definition and links against the CUDA libraries. The perception module will use GPU-accelerated TensorRT inference for object detection. Without CUDA, only CPU-based fallback algorithms are available.

### 5.4 Building With/Without Python Bindings

```bash
# With Python bindings (requires pybind11)
cmake .. -DBUILD_PYTHON_BINDINGS=ON

# Without Python bindings (C++-only build)
cmake .. -DBUILD_PYTHON_BINDINGS=OFF
```

Python bindings are built using pybind11. If `pybind11` is not found on the system, CMake will attempt to locate it via `pybind11-config`. If neither method succeeds, a warning is printed and the bindings are silently disabled.

When enabled, a shared library `_cpp_core.<suffix>.so` is produced and installed to the `avcs` Python package directory, allowing Python code to call C++ core functions directly.

### 5.5 Building With/Without ROS2

ROS2 integration is automatically enabled when `ament_cmake` is found. This happens when you build the project within a ROS2 workspace using `colcon build`.

```bash
# Build with ROS2 (within a ROS2 workspace)
colcon build --symlink-install --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON

# Build without ROS2 (standalone C++ executable only)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

When ROS2 is enabled, the following additional targets are built:

- `avcs_ros2_node` — A ROS2 node that publishes/ subscribes to AVCS topics
- `ament_package()` — Generates the ROS2 package descriptor

### 5.6 Build Targets

The following build targets are produced:

| Target              | Type        | Description                                       |
|---------------------|-------------|---------------------------------------------------|
| `avcs_core`         | Static lib  | Core C++ library (sensor fusion, localization, SLAM, etc.) |
| `avcs_core_bin`     | Executable  | Standalone AVCS binary (`avcs_core`)              |
| `avcs_ros2_node`    | Executable  | ROS2 node (only when ROS2 is found)               |
| `_cpp_core`         | Shared lib  | Python bindings module (only with `BUILD_PYTHON_BINDINGS=ON`) |
| `avcs_tests`        | Executable  | C++ unit test runner (only with `BUILD_TESTING=ON`) |
| `avcs_benchmarks`   | Executable  | Performance benchmarks (only with `BUILD_BENCHMARKS=ON`) |

### 5.7 Install Targets

```bash
# Install to default prefix (/usr/local)
cmake --install build

# Install to a custom prefix
cmake --install build --prefix $HOME/avcs-install
```

The install layout is:

```
<prefix>/
├── bin/
│   └── avcs_core            # Standalone executable
├── lib/
│   ├── libavcs_core.a       # Static library
│   └── avcs/
│       └── _cpp_core.so     # Python bindings
├── include/
│   └── avcs/
│       └── *.hpp            # Public headers
└── config/
    └── *.yaml               # Default configuration files
```

---

## 6. Configuration

All AVCS configuration is managed through YAML files in the `config/` directory. Each file controls a specific subsystem. This section explains every configuration file and its key parameters.

### 6.1 system.yaml — System-Wide Settings

The `system.yaml` file controls global settings that apply across all modules.

| Parameter                    | Type   | Default    | Description                                           |
|------------------------------|--------|------------|-------------------------------------------------------|
| `loop_rate_hz`               | int    | 100        | Main control loop frequency (Hz)                      |
| `perception_rate_hz`         | int    | 20         | Perception pipeline frequency (Hz)                    |
| `localization_rate_hz`       | int    | 10         | Localization update frequency (Hz)                    |
| `planning_rate_hz`           | int    | 10         | Planning cycle frequency (Hz)                         |
| `control_rate_hz`            | int    | 100        | Control loop frequency (Hz)                           |
| `log_level`                  | string | "info"     | Global log level: trace, debug, info, warn, error     |
| `log_directory`              | string | "./logs"   | Directory for log file output                         |
| `ros_domain_id`              | int    | 42         | DDS domain ID for ROS2 communication                  |
| `topics.perception_output`   | string | "/avcs/perception"    | Perception output topic name            |
| `topics.localization_output` | string | "/avcs/localization"  | Localization output topic name          |
| `topics.planning_output`     | string | "/avcs/planning"      | Planning output topic name              |
| `topics.control_output`      | string | "/avcs/control"       | Control command output topic name       |
| `safety.watchdog_timeout_ms` | int    | 200        | Watchdog heartbeat timeout (ms)                       |
| `safety.emergency_decel`     | float  | 8.0        | Maximum emergency deceleration (m/s^2)                |

### 6.2 perception.yaml — Perception Pipeline

| Parameter                       | Type   | Default     | Description                                       |
|---------------------------------|--------|-------------|---------------------------------------------------|
| `detector.model_path`           | string | "models/yolox.onnx"   | Path to object detection model           |
| `detector.confidence_threshold` | float  | 0.5         | Minimum detection confidence to publish            |
| `detector.nms_threshold`        | float  | 0.3         | NMS IoU threshold for vehicles                    |
| `detector.nms_threshold_ped`    | float  | 0.1         | NMS IoU threshold for pedestrians                 |
| `tracker.max_coast_frames`      | int    | 5           | Frames to coast a track without measurement       |
| `tracker.gating_threshold`      | float  | 9.49        | Mahalanobis distance gating threshold             |
| `fusion.mode`                   | string | "late"      | Fusion mode: "late" or "hybrid"                   |
| `fusion.hybrid_min_distance`    | float  | 2.0         | Switch to hybrid when inter-object dist < this    |
| `lidar.ground_removal.method`   | string | "ransac"    | Ground removal: "ransac" or "multi_region"        |
| `lidar.ground_removal.distance` | float  | 0.15        | RANSAC inlier distance threshold (m)              |
| `lidar.voxel_resolution`        | float  | 0.1         | Voxel grid resolution (m)                         |
| `camera.undistort`              | bool   | true        | Apply lens distortion correction                  |
| `camera.bev_method`             | string | "lss"       | BEV transform: "lss" or "homography"              |

### 6.3 localization.yaml — Localization and SLAM

| Parameter                        | Type   | Default   | Description                                       |
|----------------------------------|--------|-----------|---------------------------------------------------|
| `method`                         | string | "ekf"     | Localization method: "ekf" or "slam"              |
| `ekf.process_noise_position`     | float  | 0.1       | Process noise for position states                 |
| `ekf.process_noise_velocity`     | float  | 0.5       | Process noise for velocity states                 |
| `ekf.process_noise_attitude`     | float  | 0.05      | Process noise for attitude states                 |
| `ekf.gps_noise_xy`               | float  | 0.5       | GPS measurement noise in X/Y (m)                  |
| `ekf.gps_noise_z`                | float  | 2.0       | GPS measurement noise in Z (m)                    |
| `ekf.gps_dop_threshold`          | float  | 5.0       | Max DOP for GPS update acceptance                  |
| `ekf.gps_min_satellites`         | int    | 6         | Min satellite count for GPS update acceptance      |
| `ndt.resolution`                 | float  | 1.0       | NDT voxel grid resolution (m)                     |
| `ndt.max_iterations`             | int    | 30        | Maximum NDT convergence iterations                |
| `ndt.fitness_threshold`          | float  | 2.0       | Max fitness score for accepting NDT result        |
| `slam.icp_max_iterations`        | int    | 30        | ICP convergence iterations                        |
| `slam.icp_convergence_eps`       | float  | 1e-6      | ICP convergence epsilon                           |
| `slam.keyframe_translation`      | float  | 1.0       | Min translation to create a new keyframe (m)      |
| `slam.keyframe_rotation`         | float  | 5.0       | Min rotation to create a new keyframe (deg)       |
| `slam.loop_closure_method`       | string | "scan_context" | Loop closure detection method                |

### 6.4 planning.yaml — Path and Behavior Planning

| Parameter                        | Type   | Default   | Description                                       |
|----------------------------------|--------|-----------|---------------------------------------------------|
| `path_planner.method`            | string | "astar"   | Global planner: "astar" or "lattice"              |
| `astar.weight_distance`          | float  | 1.0       | Distance weight in A* cost function               |
| `astar.weight_time`              | float  | 0.5       | Travel time weight                                |
| `astar.weight_lane_change`       | float  | 3.0       | Lane change penalty                               |
| `astar.weight_turn`              | float  | 2.0       | Turn penalty                                      |
| `behavior.default_speed`         | float  | 13.9      | Default target speed (m/s), ~50 km/h              |
| `behavior.following_time_gap`    | float  | 1.5       | Minimum time gap for car following (s)            |
| `behavior.lane_change_duration`  | float  | 3.0       | Minimum lane change duration (s)                  |
| `behavior.max_lateral_accel`     | float  | 1.5       | Maximum lateral acceleration during lane change   |
| `behavior.emergency_ttc`         | float  | 1.5       | Time-to-collision threshold for emergency stop    |
| `trajectory.longitudinal_jerk_limit` | float | 4.0    | Max longitudinal jerk (m/s^3)                     |
| `trajectory.lateral_accel_limit` | float  | 2.0       | Max lateral acceleration (m/s^2)                  |
| `trajectory.lateral_clearance_static` | float | 0.3   | Min lateral clearance from static objects (m)     |
| `trajectory.lateral_clearance_dynamic` | float | 0.5  | Min lateral clearance from dynamic objects (m)    |

### 6.5 control.yaml — Motion Control

| Parameter                        | Type   | Default   | Description                                       |
|----------------------------------|--------|-----------|---------------------------------------------------|
| `lateral_controller`             | string | "stanley" | Lateral controller: "pid", "stanley", "pure_pursuit", or "mpc" |
| `longitudinal_controller`        | string | "pid"     | Longitudinal controller: "pid" or "mpc"           |
| `pid_speed.kp`                   | float  | 1.0       | Speed PID proportional gain                       |
| `pid_speed.ki`                   | float  | 0.1       | Speed PID integral gain                           |
| `pid_speed.kd`                   | float  | 0.05      | Speed PID derivative gain                         |
| `pid_speed.anti_windup`          | float  | 5.0       | Integral anti-windup limit                        |
| `stanley.gain_heading`           | float  | 0.5       | Stanley heading error gain                        |
| `stanley.gain_cross_track`       | float  | 0.3       | Stanley cross-track error gain                    |
| `stanley.softening_constant`     | float  | 1.0       | Softening constant to avoid division by zero      |
| `pure_pursuit.min_lookahead`     | float  | 3.0       | Minimum lookahead distance (m)                    |
| `pure_pursuit.max_lookahead`     | float  | 20.0      | Maximum lookahead distance (m)                    |
| `pure_pursuit.lookahead_gain`    | float  | 0.5       | Speed-dependent lookahead gain                    |
| `mpc.horizon`                    | int    | 20        | MPC prediction horizon steps                      |
| `mpc.dt`                         | float  | 0.1       | MPC time step (s)                                 |
| `mpc.max_steer_rate`             | float  | 0.5       | Maximum steering rate (rad/s)                     |

### 6.6 vehicle.yaml — Vehicle Parameters

| Parameter                        | Type   | Default   | Description                                       |
|----------------------------------|--------|-----------|---------------------------------------------------|
| `vehicle.mass`                   | float  | 1800.0    | Vehicle mass (kg)                                 |
| `vehicle.length`                 | float  | 4.8       | Vehicle length (m)                                |
| `vehicle.width`                  | float  | 1.9       | Vehicle width (m)                                 |
| `vehicle.height`                 | float  | 1.5       | Vehicle height (m)                                |
| `vehicle.wheelbase`              | float  | 2.8       | Wheelbase (m)                                     |
| `vehicle.front_overhang`         | float  | 0.9       | Front overhang (m)                                |
| `vehicle.rear_overhang`          | float  | 1.1       | Rear overhang (m)                                 |
| `vehicle.max_speed`              | float  | 55.0      | Maximum speed (m/s), ~200 km/h                   |
| `vehicle.max_accel`              | float  | 3.0       | Maximum longitudinal acceleration (m/s^2)         |
| `vehicle.max_decel`              | float  | -8.0      | Maximum deceleration (m/s^2)                      |
| `vehicle.max_steer_angle`        | float  | 0.6       | Maximum steering angle (rad)                      |
| `sensors.lidar.position`         | list   | [0.0, 0.0, 2.0] | LiDAR position on the vehicle (m)          |
| `sensors.camera.front.position`  | list   | [1.5, 0.0, 1.2] | Front camera position (m)                  |
| `sensors.gps.position`           | list   | [-0.5, 0.0, 1.0] | GPS antenna position (m)                   |

### 6.7 simulation.yaml — Simulator Configuration

| Parameter                        | Type   | Default          | Description                                |
|----------------------------------|--------|------------------|--------------------------------------------|
| `carla.host`                     | string | "localhost"      | CARLA server hostname                      |
| `carla.port`                     | int    | 2000             | CARLA RPC port                             |
| `carla.streaming_port`           | int    | 2001             | CARLA streaming port                       |
| `carla.town`                     | string | "Town03"         | CARLA map name                             |
| `carla.sync_mode`                | bool   | true             | Run CARLA in synchronous mode              |
| `carla.delta_seconds`            | float  | 0.05             | Simulation time step (s)                   |
| `sumo.config_path`               | string | ""               | Path to SUMO .sumocfg file                 |
| `sumo.remote_port`               | int    | 4000             | SUMO TraCI remote port                     |
| `sumo.step_length`               | float  | 0.05             | SUMO simulation step length (s)            |
| `sumo.num_clients`               | int    | 1                | Number of TraCI clients                    |

---

## 7. Troubleshooting

### 7.1 CMake Can't Find Eigen3

**Symptoms:**
```
CMake Error at CMakeLists.txt:35 (find_package):
  Could not find a package configuration file provided by "Eigen3" with any
  of the following names:
    Eigen3Config.cmake
    eigen3-config.cmake
```

**Solutions:**

1. Install Eigen3 via apt:
   ```bash
   sudo apt install -y libeigen3-dev
   ```

2. If Eigen3 is installed but CMake cannot find it, specify the path manually:
   ```bash
   cmake .. -DEigen3_DIR=/usr/lib/cmake/eigen3
   ```

3. If you built Eigen3 from source, ensure the install prefix is in CMake's search path:
   ```bash
   cmake .. -DCMAKE_PREFIX_PATH=/usr/local
   ```

4. Verify the installed Eigen3 version meets the 3.4+ requirement:
   ```bash
   pkg-config --modversion eigen3
   ```

### 7.2 CUDA Version Mismatch

**Symptoms:**
```
CMake Error at CMakeLists.txt:48 (find_package):
  Could not find a configuration file for package "CUDA" that is compatible
  with requested version "11.8".
```

**Or at runtime:**
```
RuntimeError: CUDA version mismatch: driver supports 11.4 but toolkit requires 11.8
```

**Solutions:**

1. Verify your NVIDIA driver supports CUDA 11.8+:
   ```bash
   nvidia-smi
   # The "CUDA Version" shown in the header must be >= 11.8
   ```

2. If the driver is too old, update it:
   ```bash
   sudo apt install -y nvidia-driver-535
   sudo reboot
   ```

3. Verify the CUDA toolkit version:
   ```bash
   nvcc --version
   ```

4. If you have multiple CUDA versions installed, ensure the correct one is on your PATH:
   ```bash
   export PATH=/usr/local/cuda-11.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
   ```

5. If you do not need CUDA, disable it in the build:
   ```bash
   cmake .. -DUSE_CUDA=OFF
   ```

### 7.3 ROS2 Build Failures

**Symptoms:**
```
CMake Error: Could not find ament_cmake
```
or
```
ModuleNotFoundError: No module named 'rclpy'
```

**Solutions:**

1. Ensure ROS2 Humble is sourced before building:
   ```bash
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   ```

2. If building inside a Docker container, the Dockerfile already installs `ros-humble-ros-base`. Ensure the container has access to the ROS2 apt repository.

3. For Conda-based builds, make sure the `robostack-humble` channel is in your environment:
   ```yaml
   channels:
     - robostack-humble
     - conda-forge
   ```

4. If `colcon` is not found:
   ```bash
   sudo apt install -y python3-colcon-common-extensions
   ```

5. Clean and rebuild:
   ```bash
   rm -rf build/ install/ log/
   colcon build --symlink-install
   ```

### 7.4 Python Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'avcs'
```
or
```
ImportError: libavcs_core.so: cannot open shared object file
```

**Solutions:**

1. Set the `PYTHONPATH` to include the AVCS source directory:
   ```bash
   export PYTHONPATH=/path/to/autonomous-vehicle-control/src:$PYTHONPATH
   ```

2. If the C++ extension module fails to load, ensure the library path is set:
   ```bash
   export LD_LIBRARY_PATH=/path/to/avcs-install/lib:$LD_LIBRARY_PATH
   ```

3. Rebuild the Python bindings:
   ```bash
   mkdir build && cd build
   cmake .. -DBUILD_PYTHON_BINDINGS=ON
   make -j$(nproc)
   ```

4. Install the Python package in editable mode:
   ```bash
   pip install -e .
   ```

5. Verify all Python dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

### 7.5 GPU Driver Issues

**Symptoms:**
```
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
```
or
```
RuntimeError: Found no NVIDIA driver on your system
```

**Solutions:**

1. Verify the NVIDIA driver is loaded:
   ```bash
   lsmod | grep nvidia
   nvidia-smi
   ```

2. If the driver is not loaded, try reloading it:
   ```bash
   sudo modprobe nvidia
   ```

3. For Docker, ensure the NVIDIA Container Toolkit is installed and configured:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

4. Check for driver/kernel version mismatches (common after a kernel update without DKMS):
   ```bash
   dkms status
   # If nvidia is listed but not installed, rebuild:
   sudo dkms autoinstall
   sudo reboot
   ```

5. If you have Secure Boot enabled, the NVIDIA kernel modules may be blocked. Either disable Secure Boot in your BIOS or sign the kernel modules.

---

## 8. Verification

After completing the installation, follow these steps to verify that the AVCS is correctly installed and operational.

### 8.1 Verify C++ Core

```bash
# Run the standalone AVCS executable
./build/avcs_core --config config/system.yaml --mode autonomous --log-level info

# Expected output:
# [INFO] AVCS v0.1.0 starting...
# [INFO] Configuration loaded from config/system.yaml
# [INFO] Localization module initialized (EKF mode)
# [INFO] Perception module initialized
# [INFO] Planning module initialized
# [INFO] Control module initialized
# [INFO] Real-time engine started at 100 Hz
```

### 8.2 Run C++ Unit Tests

```bash
# Run all GTest-based tests
cd build && ctest --output-on-failure -j$(nproc)

# Expected output:
# Test project /path/to/autonomous-vehicle-control/build
#     Start 1: SensorFusionTest
#     Start 2: LocalizationTest
#     Start 3: SLAMTest
#     ...
# 100% tests passed, 0 tests failed out of N
```

Or run individual test binaries:

```bash
./build/avcs_tests --gtest_filter="LocalizationTest.*"
```

### 8.3 Run Python Unit Tests

```bash
# Run all Python tests with coverage reporting
python3 -m pytest test/ -v --cov=src --cov-report=term

# Expected output:
# test/unit/test_pid_controller.py PASSED
# test/unit/test_ekf_localizer.py PASSED
# ...
# N passed, 0 failed
# Coverage: XX%
```

### 8.4 Launch the CARLA Simulation

```bash
# Terminal 1: Start the CARLA simulator
docker run -p 2000-2002:2000-2002 --gpus all \
    carlasim/carla:0.9.15 \
    /bin/bash ./CarlaUE4.sh -RenderOffScreen

# Terminal 2: Launch AVCS with CARLA
source install/setup.bash
ros2 launch avcs carla_simulation.launch.py \
    carla_host:=localhost \
    carla_port:=2000 \
    town:=Town03 \
    sync_mode:=true
```

### 8.5 Quick Smoke Test with Makefile

```bash
# Build everything and run all tests in one command
make all && make test

# If all tests pass, the installation is verified
```

### 8.6 Verify Python and C++ Integration

```bash
# Test that the Python bindings work
python3 -c "
from avcs.control.pid_controller import PIDController
ctrl = PIDController(kp=1.0, ki=0.1, kd=0.05)
print(f'PID Controller created: kp={ctrl.kp}, ki={ctrl.ki}, kd={ctrl.kd}')
print('Python bindings: OK')
"

# Test that the C++ core module loads
python3 -c "
try:
    import _cpp_core
    print('C++ core module loaded successfully')
    print(f'AVCS version: {_cpp_core.version()}')
except ImportError as e:
    print(f'C++ core import failed: {e}')
    print('This is expected if BUILD_PYTHON_BINDINGS=OFF was used')
"
```

---

## Additional Resources

- **System Architecture:** [docs/architecture/system_architecture.md](architecture/system_architecture.md)
- **Developer Guide:** [docs/developer_guide.md](developer_guide.md)
- **API Documentation:** [docs/api_documentation/api_documentation.md](api_documentation/api_documentation.md)
- **ROS2 Humble Documentation:** https://docs.ros.org/en/humble/
- **CARLA Simulator Documentation:** https://carla.readthedocs.io/
- **Eigen3 Documentation:** https://eigen.tuxfamily.org/
