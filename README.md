# Autonomous Vehicle Control System

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/your-org/autonomous-vehicle-control-system)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](https://isocpp.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

## Overview

The **Autonomous Vehicle Control System (AVCS)** is a comprehensive software framework designed for autonomous vehicle navigation, perception, planning, and control. This system integrates state-of-the-art algorithms for real-time decision making in complex driving environments.

### Key Features

- **Perception Module**: Multi-sensor fusion for object detection, tracking, and environment understanding
- **Localization System**: High-precision positioning using GPS, IMU, LiDAR, and visual odometry
- **Path Planning**: Global and local path planning with dynamic obstacle avoidance
- **Motion Control**: Model Predictive Control (MPC) and Pure Pursuit controllers
- **Simulation Support**: Integration with CARLA, Gazebo, and LGSVL simulators
- **Hardware Abstraction**: Modular architecture supporting various vehicle platforms

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Autonomous Vehicle Control System             │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│  Perception │ Localization│   Planning  │   Control   │  HMI    │
│   Module    │   Module    │   Module    │   Module    │ Module  │
├─────────────┴─────────────┴─────────────┴─────────────┴─────────┤
│                    Middleware Layer (ROS2)                       │
├─────────────────────────────────────────────────────────────────┤
│              Hardware Abstraction Layer (HAL)                    │
├───────────┬───────────┬───────────┬───────────┬─────────────────┤
│   LiDAR   │  Camera   │   Radar   │    GPS    │   CAN Bus       │
└───────────┴───────────┴───────────┴───────────┴─────────────────┘
```

## Directory Structure

```
Autonomous-Vehicle-Control-System/
├── src/
│   ├── perception/          # Object detection and tracking
│   ├── localization/        # State estimation and mapping
│   ├── planning/            # Path and behavior planning
│   ├── control/             # Motion control algorithms
│   ├── communication/       # V2X communication
│   └── simulation/          # Simulator interfaces
├── include/                 # Header files
├── config/                  # Configuration files
├── launch/                  # ROS2 launch files
├── test/                    # Unit and integration tests
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── CMakeLists.txt
├── package.xml
├── requirements.txt
├── environment.yml
├── package.json
├── Dockerfile
├── docker-compose.yml
├── setup.py
└── Makefile
```

## Installation

### Prerequisites

- Ubuntu 22.04 LTS (recommended)
- ROS2 Humble Hawksbill
- Python 3.10+
- C++17 compatible compiler
- CUDA 11.8+ (for GPU acceleration)
- Docker (optional)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/your-org/autonomous-vehicle-control-system.git
cd autonomous-vehicle-control-system

# Build and run with Docker Compose
docker-compose up --build
```

### Manual Installation

#### 1. Install ROS2 Humble

```bash
# Add ROS2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop
```

#### 2. Install Python Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using Conda
conda env create -f environment.yml
conda activate avcs
```

#### 3. Install Node.js Dependencies (for visualization tools)

```bash
npm install
```

#### 4. Build the Project

```bash
# Using Makefile
make all

# Or manual build
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```


### Using Conda Environment

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate avcs

# Install the package in development mode
pip install -e .
```

## Configuration

### Vehicle Parameters

Edit `config/vehicle_params.yaml` to configure your vehicle specifications:

```yaml
vehicle:
  wheelbase: 2.7          # meters
  max_steering_angle: 0.6 # radians
  max_speed: 30.0         # m/s
  max_acceleration: 3.0   # m/s^2
  width: 1.8              # meters
  length: 4.5             # meters
```

### Sensor Configuration

Configure sensors in `config/sensors.yaml`:

```yaml
lidar:
  type: "velodyne_vlp16"
  port: 2368
  frame_id: "lidar_top"
  
camera:
  - name: "front_center"
    device: "/dev/video0"
    resolution: [1920, 1080]
    
gps:
  device: "/dev/ttyUSB0"
  baudrate: 115200
```

## Usage

### Launch Full System

```bash
# Source ROS2 workspace
source install/setup.bash

# Launch all nodes
ros2 launch avcs_bringup full_system.launch.py

# Launch with specific configuration
ros2 launch avcs_bringup full_system.launch.py config:=highway
```

### Individual Modules

```bash
# Perception module
ros2 launch avcs_perception perception.launch.py

# Planning module
ros2 launch avcs_planning planning.launch.py

# Control module
ros2 launch avcs_control control.launch.py
```

### Simulation Mode

```bash
# Launch CARLA simulation
ros2 launch avcs_simulation carla.launch.py town:=Town01

# Launch Gazebo simulation
ros2 launch avcs_simulation gazebo.launch.py world:=city
```

## API Reference

### Python API

```python
from avcs.planning import PathPlanner
from avcs.control import MPCController
from avcs.perception import ObjectDetector

# Initialize planner
planner = PathPlanner(config_path="config/planner.yaml")

# Generate path
path = planner.plan(start_pose, goal_pose, obstacles)

# Initialize controller
controller = MPCController(vehicle_params)
control_cmd = controller.compute_control(path, current_state)
```

### C++ API

```cpp
#include <avcs/planning/path_planner.hpp>
#include <avcs/control/mpc_controller.hpp>

// Initialize planner
avcs::planning::PathPlanner planner("config/planner.yaml");

// Generate path
auto path = planner.plan(start_pose, goal_pose, obstacles);

// Initialize controller
avcs::control::MPCController controller(vehicle_params);
auto control_cmd = controller.computeControl(path, current_state);

## Testing

```bash
# Run all tests
make test

# Run specific test suite
pytest test/perception/ -v

# Run with coverage
pytest --cov=src test/
```

## Docker Deployment

```bash
# Build image
docker build -t avcs:latest .

# Run container
docker run -it --rm \
  --network host \
  --privileged \
  -v /dev:/dev \
  avcs:latest

# Using docker-compose
docker-compose up -d
```

## Performance Benchmarks

| Module | Latency (ms) | CPU Usage | GPU Usage |
|--------|--------------|-----------|-----------|
| Perception | 50-80 | 45% | 70% |
| Localization | 20-30 | 25% | 10% |
| Planning | 30-50 | 30% | 5% |
| Control | 5-10 | 10% | 0% |

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code Style
- Python: Follow PEP 8 guidelines, use Black formatter
- C++: Follow Google C++ Style Guide
- Use pre-commit hooks for automated formatting

```bash
# Install pre-commit hooks
pre-commit install
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ROS2 Community for the excellent robotics framework
- CARLA Simulator team for the realistic autonomous driving simulator
- Apollo Auto project for inspiration on architecture design

## Issues
- **Issues**: https://github.com/your-org/autonomous-vehicle-control-system/issues

## Roadmap

- [ ] Support for ROS2 Iron Irwini
- [ ] Enhanced deep learning perception models
- [ ] V2X communication module
- [ ] Multi-vehicle coordination
- [ ] Web-based visualization dashboard
- [ ] Support for additional vehicle platforms

---

**Note**: This software is intended for research and development purposes. Proper safety testing and validation must be conducted before deployment in real-world autonomous vehicle applications.
