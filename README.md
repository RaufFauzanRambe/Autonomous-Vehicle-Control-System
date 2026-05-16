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
