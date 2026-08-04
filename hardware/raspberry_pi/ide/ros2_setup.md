# ROS 2 Humble Setup on Raspberry Pi

> **File:** `ide/ros2_setup.md`
> **Brief:** End-to-end guide for installing, building, and running ROS 2 Humble on Raspberry Pi 4/5, including workspace setup, rosdep, and common troubleshooting.
> **Author:** Autonomous Vehicle Team
> **Date:** 2025-01-30
> **License:** MIT

## 1. Prerequisites

* Raspberry Pi 4 (4 GB min) or Pi 5 (4 GB min)
* Ubuntu Server 22.04 LTS (64-bit) — **ROS 2 Humble officially supports 22.04**
* At least 16 GB free SD card space
* Internet connection (Ethernet preferred during install)
* Locale set:

```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

Verify:

```bash
locale    # should show en_US.UTF-8 everywhere
```

---

## 2. Install ROS 2 Humble

### 2.1 Add the OSRF repository

```bash
sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 2.2 Install ROS 2 packages

Two flavors:

| Variant        | Size after install | Contents                                  |
|----------------|-------------------|-------------------------------------------|
| `ros-base`     | ~750 MB            | Core rclcpp / rclpy / rmw / message packages |
| `desktop`      | ~2.0 GB            | + RViz, rqt, demos, demo nodes             |

For a Pi, **install `ros-base`** plus only the packages you need:

```bash
sudo apt update
sudo apt install -y \
    ros-humble-ros-base \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-image-transport \
    ros-humble-image-pipeline \
    ros-humble-diagnostic-updater \
    ros-humble-robot-localization \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-xacro \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-argcomplete
```

### 2.3 Initialize rosdep

```bash
sudo rosdep init
rosdep update            # run as your normal user
rosdep install --from-paths ~/ros2_ws/src --ignore-src -y -r
```

If `rosdep init` fails because `/etc/ros/rosdep/sources.list.d/20-default.list`
already exists, just delete it and retry.

### 2.4 Source ROS 2

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
ros2 --help
ros2 topic list        # empty but no error
```

---

## 3. Create the workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build --symlink-install --parallel-workers 2
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

**Pi 4 with 2 GB RAM:** use `--parallel-workers 1` and `--executor serial`
or you'll run out of memory on the first build.

```bash
colcon build \
    --symlink-install \
    --parallel-workers 1 \
    --executor serial \
    --event-handlers console_direct+
```

---

## 4. Create your packages

### 4.1 From the project templates

```bash
cd ~/ros2_ws/src
cp -r /path/to/autonomous_vehicle_hardware/raspberry_pi/ros2/av_perception .
cp -r /path/to/autonomous_vehicle_hardware/raspberry_pi/ros2/av_control .
cp -r /path/to/autonomous_vehicle_hardware/raspberry_pi/ros2/av_localization .
cp -r /path/to/autonomous_vehicle_hardware/raspberry_pi/ros2/av_navigation .
```

### 4.2 From scratch

```bash
cd ~/ros2_ws/src
# Python node
ros2 pkg create --build-type ament_python --license MIT \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs \
    av_perception

# C++ node
ros2 pkg create --build-type ament_cmake --license MIT \
    --dependencies rclcpp std_msgs sensor_msgs \
    av_motor_driver
```

### 4.3 Build a single package

```bash
cd ~/ros2_ws
colcon build --packages-select av_perception --symlink-install
source install/setup.bash
```

---

## 5. RMW / DDS tuning

ROS 2 Humble defaults to **Fast DDS**. On a constrained Wi-Fi network the
default discovery may drop packets. Two options:

### 5.1 Switch to Cyclone DDS (recommended for Pi)

```bash
sudo apt install -y ros-humble-rmw-cyclonedds-cpp
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
```

Place a `cyclonedds.xml` in `config/`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain id="any">
    <General>
      <NetworkInterfaceAddress>auto</NetworkInterfaceAddress>
      <AllowMulticast>spdp</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <Peers>
        <Peer address="10.0.0.10"/>  <!-- dashboard host -->
      </Peers>
    </Discovery>
    <Internal>
      <SocketReceiveBufferSize>10MB</SocketReceiveBufferSize>
    </Internal>
  </Domain>
</CycloneDDS>
```

```bash
echo 'export CYCLONEDDS_URI=$PWD/config/cyclonedds.xml' >> ~/.bashrc
```

### 5.2 Domain ID

To avoid crosstalk with other ROS 2 users on the same network, set a
non-default domain:

```bash
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
```

Pick a value between 0 and 101 (avoid 0 in production).

---

## 6. Lifecycle nodes

Perception / control / localization / navigation nodes are
`rclpy.lifecycle.LifecycleNode` subclasses. They expose states:

```
UNCONFIGURED → INACTIVE → ACTIVE → FINALIZED
```

Drive them with:

```bash
ros2 lifecycle set /perception configure
ros2 lifecycle set /perception activate
ros2 lifecycle list /perception
```

The bringup launch file does this automatically.

---

## 7. Quality of Service (QoS)

ROS 2 supports per-topic QoS. Match publisher / subscriber QoS or messages
won't flow. Common profiles used in this project:

| Topic                    | QoS profile              |
|--------------------------|--------------------------|
| `/camera/image_raw`      | `sensor_data` (BEST_EFFORT, KEEP_LAST 5) |
| `/lidar/scan`            | `sensor_data`            |
| `/imu/data`              | `sensor_data`            |
| `/gps/fix`               | `sensor_data`            |
| `/localization/odom`     | `reliable` (KEEP_LAST 10) |
| `/cmd_vel`               | `reliable` (KEEP_LAST 1)  |
| `/estop`                 | `reliable` (KEEP_ALL, durability=TRANSIENT_LOCAL) |

---

## 8. Run the bringup

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch av_bringup launch.py vehicle_id:=av-01
```

In another terminal:

```bash
ros2 topic list
ros2 topic echo /cmd_vel
ros2 run rqt_graph rqt_graph
```

---

## 9. `rosdep` issues

### 9.1 "rosdep cannot find ros-humble-foo"

```bash
sudo rm /etc/ros/rosdep/sources.list.d/20-default.list
sudo rosdep init
rosdep update
```

### 9.2 "OS not supported"

You must be on Ubuntu (Jammy/Jammy+). On Raspberry Pi OS (Bookworm) ROS 2
Humble is **not officially supported** — use the Docker container or
[mcgive/ros2-raspbian](https://github.com/mcgive/ros2-raspbian) at your own
risk.

### 9.3 Behind a proxy

```bash
sudo -E rosdep init
rosdep update --include-eol-distros
```

---

## 10. Common issues

| Symptom                                      | Likely cause                  | Fix                                            |
|----------------------------------------------|-------------------------------|------------------------------------------------|
| `ros2 topic list` hangs forever              | DDS discovery blocked         | Switch to Cyclone DDS (section 5.1)            |
| Build fails with `C++ compiler out of memory`| Pi 4 with 2 GB RAM            | Use `--parallel-workers 1 --executor serial`  |
| `ImportError: No module named rclpy._rclpy`  | Mixed Python versions         | Use system Python 3.10; rebuild the package    |
| `TF_REPEATED_DATA` warning                   | Multiple tf broadcasters      | Only localization_node should publish odom→base |
| Camera `/image_raw` not appearing            | QoS mismatch                  | Use `sensor_data` QoS on subscriber            |
| Cannot `ros2 launch` from systemd            | Environment not sourced       | Source `/opt/ros/humble/setup.bash` in unit     |
| Clock skew warning                           | NTP not synced                | `sudo systemctl enable --now systemd-timesyncd` |
| "Package 'av_perception' not found"          | Not sourced                   | `source ~/ros2_ws/install/setup.bash`          |

---

## 11. Auto-start on boot (systemd)

```bash
sudo tee /etc/systemd/system/av-ros2.service >/dev/null <<'EOF'
[Unit]
Description=Autonomous Vehicle ROS 2 bringup
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
ExecStart=/bin/bash -lc 'source /opt/ros/humble/setup.bash && \
  source /home/ubuntu/ros2_ws/install/setup.bash && \
  ros2 launch av_bringup launch.py vehicle_id:=av-01'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable av-ros2.service
sudo systemctl start av-ros2.service
journalctl -u av-ros2.service -f
```

---

## 12. Useful commands cheat sheet

```bash
# Discovery
ros2 daemon stop; ros2 daemon start
ros2 node list
ros2 topic list -t
ros2 service list
ros2 param list

# Inspect
ros2 topic echo /cmd_vel
ros2 topic hz /camera/image_raw
ros2 topic bw /camera/image_raw
ros2 topic info /cmd_vel -v      # shows QoS

# Lifecycle
ros2 lifecycle list /perception
ros2 lifecycle set /perception activate

# Bag (record for replay)
ros2 bag record -o run_20250130 /cmd_vel /imu/data /localization/odom
ros2 bag play run_20250130

# Clean rebuild
rm -rf build/ install/ log/
colcon build --symlink-install
```

---

## 13. References

* ROS 2 Humble docs: <https://docs.ros.org/en/humble/index.html>
* Colcon docs: <https://colcon.readthedocs.io/>
* Cyclone DDS: <https://github.com/eclipse-cyclonedds/cyclonedds>
* Nav2: <https://docs.nav2.org/>
