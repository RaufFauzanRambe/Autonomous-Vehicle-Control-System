#!/usr/bin/env bash
# File:        bash/install.sh
# Brief:       Install ALL dependencies for the Raspberry Pi platform
#              of the Autonomous Vehicle Control System. Idempotent:
#              safe to re-run. Installs apt packages, Python deps,
#              ROS 2 Humble, Docker, GPIO permissions, and udev rules.
# Author:      Autonomous Vehicle Team
# Date:        2025-01-30
# License:     MIT

set -euo pipefail

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="/tmp/av_install_$(date +%Y%m%d_%H%M%S).log"

# Ros distro
ROS_DISTRO="${ROS_DISTRO:-humble}"

# Telemetry
log()   { echo -e "\033[1;32m[install]\033[0m $*" | tee -a "$LOG_FILE"; }
warn()  { echo -e "\033[1;33m[install:WARN]\033[0m $*" | tee -a "$LOG_FILE" >&2; }
err()   { echo -e "\033[1;31m[install:ERR]\033[0m $*" | tee -a "$LOG_FILE" >&2; }
die()   { err "$*"; exit 1; }

trap 'err "Install failed at line $LINENO. See $LOG_FILE";' ERR

# ----------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------
[[ "$(id -u)" -eq 0 ]] || die "Please run with sudo: sudo $0"

if ! command -v apt-get &>/dev/null; then
    die "This script requires apt-get (Debian/Ubuntu/Raspberry Pi OS)."
fi

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "armv7l" ]]; then
    warn "This script is intended for ARM (Pi). Detected: $(uname -m)"
    warn "Continuing anyway — some packages may fail."
fi

log "Logging to $LOG_FILE"
log "Detected arch: $(uname -m)"
log "Detected OS:   $(. /etc/os-release && echo "$PRETTY_NAME")"

# ----------------------------------------------------------------------
# Step 1: apt update + base packages
# ----------------------------------------------------------------------
log "Step 1/7: apt update + base packages"
apt-get update -y
apt-get install -y \
    curl wget gnupg lsb-release software-properties-common \
    ca-certificates apt-transport-https \
    build-essential cmake ninja-build git \
    python3 python3-pip python3-venv python3-dev python3-wheel \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    i2c-tools spi-tools \
    libgpiod-dev gpiod \
    libv4l-dev v4l-utils \
    libcamera-apps python3-picamera2 \
    mosquitto mosquitto-clients \
    libserial-dev \
    htop vim less tmux rsync \
    udev usbutils

# ----------------------------------------------------------------------
# Step 2: Enable hardware interfaces
# ----------------------------------------------------------------------
log "Step 2/7: Enable I2C / SPI / Serial"
if command -v raspi-config &>/dev/null; then
    raspi-config nonint do_i2c  0 || true
    raspi-config nonint do_spi  0 || true
    raspi-config nonint do_serial 2 || true   # 2 = enable serial, disable shell
else
    log "raspi-config not found — manually enabling via device tree"
    if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt; then
        echo "dtparam=i2c_arm=on"  >> /boot/config.txt
    fi
    if ! grep -q "^dtparam=spi=on" /boot/config.txt; then
        echo "dtparam=spi=on"      >> /boot/config.txt
    fi
    if ! grep -q "^enable_uart=1" /boot/config.txt; then
        echo "enable_uart=1"       >> /boot/config.txt
    fi
fi

# Load modules immediately
modprobe i2c-dev 2>/dev/null || true
modprobe spidev  2>/dev/null || true

# ----------------------------------------------------------------------
# Step 3: User groups
# ----------------------------------------------------------------------
log "Step 3/7: User groups"
REAL_USER="${SUDO_USER:-$USER}"
usermod -aG dialout  "$REAL_USER" || true
usermod -aG i2c      "$REAL_USER" || true
usermod -aG spi      "$REAL_USER" || true
usermod -aG gpio     "$REAL_USER" || true
usermod -aG video    "$REAL_USER" || true
usermod -aG docker   "$REAL_USER" || true
usermod -aG plugdev  "$REAL_USER" || true
log "User '$REAL_USER' added to: dialout i2c spi gpio video docker plugdev"

# ----------------------------------------------------------------------
# Step 4: Python virtualenv + pip deps
# ----------------------------------------------------------------------
log "Step 4/7: Python virtualenv"
VENV_DIR="${PROJECT_ROOT}/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    sudo -u "$REAL_USER" python3 -m venv --system-site-packages "$VENV_DIR"
fi
sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools
if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
    log "Installing Python deps from requirements.txt"
    sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install -r "${PROJECT_ROOT}/requirements.txt"
else
    warn "requirements.txt not found — skipping pip install"
fi

# ----------------------------------------------------------------------
# Step 5: ROS 2 Humble
# ----------------------------------------------------------------------
log "Step 5/7: ROS 2 Humble"
if [[ -f /opt/ros/${ROS_DISTRO}/setup.bash ]]; then
    log "ROS 2 ${ROS_DISTRO} already installed"
else
    if [[ "$(. /etc/os-release && echo "$ID")" == "ubuntu" ]]; then
        log "Adding OSRF apt repository"
        curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
            -o /usr/share/keyrings/ros-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
            > /etc/apt/sources.list.d/ros2.list
        apt-get update -y
        log "Installing ROS 2 ${ROS_DISTRO}-ros-base + extras"
        apt-get install -y \
            ros-${ROS_DISTRO}-ros-base \
            ros-${ROS_DISTRO}-cv-bridge \
            ros-${ROS_DISTRO}-vision-msgs \
            ros-${ROS_DISTRO}-sensor-msgs \
            ros-${ROS_DISTRO}-geometry-msgs \
            ros-${ROS_DISTRO}-nav-msgs \
            ros-${ROS_DISTRO}-tf2 \
            ros-${ROS_DISTRO}-tf2-ros \
            ros-${ROS_DISTRO}-tf2-geometry-msgs \
            ros-${ROS_DISTRO}-image-transport \
            ros-${ROS_DISTRO}-diagnostic-updater \
            ros-${ROS_DISTRO}-robot-localization \
            ros-${ROS_DISTRO}-navigation2 \
            ros-${ROS_DISTRO}-nav2-bringup \
            ros-${ROS_DISTRO}-slam-toolbox \
            python3-colcon-common-extensions \
            python3-rosdep python3-vcstool python3-argcomplete
        log "Initializing rosdep"
        rosdep init || true   # may already exist
        sudo -u "$REAL_USER" rosdep update || true
    else
        warn "ROS 2 Humble is officially supported only on Ubuntu 22.04."
        warn "Skipping apt install. Use Docker instead (see docker-compose.yml)."
    fi
fi

# ----------------------------------------------------------------------
# Step 6: Docker
# ----------------------------------------------------------------------
log "Step 6/7: Docker + Compose"
if command -v docker &>/dev/null; then
    log "Docker already installed"
else
    log "Installing Docker via official convenience script"
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh
    rm -f /tmp/get-docker.sh
    systemctl enable --now docker
fi
log "Installing docker compose plugin"
apt-get install -y docker-compose-plugin || true
if ! docker compose version &>/dev/null; then
    warn "docker compose plugin not available — install manually if needed"
fi

# ----------------------------------------------------------------------
# Step 7: udev rules + final config
# ----------------------------------------------------------------------
log "Step 7/7: udev rules + systemd tweaks"
UDEV_DIR="${PROJECT_ROOT}/udev"
if [[ -d "$UDEV_DIR" ]]; then
    cp -v "$UDEV_DIR"/*.rules /etc/udev/rules.d/ 2>/dev/null || true
    udevadm control --reload-rules || true
    udevadm trigger || true
fi

# Disable serial console on UART so GPS/LiDAR can use it
if [[ -f /boot/cmdline.txt ]]; then
    if grep -q "console=serial0,115200" /boot/cmdline.txt; then
        log "Removing serial console from /boot/cmdline.txt"
        sed -i 's/console=serial0,115200 //' /boot/cmdline.txt
    fi
fi

# Bump GPU memory (or rather, leave it minimal for headless)
if [[ -f /boot/config.txt ]] && ! grep -q "^gpu_mem=" /boot/config.txt; then
    echo "gpu_mem=16" >> /boot/config.txt
fi

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
log "================================================================"
log "  Installation complete!"
log "  Log:       $LOG_FILE"
log "  Venv:      $VENV_DIR"
log "  ROS 2:     /opt/ros/${ROS_DISTRO}/setup.bash"
log "  Docker:    $(docker --version 2>/dev/null || echo 'n/a')"
log ""
log "  Next steps:"
log "    1. Reboot the Pi to apply groups + device tree changes."
log "    2. Run:  source ${VENV_DIR}/bin/activate"
log "    3. Run:  source /opt/ros/${ROS_DISTRO}/setup.bash"
log "    4. Run:  ${SCRIPT_DIR}/setup_ros2.sh"
log "    5. Run:  ${SCRIPT_DIR}/start_services.sh"
log "================================================================"
