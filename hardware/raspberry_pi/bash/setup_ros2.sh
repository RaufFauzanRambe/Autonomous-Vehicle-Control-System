#!/usr/bin/env bash
# File:        bash/setup_ros2.sh
# Brief:       Set up the ROS 2 Humble workspace on the Raspberry Pi:
#              source ROS 2, create the workspace, copy package
#              templates, run colcon build, source the install, and
#              optionally create a new package skeleton.
# Author:      Autonomous Vehicle Team
# Date:        2025-01-30
# License:     MIT

set -euo pipefail

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"

NEW_PACKAGE_NAME="${1:-}"
NEW_PACKAGE_TYPE="${NEW_PACKAGE_TYPE:-python}"   # python|cpp

# Logging
log()  { echo -e "\033[1;34m[ros2]\033[0m $*"; }
warn() { echo -e "\033[1;33m[ros2:WARN]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[ros2:ERR]\033[0m $*" >&2; exit 1; }

trap 'die "Failed at line $LINENO"' ERR

# ----------------------------------------------------------------------
# Step 1: Source ROS 2
# ----------------------------------------------------------------------
log "Step 1/6: Source ROS 2 ${ROS_DISTRO}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ ! -f "$ROS_SETUP" ]]; then
    die "ROS 2 ${ROS_DISTRO} not found at $ROS_SETUP. Run install.sh first."
fi
# shellcheck disable=SC1090
source "$ROS_SETUP"
log "ROS_DISTRO=$ROS_DISTRO  RMW=$RMW_IMPLEMENTATION"

# ----------------------------------------------------------------------
# Step 2: Create workspace
# ----------------------------------------------------------------------
log "Step 2/6: Create workspace at $ROS_WS"
mkdir -p "$ROS_WS/src"

# Persist sourcing to ~/.bashrc (idempotent)
if ! grep -q "source $ROS_SETUP" "$HOME/.bashrc"; then
    echo "source $ROS_SETUP" >> "$HOME/.bashrc"
    log "Added 'source $ROS_SETUP' to ~/.bashrc"
fi
if ! grep -q "source $ROS_WS/install/setup.bash" "$HOME/.bashrc"; then
    echo "source $ROS_WS/install/setup.bash" >> "$HOME/.bashrc"
    log "Added workspace source to ~/.bashrc"
fi

# ----------------------------------------------------------------------
# Step 3: Copy package templates
# ----------------------------------------------------------------------
log "Step 3/6: Copy package templates"
TPL_DIR="${PROJECT_ROOT}/ros2"
if [[ -d "$TPL_DIR" ]]; then
    for pkg_dir in "$TPL_DIR"/*; do
        [[ -d "$pkg_dir" ]] || continue
        pkg_name="$(basename "$pkg_dir")"
        if [[ -d "$ROS_WS/src/$pkg_name" ]]; then
            log "  • $pkg_name already in workspace — skipping copy"
        else
            log "  • Copying $pkg_name → $ROS_WS/src/"
            cp -r "$pkg_dir" "$ROS_WS/src/$pkg_name"
        fi
    done
else
    warn "No template dir at $TPL_DIR — skipping copy"
fi

# ----------------------------------------------------------------------
# Step 4: rosdep install dependencies
# ----------------------------------------------------------------------
log "Step 4/6: rosdep install"
if command -v rosdep &>/dev/null; then
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
        sudo rosdep init || true
    fi
    rosdep update || true
    rosdep install --from-paths "$ROS_WS/src" \
                   --ignore-src -y -r || \
        warn "rosdep reported issues (non-fatal)"
else
    warn "rosdep not installed — skipping"
fi

# ----------------------------------------------------------------------
# Step 5: colcon build
# ----------------------------------------------------------------------
log "Step 5/6: colcon build"
cd "$ROS_WS"

# Pick a parallelism based on available memory (Pi 4 with 2 GB → 1 worker)
RAM_KB="$(grep MemTotal /proc/meminfo | awk '{print $2}')"
RAM_GB=$(( RAM_KB / 1024 / 1024 ))
if [[ $RAM_GB -lt 3 ]]; then
    WORKERS=1
    EXTRA_FLAGS=(--executor serial)
    log "  Low-RAM system (${RAM_GB} GB): using 1 worker + serial executor"
else
    WORKERS=$(( RAM_GB / 2 ))
    [[ $WORKERS -lt 1 ]] && WORKERS=1
    [[ $WORKERS -gt 4 ]] && WORKERS=4
    EXTRA_FLAGS=()
    log "  Using $WORKERS parallel workers (${RAM_GB} GB RAM)"
fi

colcon build \
    --symlink-install \
    --parallel-workers "$WORKERS" \
    "${EXTRA_FLAGS[@]}" \
    --event-handlers console_cohesion+

# Source the freshly built workspace
# shellcheck disable=SC1091
source "$ROS_WS/install/setup.bash"

log "Available packages:"
colcon list 2>/dev/null | sed 's/^/  /' || true

# ----------------------------------------------------------------------
# Step 6: Optionally create a new package skeleton
# ----------------------------------------------------------------------
if [[ -n "$NEW_PACKAGE_NAME" ]]; then
    log "Step 6/6: Creating new package '$NEW_PACKAGE_NAME' (type=$NEW_PACKAGE_TYPE)"
    cd "$ROS_WS/src"
    if [[ -d "$NEW_PACKAGE_NAME" ]]; then
        warn "Package '$NEW_PACKAGE_NAME' already exists — skipping create"
    else
        case "$NEW_PACKAGE_TYPE" in
            python|py)
                ros2 pkg create --build-type ament_python --license MIT \
                    --dependencies rclpy std_msgs sensor_msgs geometry_msgs \
                    "$NEW_PACKAGE_NAME"
                ;;
            cpp|c++)
                ros2 pkg create --build-type ament_cmake --license MIT \
                    --dependencies rclcpp std_msgs sensor_msgs \
                    "$NEW_PACKAGE_NAME"
                ;;
            *)
                die "Unknown package type: $NEW_PACKAGE_TYPE (use python|cpp)"
                ;;
        esac
        log "Created $NEW_PACKAGE_NAME. Re-run $0 to build it."
    fi
else
    log "Step 6/6: (skipped — no new package name provided)"
fi

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
log "================================================================"
log "  ROS 2 workspace ready at $ROS_WS"
log "  Build artifacts in $ROS_WS/install/"
log "  Run 'ros2 launch av_bringup launch.py' to start the stack."
log "================================================================"
