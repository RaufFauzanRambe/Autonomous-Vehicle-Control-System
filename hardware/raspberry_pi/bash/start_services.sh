#!/usr/bin/env bash
# File:        bash/start_services.sh
# Brief:       Launch all autonomous-vehicle services on the Raspberry
#              Pi: docker compose stack, ROS 2 nodes, telemetry
#              dashboard, and a background daemon. Safe to re-run.
# Author:      Autonomous Vehicle Team
# Date:        2025-01-30
# License:     MIT

set -euo pipefail

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
VEHICLE_ID="${VEHICLE_ID:-av-01}"
LOG_DIR="${LOG_DIR:-/var/log/autovehicle}"
PID_DIR="${PID_DIR:-/var/run/autovehicle}"

# Flags
USE_DOCKER=1
USE_ROS2=1
USE_DAEMON=0
FOREGROUND=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-docker)       USE_DOCKER=0; shift ;;
        --no-ros2)         USE_ROS2=0;   shift ;;
        --daemon)          USE_DAEMON=1; shift ;;
        --foreground|-f)   FOREGROUND=1; shift ;;
        --vehicle-id)      VEHICLE_ID="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [options]
  --no-docker        Skip docker compose
  --no-ros2          Skip ROS 2 launch
  --daemon           Background daemon mode (logs to $LOG_DIR)
  --foreground|-f    Run in foreground (don't daemonize)
  --vehicle-id ID    Override vehicle id (default: $VEHICLE_ID)
EOF
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Logging
log()  { echo -e "\033[1;36m[start]\033[0m $*"; }
warn() { echo -e "\033[1;33m[start:WARN]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[start:ERR]\033[0m $*" >&2; exit 1; }

trap 'die "Failed at line $LINENO"' ERR

# ----------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------
mkdir -p "$LOG_DIR" "$PID_DIR"

# Source ROS 2 if needed
if [[ $USE_ROS2 -eq 1 ]]; then
    if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
        # shellcheck disable=SC1091
        source "/opt/ros/${ROS_DISTRO}/setup.bash"
    else
        warn "ROS 2 ${ROS_DISTRO} not found — disabling ROS 2 launch"
        USE_ROS2=0
    fi
    if [[ -f "$ROS_WS/install/setup.bash" ]]; then
        # shellcheck disable=SC1091
        source "$ROS_WS/install/setup.bash"
    else
        warn "Workspace not found at $ROS_WS — disabling ROS 2 launch"
        USE_ROS2=0
    fi
fi

# ----------------------------------------------------------------------
# Docker compose
# ----------------------------------------------------------------------
if [[ $USE_DOCKER -eq 1 ]]; then
    log "Step 1/3: docker compose up"
    if command -v docker &>/dev/null; then
        cd "$PROJECT_ROOT"
        docker compose pull --ignore-pull-failures || true
        docker compose up -d
        log "  Services:"
        docker compose ps --format 'table {{.Name}}\t{{.Status}}' | sed 's/^/    /'
        cd - >/dev/null
    else
        warn "docker not installed — skipping compose"
    fi
else
    log "Step 1/3: (skipped docker via --no-docker)"
fi

# ----------------------------------------------------------------------
# ROS 2 launch
# ----------------------------------------------------------------------
ROS2_PID=""
if [[ $USE_ROS2 -eq 1 ]]; then
    log "Step 2/3: ROS 2 bringup (vehicle_id=$VEHICLE_ID)"
    if ros2 pkg list 2>/dev/null | grep -q "^av_bringup "; then
        if [[ $FOREGROUND -eq 1 ]]; then
            log "  Running in foreground — Ctrl+C to stop"
            exec ros2 launch av_bringup launch.py "vehicle_id:=$VEHICLE_ID"
        else
            nohup ros2 launch av_bringup launch.py \
                "vehicle_id:=$VEHICLE_ID" \
                >"$LOG_DIR/ros2.log" 2>&1 &
            ROS2_PID=$!
            echo "$ROS2_PID" > "$PID_DIR/ros2.pid"
            log "  ROS 2 bringup started (pid=$ROS2_PID, log=$LOG_DIR/ros2.log)"
        fi
    else
        warn "av_bringup package not found — run setup_ros2.sh first"
    fi
else
    log "Step 2/3: (skipped ROS 2 via --no-ros2)"
fi

# ----------------------------------------------------------------------
# Optional background daemon (e.g. Python main loop alongside ROS 2)
# ----------------------------------------------------------------------
if [[ $USE_DAEMON -eq 1 ]]; then
    log "Step 3/3: Launching Python controller daemon"
    VENV_PY="${PROJECT_ROOT}/venv/bin/python"
    if [[ -x "$VENV_PY" ]]; then
        nohup "$VENV_PY" "${PROJECT_ROOT}/python/main.py" \
            --rate 30 \
            --config "${PROJECT_ROOT}/config/${VEHICLE_ID}.yaml" \
            --log-file "$LOG_DIR/python.log" \
            >"$LOG_DIR/python.stdout" 2>&1 &
        PYTHON_PID=$!
        echo "$PYTHON_PID" > "$PID_DIR/python.pid"
        log "  Python controller started (pid=$PYTHON_PID)"
    else
        warn "Venv python not found at $VENV_PY — skipping Python daemon"
    fi
else
    log "Step 3/3: (skipped Python daemon — use --daemon to enable)"
fi

# ----------------------------------------------------------------------
# Health summary
# ----------------------------------------------------------------------
if [[ $FOREGROUND -eq 0 ]]; then
    log "================================================================"
    log "  Services launched for vehicle '$VEHICLE_ID'"
    log "  Logs:   $LOG_DIR/"
    log "  PIDs:   $PID_DIR/"
    if [[ -n "$ROS2_PID" ]]; then
        log "  ROS 2:  pid=$ROS2_PID"
        log ""
        log "  Tail logs:    tail -f $LOG_DIR/ros2.log"
        log "  Stop ROS 2:   kill \$(cat $PID_DIR/ros2.pid)"
    fi
    log "  Stop all:     ${SCRIPT_DIR}/stop_services.sh  (not yet implemented)"
    log "================================================================"
fi
