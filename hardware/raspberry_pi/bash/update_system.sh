#!/usr/bin/env bash
# File:        bash/update_system.sh
# Brief:       Update the Raspberry Pi: apt update/upgrade, rpi-update
#              firmware (Pi OS only), kernel headers, and clean caches.
#              Safe to re-run.
# Author:      Autonomous Vehicle Team
# Date:        2025-01-30
# License:     MIT

set -euo pipefail

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="/tmp/av_update_$(date +%Y%m%d_%H%M%S).log"
DO_FIRMWARE=1
DO_HEADERS=1
DO_CLEAN=1

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-firmware) DO_FIRMWARE=0; shift ;;
        --no-headers)  DO_HEADERS=0;  shift ;;
        --no-clean)    DO_CLEAN=0;    shift ;;
        -h|--help)
            echo "Usage: $0 [--no-firmware] [--no-headers] [--no-clean]"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Logging
log()  { echo -e "\033[1;32m[update]\033[0m $*" | tee -a "$LOG_FILE"; }
warn() { echo -e "\033[1;33m[update:WARN]\033[0m $*" | tee -a "$LOG_FILE" >&2; }
err()  { echo -e "\033[1;31m[update:ERR]\033[0m $*" | tee -a "$LOG_FILE" >&2; }
die()  { err "$*"; exit 1; }

trap 'die "Failed at line $LINENO (see $LOG_FILE)"' ERR

[[ "$(id -u)" -eq 0 ]] || die "Please run with sudo: sudo $0"

log "Logging to $LOG_FILE"
log "Arch: $(uname -m)  Kernel: $(uname -r)"

# ----------------------------------------------------------------------
# Step 1: apt update + upgrade
# ----------------------------------------------------------------------
log "Step 1/5: apt update"
apt-get update -y

log "Step 2/5: apt upgrade (this may take a while)"
DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" upgrade

# Distribution upgrade if any packages still held back
log "Step 2b: apt dist-upgrade"
DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" dist-upgrade

# ----------------------------------------------------------------------
# Step 3: rpi-update (firmware + bootloader)
# ----------------------------------------------------------------------
if [[ $DO_FIRMWARE -eq 1 ]]; then
    log "Step 3/5: rpi-update (firmware)"
    if command -v rpi-update &>/dev/null; then
        # rpi-update is officially recommended only on Pi OS. On Ubuntu,
        # firmware comes from the linux-raspi package.
        if [[ -f /etc/os-release ]] && grep -q "Raspberry Pi" /etc/os-release; then
            log "  Detected Raspberry Pi OS — running rpi-update"
            YES=1 rpi-update || warn "rpi-update failed (non-fatal)"
        else
            log "  Detected non-Pi-OS — installing linux-firmware-raspi2 if needed"
            apt-get install -y linux-firmware-raspi2 linux-raspi || true
        fi
    else
        warn "rpi-update not installed — skipping firmware step"
    fi
else
    log "Step 3/5: (skipped firmware via --no-firmware)"
fi

# ----------------------------------------------------------------------
# Step 4: kernel headers (needed for compiling kernel modules,
# e.g. for some SPI CAN drivers or Realtek Wi-Fi dongles)
# ----------------------------------------------------------------------
if [[ $DO_HEADERS -eq 1 ]]; then
    log "Step 4/5: kernel headers"
    KERNEL_VER="$(uname -r)"
    if [[ -f /etc/os-release ]] && grep -q "Raspberry Pi" /etc/os-release; then
        apt-get install -y "linux-headers-${KERNEL_VER}" || \
            warn "linux-headers-${KERNEL_VER} not available"
    else
        apt-get install -y "linux-headers-${KERNEL_VER}" || \
            apt-get install -y "linux-headers-raspi" || true
    fi
    # dkms for out-of-tree modules
    apt-get install -y dkms || true
else
    log "Step 4/5: (skipped headers via --no-headers)"
fi

# ----------------------------------------------------------------------
# Step 5: clean caches
# ----------------------------------------------------------------------
if [[ $DO_CLEAN -eq 1 ]]; then
    log "Step 5/5: clean apt caches"
    apt-get autoremove -y
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    journalctl --vacuum-time=7d || true
    log "  Cleaned apt cache, removed unused kernels, vacuumed journals"
else
    log "Step 5/5: (skipped clean via --no-clean)"
fi

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
log "================================================================"
log "  System update complete"
log "  Log: $LOG_FILE"
log ""
log "  If kernel/firmware was updated, REBOOT to apply changes:"
log "    sudo reboot"
log "================================================================"

# Detect if reboot is required
if [[ -f /var/run/reboot-required ]]; then
    warn "Reboot required (see /var/run/reboot-required)"
fi
