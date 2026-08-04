# JetPack SDK Installation Guide

> Target modules: Jetson Nano, Xavier NX, Orin Nano/NX, AGX Orin
> Recommended JetPack: **5.1.2** (L4T R35.4.1, CUDA 11.4, cuDNN 8.6, TensorRT 8.5)
> Latest tested: **6.0 / 6.1** (L4T R36.x, CUDA 12.x, TensorRT 10.x)

This guide walks through flashing JetPack onto a Jetson module and verifying
the post-install toolchain. Choose one of two paths depending on the module:

| Path | Recommended for |
|------|-----------------|
| **A — SD card image** | Jetson Nano (developer kit only) |
| **B — NVIDIA SDK Manager** | Xavier NX dev kit, Orin Nano/NX dev kit, AGX Orin |

---

## Path A — SD Card Image (Nano Developer Kit)

The Nano dev kit boots directly from a microSD card. No host PC is required.

### Step A.1 — Download the image
1. Visit https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit
2. Download `jp461.zip` (JetPack 4.6.1) or `jp512.zip` (JetPack 5.1.2, supported on the new B01 4 GB carrier).
3. Verify the SHA-256 checksum:
   ```bash
   sha256sum jp512.zip
   # compare against the value posted on the download page
   ```

### Step A.2 — Flash the card
Use `balenaEtcher`, `Raspberry Pi Imager`, or `dd`:

```bash
# Identify the SD card device (e.g. /dev/sdb)
lsblk

# Unzip and flash
unzip jp512.zip -d jp512
sudo dd if=jp512/sd-blob.img of=/dev/sdX bs=4M status=progress conv=fsync
sync
```

> ⚠️ Flashing the wrong device (`/dev/sda`) will destroy your host OS. Double
> check with `lsblk` before pressing Enter.

### Step A.3 — First boot
1. Insert the SD card into the Nano ( underside slot, contacts facing the board ).
2. Connect HDMI, keyboard, mouse, and the official 5 V 4 A power supply.
3. Accept the NVIDIA EULA and complete the Ubuntu OEM config wizard.
4. The `nvidia-l4t-bootloader` will resize the root partition on first boot —
   this can take a few minutes; **do not power off**.

### Step A.4 — Post-flash install (JetPack components)
The SD-card image already contains CUDA, cuDNN, TensorRT. To install
additional components (Multimedia API, CUDA samples):

```bash
sudo apt update
sudo apt install -y nvidia-l4t-jetson-multimedia-api cuda-samples-11-4
```

---

## Path B — NVIDIA SDK Manager (Xavier / Orin)

SDK Manager runs on an Ubuntu 20.04/22.04 **host PC** and flashes the
Jetson over USB-C.

### Step B.1 — Install SDK Manager
```bash
# On the host PC
sudo apt install -y libssl-dev libglib2.0-dev
wget https://developer.nvidia.com/sdkmanager-download
# The downloaded file is named like: sdkmanager_1.9.4-10804_amd64.deb
sudo dpkg -i sdkmanager_*.deb
sudo apt -f install -y
```

### Step B.2 — Put the Jetson into Force Recovery Mode
1. Power off the Jetson.
2. Short the **FORCE RECOVERY** pins (Nano: J40; Xavier NX: J512; Orin: button).
3. Connect the USB-C cable from the Jetson dev kit to the host PC.
4. Power on the Jetson.

Verify the device is visible to the host:
```bash
lsusb | grep NVIDIA
# Bus 001 Device 014: ID 0955:7020 NVIDIA Corp. APX   (Xavier)
# Bus 001 Device 014: ID 0955:7523 NVIDIA Corp. APX   (Orin)
```

### Step B.3 — Run SDK Manager
```bash
sdkmanager
```
1. Log in with an NVIDIA developer account.
2. On the **Step 01** screen, select the product family (e.g. *Jetson Xavier NX*).
3. Choose the JetPack version (5.1.2 recommended).
4. Accept licenses and download to the default `~/Downloads/nvidia/sdkm/downloads`.
5. On **Step 02** choose *Manual Setup* so SDK Manager flashes the device.
6. Enter the Jetson's IP / username / password when prompted for the
   runtime install of SDK components.

> Tip: For unattended installs, use `sdkmanager --cli install` with a
> `--recovery` JSON profile. This is the only way to flash multiple
> devices in a CI pipeline.

### Step B.4 — First boot
After flashing, the Jetson reboots into Ubuntu. Set up the user account,
connect to Wi-Fi/Ethernet, and continue to the **Verification** section below.

---

## Component Reference

After a successful install, the following components live under these paths:

| Component | Path | Notes |
|-----------|------|-------|
| CUDA Toolkit | `/usr/local/cuda` | symlink to `/usr/local/cuda-11.4` |
| cuDNN | `/usr/lib/aarch64-linux-gnu/libcudnn.so*` | header files in `/usr/include/aarch64-linux-gnu/` |
| TensorRT | `/usr/include/aarch64-linux-gnu/NvInfer*.h`, `libnvinfer.so*` | Python bindings in `/usr/lib/python3/dist-packages/tensorrt/` |
| VisionWorks | `/usr/lib/aarch64-linux-gnu/libVisionWorks.so*` | deprecated in JetPack 6, replaced by VPI |
| VPI | `/opt/nvidia/vpi` | cross-platform vision primitive library |
| Multimedia API | `/usr/src/jetson_multimedia_api` | Argus, V4L2, NvVideoEncoder samples |
| CUDA Samples | `/usr/local/cuda/samples` | build with `make -j$(nproc)` |
| DeepStream | `/opt/nvidia/deepstream/deepstream-6.x` | optional, install via SDK Manager |

---

## Post-Install Verification

Run each of these from a terminal on the Jetson itself.

### 1. CUDA — verify `nvcc` and run a sample
```bash
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 11.4, V11.4.315

cd /usr/local/cuda/samples/0_Simple/vectorAdd
sudo make -j$(nproc)
./vectorAdd
# [vectorAdd] PASSED
```

### 2. cuDNN — link check
```bash
ls /usr/lib/aarch64-linux-gnu/libcudnn* | head
# /usr/lib/aarch64-linux-gnu/libcudnn.so.8
# /usr/lib/aarch64-linux-gnu/libcudnn.so.8.6.0
# /usr/lib/aarch64-linux-gnu/libcudnn.so.8.6.0.140

# Compile a tiny test program
cat > /tmp/cudnn_check.cpp <<'EOF'
#include <cudnn_version.h>
#include <cstdio>
int main(){ printf("cuDNN %d.%d.%d\n",
    CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL); return 0; }
EOF
g++ /tmp/cudnn_check.cpp -lcudnn -o /tmp/cudnn_check && /tmp/cudnn_check
# cuDNN 8.6.0
```

### 3. TensorRT — Python + C++ sanity check
```bash
# Python
python3 - <<'PY'
import tensorrt as trt
print("TensorRT", trt.__version__)
print("GPU alive:", trt.Runtime(trt.Logger()).default_device_type)
PY

# C++ — use trtexec (the bundled engine benchmark tool)
/usr/src/tensorrt/bin/trtexec --help | head -3
```

### 4. GPU + clocks — `jetson-stats`
```bash
sudo apt install -y python3-pip
sudo pip3 install -U jetson-stats
sudo systemctl enable --now jetson_stats
jtop
# Interactive TUI showing CPU/GPU/EMC clocks, power, temperatures.
```

To force maximum clocks for benchmarking:
```bash
sudo nvpmodel -m 0          # MAXN
sudo jetson_clocks
```

### 5. Camera (Argus / V4L2)
```bash
# List video devices
ls -l /dev/video*
v4l2-ctl --list-devices

# Capture a single JPEG using the NVIDIA Argus GStreamer pipeline
nvgstcapture-1.0 --automate --capture-auto-jpeg --image-res=12
# Image is saved as ~/nvgst_capture/...
```

If the camera does not show up:
- Verify the CSI ribbon cable is seated correctly (contacts facing the PCB,
  blue tape facing away on Nano).
- Run `dmesg | grep -i imx` to confirm the IMX219/IMX477 driver probed.
- On Orin, the IMX477 driver may need to be enabled with `sudo modprobe imx477`.

### 6. Network — install wireless driver (Nano only)
```bash
sudo apt install -y libnvidia-nscq-0
sudo apt install -y bcmwl-kernel-source
sudo modprobe wl
ip link set wlan0 up
nmcli device wifi list
```

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sudo apt update` 404s | L4T apt sources stale | Edit `/etc/apt/sources.list.d/nvidia-l4t-source.list` and switch to the matching R35.x release repo |
| Black screen on HDMI | EDID negotiation | Append `video=HDMI-A-1:1920x1080@60D` to `APPEND` in `/boot/extlinux/extlinux.conf` |
| `nvcc: command not found` | PATH not set | Add `export PATH=/usr/local/cuda/bin:$PATH` to `~/.bashrc` |
| TensorRT import fails | Python bindings not on PYTHONPATH | `export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH` |
| `nvpmodel -m 0` errors | User not in `video` / `i2c` groups | `sudo usermod -aG video,i2c $USER` then re-login |
| jtop shows "daemon not running" | systemd unit not enabled | `sudo systemctl enable --now jetson_stats` |

---

## Next Steps

1. Install Docker with NVIDIA runtime support:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
     && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia.list
   sudo apt update && sudo apt install -y docker.io nvidia-container-toolkit
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```
2. Build the `av-jetson:latest` image using the `Dockerfile` in this directory.
3. Continue with `ide/tensorrt_setup.md` to build your first inference engine.
