# =============================================================================
# File: Dockerfile
# Brief: Multi-stage Docker image for NVIDIA Jetson (L4T) autonomous vehicle
#        stack — CUDA, cuDNN, TensorRT, PyTorch, OpenCV (CUDA), ROS2 rclpy.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
#
# Build on the Jetson device itself (the L4T base images are arm64 only and
# pull from nvcr.io). Requires nvidia-container-runtime / nvidia-docker2.
#
#   docker build -t av-jetson:latest .
#   docker run --rm -it --runtime nvidia --network host av-jetson:latest
#
# The build is split into four stages to keep the final image small:
#   1. base    — pull nvcr.io/nvidia/l4t-tensorrt, install system packages
#   2. opencv  — build OpenCV with CUDA backend (used for testing/viz)
#   3. python  — install Python deps + aarch64 wheels for PyTorch + TRT
#   4. runtime — copy only what is needed into a clean L4T image
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Stage 1: base
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config \
        python3 python3-pip python3-dev python3-setuptools \
        libpython3-dev \
        libopencv-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libfreetype6-dev libpng-dev libjpeg-dev libtiff-dev libwebp-dev \
        libopenblas-dev liblapack-dev libeigen3-dev \
        libhdf5-serial-dev hdf5-tools \
        libcanberra-gtk-module libcanberra-gtk3-module \
        libssl-dev libcurl4-openssl-dev \
        git wget curl unzip \
        udev i2c-tools v4l-utils \
        nano less file htop \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 2: opencv (built with CUDA) — optional, can be skipped if you are
# happy with the prebuilt libopencv-dev. Produces /opt/opencv.
# -----------------------------------------------------------------------------
FROM base AS opencv-builder

ARG OPENCV_VERSION=4.8.0
ARG CUDA_ARCH_BIN=7.2
ARG NUM_JOBS=4

RUN cd /tmp && \
    wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -O opencv.tar.gz && \
    tar xf opencv.tar.gz && \
    git clone --depth 1 https://github.com/opencv/opencv_contrib.git /tmp/opencv_contrib && \
    mkdir -p /tmp/opencv-${OPENCV_VERSION}/build && cd /tmp/opencv-${OPENCV_VERSION}/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
          -DBUILD_SHARED_LIBS=ON \
          -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
          -DBUILD_opencv_python3=ON \
          -DWITH_CUDA=ON -DWITH_CUDNN=ON \
          -DOPENCV_DNN_CUDA=ON \
          -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} -DCUDA_ARCH_PTX="" \
          -DWITH_GSTREAMER=ON -DWITH_FFMPEG=ON \
          -DWITH_LIBV4L=ON -DWITH_V4L=ON \
          -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON \
          -DCUDA_NVCC_FLAGS="--expt-relaxed-constexpr" \
          .. && \
    make -j${NUM_JOBS} && make install && ldconfig && \
    rm -rf /tmp/opencv*

# -----------------------------------------------------------------------------
# Stage 3: python — install Python deps and the aarch64 PyTorch wheels that
# NVIDIA publishes for JetPack.
# -----------------------------------------------------------------------------
FROM base AS python-builder

ARG TORCH_WHL=https://nvidia.box.com/shared/static/9olrt7pfdfqrqdfh8ao3etufc9d9m5qq.whl
ARG TORCHVISION_WHL=https://nvidia.box.com/shared/static/2q4j7mksm5w2h6r1s2d7t3z8f5o7w5p9.whl

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --upgrade pip wheel setuptools

# Install PyTorch + torchvision from NVIDIA's Jetson-specific wheels.
# These wheels are compiled against the L4T CUDA 11.4 / cuDNN 8.6 stack.
RUN pip3 install --no-cache-dir \
        ${TORCH_WHL} \
    && pip3 install --no-cache-dir \
        ${TORCHVISION_WHL}

# Install remaining Python dependencies
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# jetson-stats is needed for power/thermal monitoring at runtime
RUN pip3 install --no-cache-dir jetson-stats

# -----------------------------------------------------------------------------
# Stage 4: runtime — clean L4T image with only what we need
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        libcanberra-gtk-module libcanberra-gtk3-module \
        libssl3 libcurl4 \
        i2c-tools v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy site-packages from the python builder (includes torch, torchvision,
# tensorrt, opencv-python-headless, etc.)
COPY --from=python-builder /usr/local/lib/python3.10/dist-packages \
     /usr/local/lib/python3.10/dist-packages

# Copy OpenCV built with CUDA (if you used the opencv-builder stage)
# COPY --from=opencv-builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
# COPY --from=opencv-builder /usr/local/include/opencv4 /usr/local/include/opencv4

# Application code
WORKDIR /app
COPY python/ ./python/
COPY cpp/    ./cpp/
COPY cuda/   ./cuda/
COPY ros2/   ./ros2/

# Models and engines are mounted at runtime, not baked in
VOLUME ["/models", "/data"]

# Default entrypoint drops into bash; override with `--entrypoint python3`
# or compose a launch file in ros2/launch.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]

# -----------------------------------------------------------------------------
# Inline entrypoint script (also written to the image as docker-entrypoint.sh)
# -----------------------------------------------------------------------------
# docker-entrypoint.sh contents (for reference — also committed alongside
# the Dockerfile in a real repo):
#
# #!/usr/bin/env bash
# set -euo pipefail
#
# # Make sure the nvidia runtime has surfaced devices under /dev
# if [ ! -e /dev/nvidiactl ]; then
#   echo "[entrypoint] WARNING: /dev/nvidiactl not present — did you use --runtime=nvidia?"
# fi
#
# # jetson-stats needs sudo / sudoers; in containers we just allow it
# export JETSON_STATS_PREFER_HOST=1
#
# # If the first arg looks like a flag, prepend 'bash'; otherwise exec as-is.
# if [ "${1#-}" != "$1" ]; then
#   set -- bash "$@"
# fi
#
# exec "$@"
# -----------------------------------------------------------------------------
