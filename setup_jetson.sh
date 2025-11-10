#!/bin/bash
#
# This script installs the required dependencies for running the face recognition 
# application on a NVIDIA Jetson device.
#
# It installs PyTorch, TorchVision, and other libraries that require specific builds
# for the Jetson's ARM architecture.

set -e

# 1. Install system-level dependencies
echo "[INFO] Installing system-level dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev libjpeg-dev libpython3-dev python3-pip

# 2. Install PyTorch
echo "[INFO] Installing PyTorch v1.10.2 for Jetson..."
# URL for PyTorch 1.10.2 for JetPack 4.6.x (Python 3.6)
# For other versions, see: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.10.2-cp36-cp36m-linux_aarch64.whl -O torch-1.10.2-cp36-cp36m-linux_aarch64.whl

sudo apt-get install -y libopenblas-base libopenmpi-dev
pip3 install Cython
pip3 install numpy torch-1.10.2-cp36-cp36m-linux_aarch64.whl
rm torch-1.10.2-cp36-cp36m-linux_aarch64.whl

# 3. Install TorchVision
echo "[INFO] Building and installing TorchVision v0.11.1 from source..."
sudo apt-get install -y libjpeg-dev zlib1g-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
rm -rf torchvision

# 4. Install dlib and face_recognition
echo "[INFO] Installing dlib and face_recognition..."
# This may take a very long time as dlib is built from source.
pip3 install dlib
pip3 install face_recognition

# 5. Install OpenCV
echo "[INFO] Installing OpenCV..."
pip3 install opencv-python

# 6. Install remaining packages from requirements.txt
echo "[INFO] Installing remaining Python packages..."
pip3 install -r requirements.txt

echo "[SUCCESS] Setup complete. You can now run the application."
