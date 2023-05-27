# Based off:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements


set -xe

# Verification
lspci | grep -i nvidia
uname -m && cat /etc/*release
gcc --version
uname -r


sudo apt-key del 7fa2af80


# install keyring?
distro=ubuntu2204
arch=x86_64

wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Install cuda
sudo apt-get install cuda

