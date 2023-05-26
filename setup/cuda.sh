set -xe

# Verification
lspci | grep -i nvidia
uname -m && cat /etc/*release
gcc --version
uname -r


sudo apt-key del 7fa2af80
