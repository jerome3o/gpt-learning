# Based off:
# https://github.com/utkuozdemir/nvidia_gpu_exporter/blob/master/INSTALL.md

wget https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v1.2.0/nvidia-gpu-exporter_1.2.0_linux_amd64.deb

sudo dpkg -i nvidia-gpu-exporter_1.2.0_linux_amd64.deb

rm nvidia-gpu-exporter_1.2.0_linux_amd64.deb

sudo tee /etc/systemd/system/nvidia_gpu_exporter.service > /dev/null <<'EOF'
[Unit]
Description=Nvidia GPU Exporter
After=network-online.target

[Service]
Type=simple

User=nvidia_gpu_exporter
Group=nvidia_gpu_exporter

ExecStart=/usr/bin/nvidia_gpu_exporter

SyslogIdentifier=nvidia_gpu_exporter

Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now nvidia_gpu_exporter
