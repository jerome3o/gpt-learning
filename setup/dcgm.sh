# Based off:
# https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/dcgm-exporter.html


DCGM_EXPORTER_VERSION=2.1.4-2.3.1 && \
sudo docker run -d --rm \
   --gpus all \
   --net host \
   --cap-add SYS_ADMIN \
   nvcr.io/nvidia/k8s/dcgm-exporter:${DCGM_EXPORTER_VERSION}-ubuntu20.04 \
   -f /etc/dcgm-exporter/dcp-metrics-included.csv
