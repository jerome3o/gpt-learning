set -xe

(
  cd stable-diffusion-webui-docker/
  sudo docker compose --profile auto up --build -d
)
