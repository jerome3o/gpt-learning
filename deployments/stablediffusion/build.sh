set -xe
lenv .env

git clone git@github.com:AbdBarho/stable-diffusion-webui-docker.git
(
  cd stable-diffusion-webui-docker/
  sudo docker compose --profile download up --build
)

