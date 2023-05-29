set -xe

python -m fastchat.serve.model_worker \
  --model_name 'vicuna-13b-v1.1-8bit' \
  --model-path /mnt/raid/vicuna/13B/ \
  --load-8bit
