set -xe

python -m fastchat.serve.model_worker \
  --model-name 'vicuna-13b-v1.1-8bit' \
  --model-path /mnt/raid/vicuna/13B/ \
  --load-8bit \
  --host 0.0.0.0
