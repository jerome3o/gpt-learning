set -xe


python -m fastchat.model.apply_delta \
    --base-model-path /mnt/raid/llama/llama-dl/13B \
    --target-model-path /mnt/raid/vicuna/13B \
    --delta-path lmsys/vicuna-13b-delta-v1.1

