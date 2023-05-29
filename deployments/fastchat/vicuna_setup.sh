set -xe

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python -m transformers.models.llama.convert_llama_weights_to_hf \
  --input_dir /mnt/raid/llama/llama-dl/ \
  --output_dir /mnt/raid/llama/hf/7B/ \
  --model_size 7B

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python -m transformers.models.llama.convert_llama_weights_to_hf \
  --input_dir /mnt/raid/llama/llama-dl/ \
  --output_dir /mnt/raid/llama/hf/13B/ \
  --model_size 13B

python -m fastchat.model.apply_delta \
    --base-model-path "/mnt/raid/llama/hf/7B/" \
    --target-model-path "/mnt/raid/vicuna/7B/" \
    --delta-path lmsys/vicuna-7b-delta-v1.1

python -m fastchat.model.apply_delta \
    --base-model-path "/mnt/raid/llama/hf/13B/" \
    --target-model-path "/mnt/raid/vicuna/13B/" \
    --delta-path lmsys/vicuna-13b-delta-v1.1
