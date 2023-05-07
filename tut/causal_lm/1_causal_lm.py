# working from this tutorial:
# https://huggingface.co/docs/transformers/tasks/language_modeling

# pip install transformers datasets evaluate
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    eli5 = load_dataset("eli5", split="train_asks[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2)
    print(eli5["train"][0])

    tokensizer = AutoTokenizer.from_pretrained("distilgpt2")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
