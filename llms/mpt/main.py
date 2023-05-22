from transformers import AutoModelForCausalLM

from llms.model_loaders import DEFAULT_CACHE_DIR


def main():
    model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b-instruct")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
