from transformers import AutoTokenizer, AutoModelForCausalLM

_CACHE_DIR = "/media/raid2/transformers_cache"


def main():
    tokenizer_3b = AutoTokenizer.from_pretrained(
        "databricks/dolly-v2-3b", cache_dir=_CACHE_DIR
    )
    model_3b = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-3b", cache_dir=_CACHE_DIR
    )

    tokenizer_6b = AutoTokenizer.from_pretrained(
        "databricks/dolly-v1-6b", cache_dir=_CACHE_DIR
    )
    model_6b = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v1-6b", cache_dir=_CACHE_DIR
    )

    tokenizer_7b = AutoTokenizer.from_pretrained(
        "databricks/dolly-v2-7b", cache_dir=_CACHE_DIR
    )
    model_7b = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-7b", cache_dir=_CACHE_DIR
    )

    tokenizer_12b = AutoTokenizer.from_pretrained(
        "databricks/dolly-v2-12b", cache_dir=_CACHE_DIR
    )
    model_12b = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-12b", cache_dir=_CACHE_DIR
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
