from transformers import AutoTokenizer, AutoModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline

_CACHE_DIR = "/mnt/raid2/transformers_cache"


def main():
    tokenizer_3b = AutoTokenizer.from_pretrained(
        "databricks/dolly-v2-3b",
        cache_dir=_CACHE_DIR,
        device_map="auto",
    )
    model_3b = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-3b",
        cache_dir=_CACHE_DIR,
        device_map="auto",
        # load_in_8bit=True,
    )

    # tokenizer_6b = AutoTokenizer.from_pretrained(
    #     "databricks/dolly-v1-6b", cache_dir=_CACHE_DIR
    # )
    # model_6b = AutoModelForCausalLM.from_pretrained(
    #     "databricks/dolly-v1-6b", cache_dir=_CACHE_DIR
    # )

    # tokenizer_7b = AutoTokenizer.from_pretrained(
    #     "databricks/dolly-v2-7b", cache_dir=_CACHE_DIR
    # )
    # model_7b = AutoModelForCausalLM.from_pretrained(
    #     "databricks/dolly-v2-7b", cache_dir=_CACHE_DIR
    # )

    # tokenizer_12b = AutoTokenizer.from_pretrained(
    #     "databricks/dolly-v2-12b",
    #     cache_dir=_CACHE_DIR,
    #     device_map="auto",
    # )
    # model_12b = AutoModelForCausalLM.from_pretrained(
    #     "databricks/dolly-v2-12b",
    #     cache_dir=_CACHE_DIR,
    #     device_map="auto",
    #     load_in_8bit=True,
    # )
    model = model_3b
    tokenizer = tokenizer_3b

    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    while True:
        print("AI:", generate_text(input("Human: "))[0]["generated_text"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
