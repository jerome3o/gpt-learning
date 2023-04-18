from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    tokenizer_3b = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
    model_3b = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")

    tokenizer_6b = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b")
    model_6b = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b")

    tokenizer_7b = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b")
    model_7b = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b")

    tokenizer_12b = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
    model_12b = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
