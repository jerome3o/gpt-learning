from transformers import AutoTokenizer, AutoModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline


def main():
    model_name = "databricks/dolly-v2-12b"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
    )

    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    while True:
        print("AI:", generate_text(input("Human: "))[0]["generated_text"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
