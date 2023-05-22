from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

from llms.model_loaders import DEFAULT_CACHE_DIR


def main():
    model_name = "mosaicml/mpt-7b-instruct"

    cache_dir = DEFAULT_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision="671f67f",
        cache_dir=cache_dir,
    )
    config.attn_config["attn_impl"] = "flash"
    config.attn_config["alibi"] = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        revision="671f67f",
        cache_dir=cache_dir,
    )
    print(model, tokenizer)

    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    PROMPT_FOR_GENERATION_FORMAT = """{intro}
    {instruction_key}
    {instruction}
    {response_key}
    """.format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction="{instruction}",
        response_key=RESPONSE_KEY,
    )

    example = "Please write a poem about frogs."
    fmt_ex = PROMPT_FOR_GENERATION_FORMAT.format(instruction=example)

    inputs = tokenizer([fmt_ex], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        temperature=0.7,
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0])
    print(text)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
