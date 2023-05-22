from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    AutoConfig,
)
import torch
import json
from pathlib import Path

from llms.model_loaders import DEFAULT_CACHE_DIR

import json


def main():
    model_name = "mosaicml/mpt-7b-instruct"

    print("loading tokenizer")

    cache_dir = DEFAULT_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    print("loading config")
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision="671f67f",
        cache_dir=cache_dir,
    )
    # config.attn_config["attn_impl"] = "triton"
    # config.attn_config["alibi"] = False

    print("creating device_map")
    with open(Path(__file__).parent / "names.json") as f:
        param_names = json.load(f)

    device_map = {n: 0 for n in param_names}

    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        device_map=device_map,
        # torch_dtype=torch.float16,
        load_in_8bit=True,
        revision="671f67f",
        cache_dir=cache_dir,
    )

    print("running inference")

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

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
    print(fmt_ex)

    inputs = tokenizer([fmt_ex], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        temperature=0.7,
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )
    text = tokenizer.decode(outputs[0])
    print(text)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
