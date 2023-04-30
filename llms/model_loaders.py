from typing import Callable, List, Tuple
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from llms.dolly.helpers import InstructionTextGenerationPipeline

_DEFAULT_CACHE_DIR = "/mnt/raid2/transformers_cache/"
_DEFAULT_CACHE_DIR = _DEFAULT_CACHE_DIR if Path(_DEFAULT_CACHE_DIR).exists() else None


def _load_transformers_model(
    model_name: str,
    load_in_8bit: bool = True,
    cache_dir: str = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        cache_dir=cache_dir,
    )

    return tokenizer, model


def load_stablelm(model_name: str, load_in_8bit: bool = True, cache_dir: str = None):
    tokenizer, model = _load_transformers_model(model_name, load_in_8bit, cache_dir)

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    # system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    # - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    # - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    # - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    # - StableLM will refuse to participate in anything that could harm a human.
    # """
    system_prompt = ""
    end_token_text_list = [tokenizer.decode([x]) for x in [50278, 50279, 50277, 1, 0]]

    def generate_text(s: str, batch_size: int = 1):
        torch.cuda.empty_cache()
        prompt = f"{system_prompt}<|USER|>{s}<|ASSISTANT|>"

        outputs = []
        inputs = tokenizer([prompt] * batch_size, return_tensors="pt").to("cuda")
        all_tokens = model.generate(
            **inputs,
            temperature=0.7,
            max_new_tokens=4000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        )
        for tokens in all_tokens:
            output = tokenizer.decode(tokens)[len(prompt) :]
            for end_token_text in end_token_text_list:
                if output.endswith(end_token_text):
                    output = output[: -len(end_token_text)]
                    break
            outputs.append(output)

        return outputs

    return tokenizer, model, generate_text


def load_openassistant(
    model_name: str,
    load_in_8bit: bool = True,
    cache_dir: str = None,
):
    tokenizer, model = _load_transformers_model(model_name, load_in_8bit, cache_dir)
    stop_token_ids = tokenizer.all_special_ids

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = stop_token_ids
            for stop_id in stop_ids:
                if all(v[-1] == stop_id for v in input_ids):
                    return True
            return False

    end_token_text_list = [tokenizer.decode([x]) for x in stop_token_ids]

    system_prompt = """<|system|># OpenAssistant
    - OpenAssistant is a helpful and harmless open-source AI language model developed by StabilityAI.
    - OpenAssistant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - OpenAssistant is more than just an information source, OpenAssistant is also able to write poetry, short stories, and make jokes.
    - OpenAssistant will refuse to participate in anything that could harm a human.
    """

    def generate_text(s: str, batch_size: int = 1):
        prompt = f"{system_prompt}<|prompter|>{s}<|assistant|>"

        outputs = []
        inputs = tokenizer([prompt] * batch_size, return_tensors="pt").to("cuda")
        all_tokens = model.generate(
            **inputs,
            temperature=0.7,
            max_new_tokens=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        )
        for tokens in all_tokens:
            output = tokenizer.decode(tokens)[len(prompt) :]
            for end_token_text in end_token_text_list:
                if output.endswith(end_token_text):
                    output = output[: -len(end_token_text)]
                    break

            output = output.replace("<|endoftext|>", "")
            outputs.append(output)

        return outputs

    return tokenizer, model, generate_text


def load_dolly(
    model_name: str,
    load_in_8bit: bool = True,
    cache_dir: str = None,
) -> Callable[[str, int], List[str]]:
    tokenizer, model = _load_transformers_model(model_name, load_in_8bit, cache_dir)
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    def _generate_text(prompt: str, batch_size: int = None) -> List[str]:
        return [
            o["generated_text"] for o in generate_text(prompt, batch_size=batch_size)
        ]

    return tokenizer, model, _generate_text


def load_model(
    model_name: str,
    load_in_8bit: bool = True,
    load_in_half: bool = True,
    cache_dir: str = None,
) -> Callable[[str, int], List[str]]:
    args = (model_name, load_in_8bit, cache_dir)

    tokenizer, model, generate_text = None, None, None

    if "dolly" in model_name.lower():
        tokenizer, model, generate_text = load_dolly(*args)

    if "oasst" in model_name.lower() or "openassistant" in model_name.lower():
        tokenizer, model, generate_text = load_openassistant(*args)

    if "stablelm" in model_name.lower():
        tokenizer, model, generate_text = load_stablelm(*args)

    if generate_text is None:
        raise ValueError(f"Unknown model {model_name}")

    if load_in_half:
        model = model.half()

    return generate_text
