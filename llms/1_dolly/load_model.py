from typing import Callable, List
from pydantic import BaseModel
import json
import logging
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline
from benchmark_prompts import (
    BRIEF_BENCHMARK_PROMPTS,
    CODE_BENCHMARK_PROMPTS,
    LONG_BENCHMARK_PROMPTS,
)

_logger = logging.getLogger(__name__)

_DOLLY_LIST = [
    "databricks/dolly-v2-3b",
    # "databricks/dolly-v1-6b",
    # "databricks/dolly-v2-7b",
    # "databricks/dolly-v2-12b",
]


class PromptResult(BaseModel):
    prompt: str
    duration: float
    outputs: str


class PromptBatchResult(BaseModel):
    prompt: str
    outputs: List[str]


class ModelBenchmarkResult(BaseModel):
    model_name: str
    brief_prompt_results: List[PromptBatchResult]
    long_prompt_results: List[PromptBatchResult]
    code_prompt_results: List[PromptBatchResult]


def load_model(model_name: str) -> Callable[[str, int], List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # load_in_8bit=True,
    )

    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    def _generate_text(prompt: str, batch_size: int = None) -> List[str]:
        return [o["generated_text"] for o in generate_text(prompt, batch_size=batch_size)]

    return _generate_text


def run_benchmarks_on_model(model_name: str, batch_size: int) -> ModelBenchmarkResult:
    model = load_model(model_name)

    def run_benchmark(prompts: List[str], batch_size: int, title: str) -> List[PromptBatchResult]:
        _logger.info(f"Running {title} benchmarks")
        results = []
        for prompt in tqdm(prompts):
            outputs = model(prompt, batch_size)
            results.append(
                PromptBatchResult(
                    prompt=prompt,
                    outputs=outputs,
                )
            )
        return results

    return ModelBenchmarkResult(
        model_name=model_name,
        brief_prompt_results=run_benchmark(
            BRIEF_BENCHMARK_PROMPTS, batch_size=batch_size, title="Brief"
        ),
        long_prompt_results=run_benchmark(
            LONG_BENCHMARK_PROMPTS, batch_size=batch_size, title="Long"
        ),
        code_prompt_results=run_benchmark(
            CODE_BENCHMARK_PROMPTS, batch_size=batch_size, title="Code"
        ),
    )


def make_path_safe_file(model_name: str) -> str:
    return "result_" + model_name.replace("/", "_") + ".json"


def run_and_save_dolly_benchmarks(batch_size: int):
    # run and save benchmarks after each run
    results = []
    for model_name in _DOLLY_LIST:
        _logger.info(f"Running benchmarks on {model_name}")
        result = run_benchmarks_on_model(model_name, batch_size)
        with open(make_path_safe_file(model_name), "w") as f:
            json.dump(result.dict(), f, indent=2)

        results.append(result)

    return results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    run_and_save_dolly_benchmarks(1)
