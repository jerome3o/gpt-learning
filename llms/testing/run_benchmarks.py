from typing import Callable, List
from pathlib import Path
from pydantic import BaseModel
import json
import logging
from tqdm import tqdm
import gc
import torch

from load_stablelm import load_stablelm
from load_dolly import load_dolly
from load_openassistant import load_openassistant
from benchmark_prompts import (
    BRIEF_BENCHMARK_PROMPTS,
    CODE_BENCHMARK_PROMPTS,
    LONG_BENCHMARK_PROMPTS,
)

_logger = logging.getLogger(__name__)

_MODEL_LIST = [
    # "databricks/dolly-v2-3b",
    # "databricks/dolly-v1-6b",
    # "databricks/dolly-v2-7b",
    # "databricks/dolly-v2-12b",
    # "OpenAssistant/stablelm-7b-sft-v7-epoch-3",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "StabilityAI/stablelm-tuned-alpha-3b",
    "StabilityAI/stablelm-tuned-alpha-7b",
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
    if "dolly" in model_name.lower():
        return load_dolly(model_name)

    if "oasst" in model_name.lower() or "openassistant" in model_name.lower():
        return load_openassistant(model_name)

    if "stablelm" in model_name.lower():
        return load_stablelm(model_name)

    raise ValueError(f"Unknown model {model_name}")


def _load_if_exists(i: int, title: str, model: str) -> PromptBatchResult:
    key = ".cache/" + make_path_safe_file(f"{title}_{i}_{model}")
    if not Path(key).exists():
        return None, key

    with open(key, "r") as f:
        return PromptBatchResult(**json.load(f)), key


def run_benchmarks_on_model(model_name: str, batch_size: int) -> ModelBenchmarkResult:
    model = load_model(model_name)

    def run_benchmark(prompts: List[str], batch_size: int, title: str) -> List[PromptBatchResult]:
        _logger.info(f"Running {title} benchmarks")
        results = []
        for i, prompt in tqdm(list(enumerate(prompts))):
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            _cached_result, fn = _load_if_exists(i, title, model_name)
            if _cached_result:
                results.append(_cached_result)
                continue

            outputs = model(prompt, batch_size)

            prompt_batch_result = PromptBatchResult(
                prompt=prompt,
                outputs=outputs,
            )
            Path(fn).parent.mkdir(parents=True, exist_ok=True)
            with open(fn, "w") as f:
                json.dump(prompt_batch_result.dict(), f, indent=2)

            results.append(prompt_batch_result)
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


def run_and_save_benchmarks(batch_size: int):
    # run and save benchmarks after each run
    results = []
    for model_name in _MODEL_LIST:
        _logger.info(f"Running benchmarks on {model_name}")
        result = run_benchmarks_on_model(model_name, batch_size)
        with open(make_path_safe_file(model_name), "w") as f:
            json.dump(result.dict(), f, indent=2)

        results.append(result)

    return results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    run_and_save_benchmarks(2)
