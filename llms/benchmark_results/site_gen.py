import os
import json
from typing import List
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader
import glob


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


def load_json_files(directory: str) -> List[ModelBenchmarkResult]:
    file_paths = glob.glob(os.path.join(directory, "*.json"))
    results = []

    for file_path in file_paths:
        with open(file_path) as f:
            data = json.load(f)
            result = ModelBenchmarkResult.parse_obj(data)
            results.append(result)

    return results


def generate_html_files(results: List[ModelBenchmarkResult], output_dir: str):
    env = Environment(loader=FileSystemLoader("templates"))

    for result in results:
        html = env.get_template("model_benchmark_result.html").render(result=result)
        file_name = f"{result.model_name}.html"
        output_path = os.path.join(output_dir, file_name)

        with open(output_path, "w") as f:
            f.write(html)


def main():
    input_dir = "json_files"
    output_dir = "output_html"
    os.makedirs(output_dir, exist_ok=True)

    results = load_json_files(input_dir)
    generate_html_files(results, output_dir)


if __name__ == "__main__":
    main()
