from typing import Callable, List


from transformers import AutoTokenizer, AutoModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline


def load_dolly(model_name: str) -> Callable[[str, int], List[str]]:
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

    def _generate_text(prompt: str, batch_size: int = None) -> List[str]:
        return [o["generated_text"] for o in generate_text(prompt, batch_size=batch_size)]

    return _generate_text
