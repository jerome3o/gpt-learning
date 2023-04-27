import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

MODELS = [
    "StabilityAI/stablelm-tuned-alpha-3b",
    "StabilityAI/stablelm-tuned-alpha-7b",
]


def load_stablelm(model_name: str, load_in_8bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=load_in_8bit,
    )

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """
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

    return generate_text


def main():
    generate_text = load_stablelm(MODELS[0])
    resp = generate_text("heeey", 2)
    print(resp)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
