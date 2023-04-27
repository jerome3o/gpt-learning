import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

MODELS = ["OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"]


def load_openassistant(model_name: str, load_in_8bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=load_in_8bit,
    )
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

    return generate_text


def main():
    generate_text = load_openassistant(MODELS[0])
    resp = generate_text("heeey", 2)
    print(resp)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
