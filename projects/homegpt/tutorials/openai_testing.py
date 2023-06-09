import os
import openai

# load in the LLM_URL from the environment
openai.api_base = os.environ["LLM_URL"]


def main():
    response = openai.Completion.create(
        model="vicuna-13b-v1.1-8bit",
        prompt="Write a poem about frogs, include murder",
        temperature=0,
        max_tokens=2000,
    )

    print(response)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
