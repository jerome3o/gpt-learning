import bitsandbytes as bnb
import torch


def main():
    fp16_model = torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.Linear(64, 64),
    )
    torch.save(fp16_model.state_dict(), "fp16_model.pt")

    int8_model = torch.nn.Sequential(
        bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False),
        bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False),
    )
    int8_model.load_state_dict(torch.load("fp16_model.pt"))
    int8_model = int8_model.to("cuda")
    print(int8_model)
    input_vector = (
        torch.tensor([1.0] * 64, dtype=torch.float32)
        .to(torch.float16)
        .view(1, 64)
        .to("cuda")
    )
    output_vector = int8_model(input_vector)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
