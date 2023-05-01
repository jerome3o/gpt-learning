from typing import Union, Tuple, Callable
from pathlib import Path
import torch
import tokenizers
import tqdm

_file_dir = Path(__file__).resolve().parent


_DEFAULT_IMDB_DATA_PATH = _file_dir / "data" / "imdb_data.pt"

# TODO(j.swannack): Add torch data loader integration


def load_sentiment_data(
    data_path: Union[Path, str] = None,
    test_fraction: float = 0.2,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    data_path = data_path or _DEFAULT_IMDB_DATA_PATH
    data_path = Path(data_path)

    # load in tokenized data
    data_dict = torch.load("data/imdb_data.pt")
    data = data_dict["reviews"]
    labels = data_dict["labels"]
    lengths = data_dict["lengths"]

    # split into train and test by 80:20
    training_fraction = 1 - test_fraction

    def _split_data(data: torch.Tensor, fraction: int):
        return data[: int(len(data) * fraction)], data[int(len(data) * fraction) :]

    train_data, test_data = _split_data(data, training_fraction)
    train_labels, test_labels = _split_data(labels, training_fraction)
    train_lengths, test_lengths = _split_data(lengths, training_fraction)

    return (
        train_data,
        train_labels,
        train_lengths,
        test_data,
        test_labels,
        test_lengths,
    )


def load_tokenizer() -> tokenizers.Tokenizer:
    return tokenizers.Tokenizer.from_file("models/tokenizer.json")


def calc_accuracy(
    model: torch.nn.Module,
    loss_function: Callable[..., float],
    data: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    batch_size: torch.Tensor,
    device: str,
):
    cum_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            input = data[i : i + batch_size].to(device)
            _labels = labels[i : i + batch_size].to(device)
            _lengths = lengths[i : i + batch_size]

            output = model(input, _lengths)
            loss = loss_function(output, _labels)

            cum_loss += loss
            _, y_pred = torch.max(output, dim=1)
            correct += sum(_labels == y_pred)
            total += len(_labels)

    return correct, total, cum_loss
