from typing import Union, Tuple
from pathlib import Path
import torch

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
