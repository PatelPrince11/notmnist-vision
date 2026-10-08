import torch

from notmnist.utils import DATA_PATH, get_device, seed_everything


def test_seed_everything_reproducible():
    seed_everything(0)
    a = torch.rand(3)
    seed_everything(0)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_get_device_prefer_cpu():
    assert get_device("cpu").type == "cpu"


def test_data_file_exists():
    assert DATA_PATH.exists()
