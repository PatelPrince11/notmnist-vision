import pytest
import torch
from torch import nn

from notmnist.models import MODEL_NAMES, build_model, count_params


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_output_shape(name):
    m = build_model(name).eval()
    assert m(torch.rand(2, 1, 28, 28)).shape == (2, 10)


def test_ports_match_keras_param_counts():
    assert count_params(build_model("mlp_baseline")) == 203_530
    assert count_params(build_model("cnn_baseline")) == 462_858


def test_improved_cnn_is_smaller_than_baseline():
    assert count_params(build_model("cnn_improved")) < 462_858


def test_resnet_has_head_and_backbone():
    m = build_model("resnet18")
    assert isinstance(m.head, nn.Linear) and m.head.out_features == 10
    assert isinstance(m.backbone, nn.Module)
    assert not any(k.startswith(("mean", "std")) for k in m.state_dict())


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("nope")
