import csv
import json
from dataclasses import replace

import torch

from notmnist import CLASS_NAMES
from notmnist.config import PRESETS
from notmnist.models import build_model, load_checkpoint
from notmnist.train import train
from notmnist.utils import seed_everything

CPU = torch.device("cpu")


def test_smoke_run_writes_artifacts(tmp_path):
    cfg = replace(PRESETS["cnn_improved"], epochs=1, subset=512)
    ckpt = train(cfg, tmp_path, device=CPU)
    assert ckpt == tmp_path / "best.pt"
    for f in ["config.json", "history.csv", "best.pt"]:
        assert (tmp_path / f).exists()
    meta = json.loads((tmp_path / "config.json").read_text())
    assert meta["config"]["preset"] == "cnn_improved"
    assert set(meta) == {"config", "device", "torch_version", "split_info", "n_params"}
    with open(tmp_path / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert list(rows[0]) == ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]


def test_checkpoint_roundtrip(tmp_path):
    cfg = replace(PRESETS["resnet18_scratch"], epochs=1, subset=256)
    ckpt = train(cfg, tmp_path, device=CPU)
    m, meta = load_checkpoint(ckpt)
    assert set(meta) == {"model_name", "state_dict", "config", "class_names", "epoch", "val_acc"}
    assert meta["class_names"] == CLASS_NAMES
    assert not m.training
    saved = torch.load(ckpt, weights_only=True)["state_dict"]
    loaded = m.state_dict()
    assert saved.keys() == loaded.keys()
    for k, v in saved.items():
        assert torch.equal(loaded[k], v), k
    x = torch.rand(3, 1, 28, 28)
    assert torch.allclose(m(x), m(x))


def test_frozen_backbone_bn_unchanged(tmp_path):
    cfg = replace(PRESETS["resnet18_probe"], pretrained=False, epochs=1, subset=256)
    ckpt = train(cfg, tmp_path, device=CPU)
    seed_everything(cfg.seed)
    before = build_model("resnet18").state_dict()
    after = torch.load(ckpt, weights_only=True)["state_dict"]
    bk = [k for k in before if k.startswith("backbone.")]
    assert any("running_mean" in k for k in bk)
    for k in bk:  # weights (frozen) and BN buffers (eval mode) must be untouched
        assert torch.equal(before[k], after[k]), k
    assert not torch.equal(before["head.weight"], after["head.weight"])


def test_checkpoint_state_dict_is_cpu(tmp_path):
    cfg = replace(PRESETS["cnn_improved"], epochs=1, subset=128)
    ckpt = train(cfg, tmp_path, device=CPU)
    sd = torch.load(ckpt, weights_only=True)["state_dict"]
    assert all(v.device.type == "cpu" for v in sd.values())
