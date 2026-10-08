import dataclasses
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from notmnist.config import PRESETS
from notmnist.data import make_splits, to_tensor
from notmnist.inference import Predictor, preprocess_image
from notmnist.train import train

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "models" / "notmnist_cnn_improved.pt"


@pytest.fixture(scope="module")
def splits():
    return make_splits(seed=42)


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    cfg = dataclasses.replace(PRESETS["cnn_improved"], epochs=1, subset=512)
    ckpt = train(cfg, tmp_path_factory.mktemp("run"), "cpu")
    return Predictor.from_checkpoint(ckpt)


def test_png_roundtrip_matches_tensor(predictor, splits, tmp_path):
    x = splits.x_val[0]
    Image.fromarray(x).save(tmp_path / "a.png")
    p_img = predictor.predict_image(Image.open(tmp_path / "a.png"), top_k=10)
    p_ten = predictor.predict_tensor(to_tensor(x[None]))[0]
    assert p_img[0]["index"] == int(p_ten.argmax())


def test_inverted_rgb_large_image_matches_with_invert(predictor, splits):
    x = splits.x_val[0]
    big = Image.fromarray(255 - x).resize((200, 200)).convert("RGB")
    assert (
        predictor.predict_image(big, top_k=1, invert=True)[0]["index"]
        == predictor.predict_image(Image.fromarray(x), top_k=1, invert=False)[0]["index"]
    )


def test_default_does_not_invert(splits):
    x = splits.x_val[0]
    a = preprocess_image(Image.fromarray(x))
    b = preprocess_image(Image.fromarray(255 - x), invert=True)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, to_tensor(x[None]), atol=1e-6)


@pytest.mark.skipif(not SHIPPED.is_file(), reason="shipped checkpoint not present")
def test_shipped_predictor_agrees_with_evaluate_path(splits):
    from notmnist.evaluate import predict_probs

    x = splits.x_test[splits.test_clean_mask][:200]
    pred = Predictor.from_checkpoint(SHIPPED)
    ref = predict_probs(pred.model, x, "cpu").argmax(axis=1)
    got = []
    for img in x:
        buf = io.BytesIO()
        pil = Image.fromarray(img)
        assert pil.mode == "L"
        pil.save(buf, format="PNG")
        buf.seek(0)
        got.append(pred.predict_image(Image.open(buf), top_k=1)[0]["index"])
    assert (np.asarray(got) == ref).all()


def test_predict_is_deterministic(predictor, splits):
    img = Image.fromarray(splits.x_val[1])
    assert predictor.predict_image(img) == predictor.predict_image(img)


def test_topk_sorted_and_sums(predictor, splits):
    out = predictor.predict_image(Image.fromarray(splits.x_val[2]), top_k=10)
    assert [o["prob"] for o in out] == sorted((o["prob"] for o in out), reverse=True)
    assert sum(o["prob"] for o in out) == pytest.approx(1.0, abs=1e-4)


def test_topk_validation(predictor, splits):
    img = Image.fromarray(splits.x_val[0])
    for bad in (0, 11):
        with pytest.raises(ValueError):
            predictor.predict_image(img, top_k=bad)


def test_inference_does_not_import_training_modules():
    code = (
        "import notmnist.inference, sys; "
        "bad = [m for m in ['notmnist.train','notmnist.data','notmnist.evaluate','notmnist.config'] if m in sys.modules]; "
        "assert not bad, bad"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_cli_missing_checkpoint(tmp_path):
    Image.new("L", (28, 28)).save(tmp_path / "a.png")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run(
        [sys.executable, "-m", "notmnist.predict", str(tmp_path / "a.png"),
         "--checkpoint", str(tmp_path / "nope.pt")],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert "nope.pt" in r.stderr
