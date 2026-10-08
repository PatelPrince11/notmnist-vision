import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app
from notmnist import CLASS_NAMES
from notmnist.data import make_splits


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "cnn_improved"}


def test_predict_returns_sorted_correct_top1(client):
    s = make_splits(seed=42)
    x, y = s.x_test[0], int(s.y_test[0])
    assert CLASS_NAMES[y] == "F" and s.test_clean_mask[0]
    r = client.post("/predict", files={"file": ("a.png", _png(x), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "cnn_improved"
    preds = body["predictions"]
    assert len(preds) == 3
    probs = [p["prob"] for p in preds]
    assert probs == sorted(probs, reverse=True)
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in probs)
    assert preds[0]["label"] == CLASS_NAMES[y]


def test_non_image_400(client):
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_top_k_too_large_422(client):
    r = client.post("/predict?top_k=11", files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 422


def test_too_large_413(client):
    r = client.post(
        "/predict", files={"file": ("a.png", b"0" * (1_048_576 + 1), "image/png")}
    )
    assert r.status_code == 413


def test_huge_declared_dimensions_400(client):
    data = bytearray(_png(__import__("numpy").zeros((4, 4), dtype="uint8")))
    # patch IHDR width/height (bytes 16..24) and recompute its CRC
    data[16:24] = struct.pack(">II", 20000, 20000)
    data[29:33] = struct.pack(">I", zlib.crc32(bytes(data[12:29])))
    r = client.post("/predict", files={"file": ("a.png", bytes(data), "image/png")})
    assert r.status_code == 400


def test_predict_invert_true_on_dark_on_light_rgb(client):
    s = make_splits(seed=42)
    x, y = s.x_test[0], int(s.y_test[0])
    big = Image.fromarray(255 - x).resize((200, 200)).convert("RGB")
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    r = client.post(
        "/predict?invert=true&top_k=1", files={"file": ("a.png", buf.getvalue(), "image/png")}
    )
    assert r.status_code == 200
    assert r.json()["predictions"][0]["label"] == CLASS_NAMES[y]
