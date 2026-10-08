"""Checkpoint-based inference. Deliberately free of training/data dependencies."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from notmnist import CLASS_NAMES
from notmnist.models import load_checkpoint


def preprocess_image(img: Image.Image, invert: bool = False) -> torch.Tensor:
    """PIL image -> (1,1,28,28) float32 in [0,1], light glyph on dark background.

    The model expects the training polarity (light glyph on dark background), so by default
    the pixels are used as-is. Dark-on-light inputs (scans, hand-drawn letters on white)
    need invert=True. There is no auto-detection: a mean-intensity heuristic inverted many
    bold, filled notMNIST glyphs and cost ~14 pp of accuracy on test_clean.

    RGBA images are composited onto white, which makes transparent backgrounds light; that
    is only correct together with invert=True (a dark glyph drawn on a transparent canvas).
    """
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("L")
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return torch.from_numpy(np.ascontiguousarray(arr)).reshape(1, 1, 28, 28)


class Predictor:
    def __init__(self, model, class_names, model_name: str, device: str = "cpu"):
        self.model = model
        self.class_names = list(class_names)
        self.model_name = model_name
        self.device = device

    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu") -> "Predictor":
        model, ckpt = load_checkpoint(path, device)
        return cls(model, ckpt.get("class_names", CLASS_NAMES), ckpt["model_name"], device)

    def predict_tensor(self, x: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            logits = self.model(x.to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict_image(self, img: Image.Image, top_k: int = 3, invert: bool = False) -> list[dict]:
        if not 1 <= top_k <= len(self.class_names):
            raise ValueError(f"top_k must be in [1, {len(self.class_names)}], got {top_k}")
        probs = self.predict_tensor(preprocess_image(img, invert))[0]
        order = np.argsort(-probs, kind="stable")[:top_k]
        return [
            {"label": str(self.class_names[i]), "index": int(i), "prob": float(probs[i])}
            for i in order
        ]
