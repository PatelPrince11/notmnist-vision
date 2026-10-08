"""CLI: python -m notmnist.predict IMAGE [--checkpoint PATH] [--top-k 3] [--invert]"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from notmnist.inference import Predictor

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "notmnist_cnn_improved.pt"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m notmnist.predict", description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--invert", action="store_true",
                   help="invert pixels for dark-on-light images (default: off; the model "
                        "expects a light glyph on a dark background)")
    args = p.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2
    try:
        predictor = Predictor.from_checkpoint(args.checkpoint)
        with Image.open(args.image) as img:
            preds = predictor.predict_image(img, top_k=args.top_k, invert=args.invert)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"model": predictor.model_name, "predictions": preds}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
