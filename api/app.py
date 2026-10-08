"""FastAPI service: uvicorn api.app:app"""
import io
import os
import struct
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from notmnist.inference import Predictor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "notmnist_cnn_improved.pt"
MAX_BYTES = 1_048_576
MAX_PIXELS = 4096 * 4096


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("NOTMNIST_CHECKPOINT", DEFAULT_CHECKPOINT))
    app.state.predictor = Predictor.from_checkpoint(path)
    yield


app = FastAPI(title="notMNIST classifier", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model": request.app.state.predictor.model_name}


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10),
    invert: bool = Query(False, description="invert pixels for dark-on-light images"),
):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="file larger than 1 MB")
    try:
        img = Image.open(io.BytesIO(data))
        if img.size[0] * img.size[1] > MAX_PIXELS:
            raise HTTPException(status_code=400, detail="image dimensions too large")
        img.load()
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        EOFError,
        Image.DecompressionBombError,
        struct.error,
    ):
        raise HTTPException(status_code=400, detail="not a valid image") from None
    predictor = request.app.state.predictor
    preds = predictor.predict_image(img, top_k=top_k, invert=invert)
    return {"model": predictor.model_name, "predictions": preds}
