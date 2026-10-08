FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first so the later install sees it as already satisfied.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir ".[api]"

COPY api/ api/
COPY models/ models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
