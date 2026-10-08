"""Training configuration and the named presets (PROJECT_GUIDE §5.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    preset: str
    model_name: str
    pretrained: bool = False
    optimizer: str = "adam"  # "adam" | "adamw"
    lr: float = 1e-3
    backbone_lr: float | None = None
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 5
    scheduler: str = "none"  # "none" | "cosine"
    early_stopping_patience: int | None = None
    freeze_backbone_epochs: int = 0
    seed: int = 42
    subset: int | None = None


_FINETUNE = dict(
    model_name="resnet18", optimizer="adamw", lr=1e-3, weight_decay=1e-4,
    batch_size=128, epochs=10, scheduler="cosine", early_stopping_patience=3,
)

PRESETS: dict[str, TrainConfig] = {
    "mlp_baseline": TrainConfig("mlp_baseline", "mlp_baseline", batch_size=64, epochs=5),
    "cnn_baseline": TrainConfig("cnn_baseline", "cnn_baseline", batch_size=32, epochs=50),
    "cnn_improved": TrainConfig(
        "cnn_improved", "cnn_improved", optimizer="adamw", weight_decay=5e-4,
        batch_size=128, epochs=30, scheduler="cosine", early_stopping_patience=5,
    ),
    "resnet18_probe": TrainConfig(
        "resnet18_probe", "resnet18", pretrained=True, batch_size=128, epochs=5,
        freeze_backbone_epochs=5,
    ),
    "resnet18_finetune": TrainConfig(
        "resnet18_finetune", pretrained=True, backbone_lr=1e-4, freeze_backbone_epochs=1,
        **_FINETUNE,
    ),
    "resnet18_scratch": TrainConfig("resnet18_scratch", pretrained=False, **_FINETUNE),
}
