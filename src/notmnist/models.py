"""Model zoo. All models take float32 (B,1,28,28) in [0,1] and return logits (B,10)."""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

MODEL_NAMES = ("mlp_baseline", "cnn_baseline", "cnn_improved", "resnet18")


class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.1),
    )


class CNNImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            _block(1, 32), _block(32, 64), _block(64, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.backbone(x))


class ResNet18Transfer(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, 10)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.head(self.backbone(x))


def build_model(name: str, pretrained: bool = False) -> nn.Module:
    if name == "mlp_baseline":
        return MLPBaseline()
    if name == "cnn_baseline":
        return CNNBaseline()
    if name == "cnn_improved":
        return CNNImproved()
    if name == "resnet18":
        return ResNet18Transfer(pretrained=pretrained)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, device="cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a training checkpoint; returns (eval-mode model on device, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
