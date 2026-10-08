"""Training loop with presets, early stopping, backbone freeze/unfreeze and run directories."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from notmnist import CLASS_NAMES
from notmnist.config import PRESETS, TrainConfig
from notmnist.data import make_loader, make_splits
from notmnist.models import build_model, count_params
from notmnist.utils import REPO_ROOT, get_device, seed_everything

HISTORY_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Toggle requires_grad on `model.backbone`; a frozen backbone is also put in eval mode."""
    for p in model.backbone.parameters():
        p.requires_grad_(trainable)
    if not trainable:
        model.backbone.eval()


def _make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    backbone = getattr(model, "backbone", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg.backbone_lr is None:
        groups = [{"params": trainable, "lr": cfg.lr}]
    else:
        groups = [{"params": [p for p in trainable if id(p) not in backbone_ids], "lr": cfg.lr}]
        bb = [p for p in trainable if id(p) in backbone_ids]
        if bb:
            groups.append({"params": bb, "lr": cfg.backbone_lr})
    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg.optimizer]
    return opt_cls(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _run_epoch(model, loader, loss_fn, device, optimizer=None, freeze_backbone=False):
    """One pass over `loader`; trains if `optimizer` is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        model.backbone.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train(cfg: TrainConfig, out_dir: Path, device: torch.device | None = None) -> Path:
    """Train `cfg`, writing config.json, history.csv and best.pt into `out_dir`; return best.pt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or get_device()
    generator = seed_everything(cfg.seed)

    s = make_splits(seed=cfg.seed)
    x_tr, y_tr, x_va, y_va = s.x_train, s.y_train, s.x_val, s.y_val
    if cfg.subset is not None:
        rng = np.random.default_rng(cfg.seed)
        tr = rng.permutation(len(x_tr))[: cfg.subset]
        va = rng.permutation(len(x_va))[: min(cfg.subset, len(x_va))]
        x_tr, y_tr, x_va, y_va = x_tr[tr], y_tr[tr], x_va[va], y_va[va]
    train_loader = make_loader(x_tr, y_tr, cfg.batch_size, shuffle=True, generator=generator)
    val_loader = make_loader(x_va, y_va, cfg.batch_size, shuffle=False)

    model = build_model(cfg.model_name, pretrained=cfg.pretrained).to(device)
    (out_dir / "config.json").write_text(json.dumps({
        "config": asdict(cfg),
        "device": str(device),
        "torch_version": torch.__version__,
        "split_info": s.info,
        "n_params": count_params(model),
    }, indent=2))

    loss_fn = nn.CrossEntropyLoss()
    n_frozen = min(cfg.freeze_backbone_epochs, cfg.epochs)
    phases = [(True, n_frozen), (False, cfg.epochs - n_frozen)]
    best_path = out_dir / "best.pt"
    best_loss, bad_epochs, epoch = float("inf"), 0, 0

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for frozen, n_epochs in phases:
            if n_epochs == 0:
                continue
            if n_frozen:
                set_backbone_trainable(model, trainable=not frozen)
            optimizer = _make_optimizer(model, cfg)
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
                if cfg.scheduler == "cosine" else None
            )
            for _ in range(n_epochs):
                epoch += 1
                t0 = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"]
                tr_loss, tr_acc = _run_epoch(model, train_loader, loss_fn, device, optimizer, frozen)
                va_loss, va_acc = _run_epoch(model, val_loader, loss_fn, device)
                if scheduler is not None:
                    scheduler.step()
                secs = time.perf_counter() - t0
                writer.writerow(dict(zip(HISTORY_FIELDS, [
                    epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}",
                    f"{lr:.2e}", f"{secs:.1f}",
                ])))
                f.flush()
                print(f"[{cfg.preset}] ep {epoch:3d}/{cfg.epochs} train {tr_loss:.4f}/{tr_acc:.4f} "
                      f"val {va_loss:.4f}/{va_acc:.4f} lr {lr:.1e} {secs:.1f}s", flush=True)

                improved = va_loss < best_loss
                if improved:
                    best_loss, bad_epochs = va_loss, 0
                else:
                    bad_epochs += 1
                if cfg.early_stopping_patience is None or improved:
                    torch.save({
                        "model_name": cfg.model_name,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": asdict(cfg),
                        "class_names": list(CLASS_NAMES),
                        "epoch": epoch,
                        "val_acc": va_acc,
                    }, best_path)
                if cfg.early_stopping_patience is not None and bad_epochs >= cfg.early_stopping_patience:
                    print(f"[{cfg.preset}] early stop at epoch {epoch} (best val loss {best_loss:.4f})")
                    return best_path
    if not best_path.exists():
        raise RuntimeError(f"training produced no checkpoint ({best_path}); val loss never improved (NaN?)")
    return best_path


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train a notMNIST preset.")
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset", type=_positive_int, default=None)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    a = p.parse_args(argv)
    cfg = replace(PRESETS[a.preset], seed=a.seed, subset=a.subset)
    name = f"{a.preset}_s{a.seed}" + (f"_sub{a.subset}" if a.subset is not None else "")
    best = train(cfg, REPO_ROOT / "runs" / name, device=get_device(a.device))
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
