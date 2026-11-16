"""Faster R-CNN (ResNet-50 + FPN) construction and checkpoint IO."""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Dict, Optional
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def build_model(
    num_classes: int = 16,
    imagenet_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 512,
    max_size: int = 683,
) -> torch.nn.Module:
    """Build Faster R-CNN with an ImageNet backbone and NO detector pretraining.

    num_classes = 15 foreground + 1 background.
    """
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if imagenet_backbone else None
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )
    return model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    iteration: int,
    rng_state: Dict,
    config: Dict,
    scaler: Optional[object] = None,
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "iteration": iteration,
        "rng": rng_state,
        "config": config,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[object] = None,
    device: str = "cpu",
) -> Dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
