"""Training loop and closed-set (base-class) evaluation."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as TF


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    print_freq: int = 50,
    scheduler=None,
) -> Dict[str, float]:
    model.train()
    loss_sum: Dict[str, float] = {}
    n = 0
    t0 = time.time()

    for images, targets in data_loader:
        images = [TF.to_tensor(img).to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k, v in loss_dict.items():
            loss_sum[k] = loss_sum.get(k, 0.0) + float(v.detach())
        n += 1
        if n % print_freq == 0:
            elapsed = time.time() - t0
            avg = " ".join(f"{k}={v / n:.4f}" for k, v in loss_sum.items())
            print(f"[epoch {epoch}] iter {n} | {avg} | {elapsed:.1f}s")

    return {k: v / max(n, 1) for k, v in loss_sum.items()}


@torch.no_grad()
def evaluate_detections(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    category_names: List[str],
    iou_threshold: float = 0.5,
    save_predictions: Optional[str | Path] = None,
) -> Dict:
    """COCO-style evaluation on the base classes."""
    model.eval()
    coco_gt: COCO = data_loader.dataset.coco
    base_cat_ids = sorted(data_loader.dataset.label_map.keys())
    inv_label_map = {v: k for k, v in data_loader.dataset.label_map.items()}

    results = []
    for images, targets in data_loader:
        images = [TF.to_tensor(img).to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"])
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [round(float(v), 2) for v in box]
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(inv_label_map[int(label)]),
                        "bbox": [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)],
                        "score": round(float(score), 4),
                    }
                )

    metrics: Dict = {}
    if results:
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.catIds = base_cat_ids
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        s = evaluator.stats
        metrics = {
            "AP50": float(s[1]),
            "AP50:95": float(s[0]),
            "AR1": float(s[7]),
            "AR10": float(s[8]),
            "AR100": float(s[9]),
        }

        per_class = {}
        for cat_id in base_cat_ids:
            ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
            ev.params.catIds = [cat_id]
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
            name = [c["name"] for c in coco_gt.loadCats([cat_id])][0]
            per_class[name] = float(ev.stats[1]) if ev.stats is not None else 0.0
        metrics["per_class_AP50"] = per_class

    if save_predictions:
        Path(save_predictions).parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(save_predictions, "w") as f:
            json.dump(results, f, indent=2)

    return metrics
