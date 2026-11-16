#!/usr/bin/env python3
"""Train the base-class Faster R-CNN (Phase 2). CPU-friendly: no AMP, num_workers=0.

Modes:
  train   - train epochs, checkpoint every epoch, evaluate on val-tune
  eval    - load a checkpoint, run inference + COCO eval on a split

Examples:
  python scripts/train_detector.py --mode train --config configs/config.yaml --limit-images 200 --epochs 1
  python scripts/train_detector.py --mode eval --config configs/config.yaml --checkpoint outputs/checkpoints/epoch_3.pt --image-ids data/splits/val_test_image_ids.json --val-annotations data/annotations/instances_val_test2017.stripped.json --images-root data/images/val2017 --save-predictions outputs/predictions/detector_val_test.json
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.openworld.categories import label_map_for_base
from src.openworld.dataset import COCODataset, RandomHorizontalFlip, collate_fn
from src.openworld.detector import build_model, load_checkpoint, save_checkpoint
from src.openworld.engine import evaluate_detections, train_one_epoch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "eval"])
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--train-ids", default="data/splits/train_image_ids.json")
    ap.add_argument("--val-ids", default="data/splits/val_tune_image_ids.json")
    ap.add_argument("--train-annotations", default="data/annotations/instances_train2017.stripped.json")
    ap.add_argument("--val-annotations", default="data/annotations/instances_val_tune2017.stripped.json")
    ap.add_argument("--images-root", default="data/images/train2017")
    ap.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    ap.add_argument("--log-file", default="outputs/metrics/train_log.csv")
    ap.add_argument("--save-predictions", default=None)
    ap.add_argument("--checkpoint", default=None, help="checkpoint to load (eval or resume)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit-images", type=int, default=None, help="smoke test: cap train images")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = cfg["project"]["device"]
    seed = cfg["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    label_map, _ = label_map_for_base(args.train_annotations)

    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"]
    lr = args.lr if args.lr is not None else cfg["train"]["lr"]

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    train_ids = json.load(open(args.train_ids))
    if args.limit_images:
        train_ids = train_ids[: args.limit_images]

    train_ds = COCODataset(
        args.train_annotations,
        args.images_root,
        train_ids,
        label_map,
        transforms=RandomHorizontalFlip(p=0.5),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    model = build_model(
        num_classes=cfg["detector"]["num_foreground_classes"] + 1,
        imagenet_backbone=cfg["detector"]["imagenet_backbone"],
        trainable_backbone_layers=cfg["detector"]["trainable_backbone_layers"],
        min_size=cfg["detector"]["min_size"],
        max_size=cfg["detector"]["max_size"],
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_iters = epochs * steps_per_epoch
    warmup = cfg["train"]["warmup_iters"]

    def lr_lambda(it):
        if it < warmup:
            return it / max(warmup, 1)
        progress = (it - warmup) / max(total_iters - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    iteration = 0
    if args.checkpoint:
        ck = load_checkpoint(args.checkpoint, model, optimizer, scheduler, device=device)
        start_epoch = int(ck.get("epoch", 0)) + 1
        iteration = int(ck.get("iteration", 0))
        print(f"Resumed from {args.checkpoint} at epoch {start_epoch}")

    if args.mode == "eval":
        val_ids = json.load(open(args.val_ids))
        val_ds = COCODataset(
            args.val_annotations,
            args.images_root,
            val_ids,
            label_map,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        cat_names = [c["name"] for c in json.load(open(args.val_annotations))["categories"]]
        metrics = evaluate_detections(
            model, val_loader, device, cat_names,
            iou_threshold=cfg["eval"]["iou_threshold"],
            save_predictions=args.save_predictions,
        )
        print(json.dumps(metrics, indent=2))
        return

    log_fields = ["epoch", "iteration", "loss_classifier", "loss_box_reg",
                  "loss_objectness", "loss_rpn_box_reg", "lr"]
    log_rows = []

    best_ap50 = 0.0
    for epoch in range(start_epoch, epochs):
        epoch_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch + 1,
            print_freq=cfg["train"]["print_freq"], scheduler=scheduler,
        )
        iteration += len(train_loader)

        row = {"epoch": epoch + 1, "iteration": iteration}
        for k in log_fields[2:-1]:
            row[k] = round(epoch_metrics.get(k, 0.0), 5)
        row["lr"] = optimizer.param_groups[0]["lr"]
        log_rows.append(row)
        if args.log_file:
            with open(args.log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                if epoch == start_epoch:
                    writer.writeheader()
                writer.writerow(row)

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        save_checkpoint(
            ckpt_dir / f"epoch_{epoch + 1}.pt",
            model, optimizer, scheduler, epoch + 1, iteration,
            rng_state, cfg,
        )
        print(f"Saved checkpoint epoch {epoch + 1}: "
              + " ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items()))

    print("Training complete. Checkpoints in", ckpt_dir)


if __name__ == "__main__":
    main()
