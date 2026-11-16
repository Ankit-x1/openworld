#!/usr/bin/env python3
"""Extract RPN proposals with objectness scores (Phase 3).

Saves one .npz per image: boxes [K,4] xyxy, scores [K].
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from src.openworld.categories import label_map_for_base
from src.openworld.dataset import COCODataset, collate_fn
from src.openworld.detector import build_model, load_checkpoint
from src.openworld.proposals import extract_proposals, save_proposals


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-ids", default="data/splits/val_tune_image_ids.json")
    ap.add_argument("--annotations", default="data/annotations/instances_val_tune2017.stripped.json")
    ap.add_argument("--images-root", default="data/images")
    ap.add_argument("--out-dir", default="outputs/proposals/val_tune")
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--pre-nms-top-n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = cfg["project"]["device"]

    label_map, _ = label_map_for_base(args.annotations)
    image_ids = json.load(open(args.image_ids))

    ds = COCODataset(args.annotations, args.images_root, image_ids, label_map)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    model = build_model(
        num_classes=cfg["detector"]["num_foreground_classes"] + 1,
        imagenet_backbone=cfg["detector"]["imagenet_backbone"],
        min_size=cfg["detector"]["min_size"],
        max_size=cfg["detector"]["max_size"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for images, targets in loader:
        images = [TF.to_tensor(im).to(device) for im in images]
        results = extract_proposals(
            model, images, None,
            pre_nms_top_n=args.pre_nms_top_n,
            post_nms_top_n=args.top_n,
        )
        batch_ids = [int(t["image_id"]) for t in targets]
        save_proposals(results, batch_ids, out_dir)
        saved += len(results)
        print(f"\rprocessed {saved}/{len(image_ids)}", end="")
    print(f"\nSaved {saved} proposal files -> {out_dir}")


if __name__ == "__main__":
    main()
