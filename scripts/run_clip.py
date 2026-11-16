#!/usr/bin/env python3
"""CLIP region recognition (Phase 4).

Modes:
  gt        - perfect-localization upper bound: CLIP on ground-truth boxes
  proposals - CLIP on RPN proposals, fused with objectness via alpha

Outputs: predictions.json per run directory.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from PIL import Image

from src.openworld.clip_regions import (
    CLIPRecognizer, fused_scores, square_context_crop,
)
from src.openworld.proposals import load_proposals
from pycocotools.coco import COCO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gt", "proposals"])
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default=None, help="detector checkpoint (proposals mode)")
    ap.add_argument("--image-ids", required=True)
    ap.add_argument("--annotations", required=True, help="stripped val json (gt boxes source)")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--proposals-dir", default=None, help="folder with per-image .npz (proposals mode)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expansion", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = cfg["project"]["device"]

    expansion = args.expansion if args.expansion is not None else cfg["clip"]["context_expansion"]
    top_k = args.top_k if args.top_k is not None else cfg["clip"]["top_proposals"]
    alpha = args.alpha if args.alpha is not None else cfg["clip"]["alpha"]
    template = cfg["clip"]["template"]

    sel = json.load(open("data/splits/selected_categories.json"))
    all_names = [c["name"] for c in sel["base"]] + [c["name"] for c in sel["novel"]]
    name_to_orig = {c["name"]: int(c["id"]) for c in sel["base"] + sel["novel"]}

    recognizer = CLIPRecognizer(
        cfg["clip"]["model"], cfg["clip"]["pretrained"], device=device
    )
    text_emb = recognizer.encode_text(all_names, template=template)

    coco = COCO(args.annotations)
    image_ids = json.load(open(args.image_ids))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for iid in image_ids:
        img_info = coco.loadImgs(int(iid))[0]
        image_path = Path(args.images_root) / img_info["file_name"]
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")

        if args.mode == "gt":
            ann_ids = coco.getAnnIds(imgIds=int(iid))
            anns = coco.loadAnns(ann_ids)
            boxes = []
            meta = []
            for ann in anns:
                if int(ann.get("iscrowd", 0)) == 1:
                    continue
                x, y, w, h = [float(v) for v in ann["bbox"]]
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                meta.append({"image_id": int(iid), "gt_category_id": int(ann["category_id"]),
                             "bbox": boxes[-1], "gt": True})
            objectness = torch.ones(len(boxes))
        else:
            data = load_proposals(int(iid), args.proposals_dir)
            boxes = data["boxes"][:top_k].tolist()
            objectness = torch.tensor(data["scores"][:top_k], dtype=torch.float32)
            meta = [{"image_id": int(iid), "bbox": b} for b in boxes]

        if not boxes:
            continue

        crops = [square_context_crop(image, b, expansion=expansion) for b in boxes]
        region_emb = recognizer.encode_regions(crops, batch_size=args.batch_size)
        sims = recognizer.similarity(region_emb, text_emb)
        probs = recognizer.class_probs(sims)
        scores = fused_scores(objectness.to(probs.device), probs, alpha=alpha)

        for r in range(scores.shape[0]):
            best_c = int(scores[r].argmax())
            entry = dict(meta[r])
            entry["score"] = float(scores[r, best_c])
            entry["category_id"] = name_to_orig[all_names[best_c]]
            entry["bbox"] = [round(float(v), 2) for v in meta[r]["bbox"]]
            predictions.append(entry)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"mode={args.mode} images={len(image_ids)} predictions={len(predictions)}")
    print(f"saved -> {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
