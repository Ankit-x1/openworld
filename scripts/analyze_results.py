#!/usr/bin/env python3
"""Failure analysis gallery (Phase 5).

Writes per-size AP and a ranked failure JSON; saves annotated crops for the
worst cases so you can inspect why the pipeline failed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.openworld.evaluation import build_gt, group_by_image, per_class_ap
from src.openworld.visualization import draw_boxes
from pycocotools.coco import COCO
from torchvision.ops import box_iou
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--selected-categories", default="data/splits/selected_categories.json")
    ap.add_argument("--image-ids", default="data/splits/val_test_image_ids.json")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--top", type=int, default=12)
    return ap.parse_args()


def size_bucket(w, h):
    area = w * h
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = json.load(open(args.selected_categories))
    name_to_orig = {c["name"]: int(c["id"]) for c in sel["base"] + sel["novel"]}
    orig_to_name = {v: k for k, v in name_to_orig.items()}
    all_names = [c["name"] for c in sel["base"] + sel["novel"]]

    predictions = json.load(open(args.predictions))
    coco = COCO(args.annotations)
    image_ids = json.load(open(args.image_ids))
    id_to_name = {int(c["id"]): c["name"] for c in coco.loadCats(coco.getCatIds())}

    gt_by_image = build_gt(coco, image_ids)
    pred_by_image = group_by_image(predictions)

    failures = []
    per_size = {"small": [], "medium": [], "large": []}

    for iid in image_ids:
        gts = gt_by_image.get(iid, [])
        preds = pred_by_image.get(iid, [])
        img_info = coco.loadImgs(int(iid))[0]

        for g in gts:
            best_j, best_iou = -1, args.iou_thr
            for j, p in enumerate(preds):
                iou = box_iou(
                    torch.tensor([g["bbox"]]), torch.tensor([p["bbox"]])
                ).item()
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j < 0:
                failures.append({
                    "image_id": iid, "file_name": img_info["file_name"],
                    "type": "missed (false negative)",
                    "gt_class": orig_to_name.get(int(g["category_id"]), str(g["category_id"])),
                    "gt_bbox": g["bbox"],
                    "bucket": size_bucket(g["bbox"][2], g["bbox"][3]),
                })

        for p in preds:
            matched = False
            for g in gts:
                if box_iou(torch.tensor([p["bbox"]]), torch.tensor([g["bbox"]])).item() >= args.iou_thr:
                    matched = True
                    break
            if not matched:
                failures.append({
                    "image_id": iid, "file_name": img_info["file_name"],
                    "type": "false positive",
                    "pred_class": id_to_name.get(int(p["category_id"]), str(p["category_id"])),
                    "pred_bbox": p["bbox"],
                    "score": p.get("score"),
                    "bucket": size_bucket(p["bbox"][2] - p["bbox"][0], p["bbox"][3] - p["bbox"][1]),
                })

    failures.sort(key=lambda f: f.get("score", 0) if f.get("score") is not None else 0)
    top_failures = failures[: args.top]

    size_ap = {}
    for bucket in ("small", "medium", "large"):
        subset_ids = []
        for iid in image_ids:
            if any(size_bucket(g["bbox"][2], g["bbox"][3]) == bucket for g in gt_by_image.get(iid, [])):
                subset_ids.append(iid)
        filt_preds = [p for p in predictions if int(p["image_id"]) in set(subset_ids)]
        filt_gt = {iid: gt_by_image.get(iid, []) for iid in subset_ids}
        aps = []
        for name in [c["name"] for c in sel["novel"]]:
            cid = name_to_orig[name]
            aps.append(per_class_ap(
                group_by_image(filt_preds), filt_gt, cid, args.iou_thr,
            ))
        size_ap[bucket] = float(np.mean(aps)) if aps else 0.0

    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(top_failures):
        image_path = Path(args.images_root) / f["file_name"]
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        box = f.get("gt_bbox") or f.get("pred_bbox")
        x1, y1, x2, y2 = [float(v) for v in box]
        pad = 40
        crop = image.crop((
            max(0, int(x1 - pad)), max(0, int(y1 - pad)),
            min(image.width, int(x2 + pad)), min(image.height, int(y2 + pad)),
        ))
        labels = [f["type"].upper() + " | " + (f.get("gt_class") or f.get("pred_class"))]
        scores = [f.get("score")] if f.get("score") is not None else None
        draw_boxes(crop, [box], labels, scores).save(gallery_dir / f"fail_{k:02d}.png")

    summary = {
        "num_failures": len(failures),
        "top_failures": top_failures,
        "novel_AP50_by_size": size_ap,
    }
    with open(out_dir / "failure_analysis.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"failures: {len(failures)} total, {len(top_failures)} saved to {gallery_dir}")
    print("novel AP50 by size:", {k: round(v, 3) for k, v in size_ap.items()})


if __name__ == "__main__":
    main()
