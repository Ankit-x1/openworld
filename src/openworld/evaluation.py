"""Open-world evaluation: per-class AP50, bootstrap CIs, confusion matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision.ops import box_iou

RECALL_THRESHOLDS = np.arange(0, 1.01, 0.01)


def load_predictions(path: str | Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def group_by_image(
    predictions: Sequence[Dict],
) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = {}
    for p in predictions:
        grouped.setdefault(int(p["image_id"]), []).append(p)
    return grouped


def build_gt(coco: COCO, image_ids: Sequence[int]) -> Dict[int, List[Dict]]:
    gt: Dict[int, List[Dict]] = {}
    for iid in image_ids:
        ann_ids = coco.getAnnIds(imgIds=int(iid))
        anns = coco.loadAnns(ann_ids)
        boxes = []
        for ann in anns:
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            boxes.append(
                {
                    "bbox": [x, y, x + w, y + h],
                    "category_id": int(ann["category_id"]),
                }
            )
        gt[int(iid)] = boxes
    return gt


def per_class_ap(
    pred_by_image: Dict[int, List[Dict]],
    gt_by_image: Dict[int, List[Dict]],
    cat_id: int,
    iou_threshold: float = 0.5,
) -> float:
    """Average precision for one class (interpolated AP over 101 recall points)."""
    preds: List[Tuple[float, int, List[float]]] = []
    gts: Dict[int, List[List[float]]] = {}
    for iid, plist in pred_by_image.items():
        for p in plist:
            if int(p["category_id"]) == cat_id:
                bbox = [float(v) for v in p["bbox"]]
                preds.append((float(p["score"]), iid, bbox))
    gts = {
        iid: [
            [float(v) for v in g["bbox"]]
            for g in boxes
            if int(g["category_id"]) == cat_id
        ]
        for iid, boxes in gt_by_image.items()
    }

    if not preds:
        return 0.0
    preds.sort(key=lambda t: t[0], reverse=True)

    matched_gt: Dict[int, set] = {iid: set() for iid in gts}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, (score, iid, bbox) in enumerate(preds):
        best_j, best_iou = -1, iou_threshold
        gt_list = gts.get(iid, [])
        for j, g in enumerate(gt_list):
            if j in matched_gt.get(iid, set()):
                continue
            iou = box_iou(
                torch.tensor([bbox]), torch.tensor([g])
            ).item()
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            tp[i] = 1
            matched_gt[iid].add(best_j)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    total_gt = sum(len(g) for g in gts.values())
    if total_gt == 0:
        return 0.0

    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    ap = 0.0
    for r in RECALL_THRESHOLDS:
        candidates = precision[recall >= r]
        ap += float(candidates.max()) if len(candidates) else 0.0
    return ap / len(RECALL_THRESHOLDS)


def evaluate_predictions(
    predictions: List[Dict],
    coco: COCO,
    image_ids: Sequence[int],
    base_names: Sequence[str],
    novel_names: Sequence[str],
    name_to_orig_id: Dict[str, int],
    iou_threshold: float = 0.5,
) -> Dict:
    pred_by_image = group_by_image(predictions)
    gt_by_image = build_gt(coco, image_ids)

    all_names = list(base_names) + list(novel_names)
    all_cat_ids = [name_to_orig_id[n] for n in all_names]

    per_class: Dict[str, float] = {}
    for name in all_names:
        per_class[name] = per_class_ap(
            pred_by_image, gt_by_image, name_to_orig_id[name], iou_threshold
        )

    base_ap = float(np.mean([per_class[n] for n in base_names]))
    novel_ap = float(np.mean([per_class[n] for n in novel_names]))
    generalized_ap = float(np.mean([per_class[n] for n in all_names]))

    return {
        "iou_threshold": iou_threshold,
        "base_AP50": base_ap,
        "novel_AP50": novel_ap,
        "generalized_AP50": generalized_ap,
        "per_class_AP50": per_class,
        "num_images": len(image_ids),
        "num_predictions": len(predictions),
    }


def bootstrap_ci(
    metric_fn,
    image_ids: Sequence[int],
    n_rounds: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    ids = list(image_ids)
    values = []
    for _ in range(n_rounds):
        sample = rng.choice(ids, size=len(ids), replace=True).tolist()
        values.append(metric_fn(sample))
    values = np.array(values)
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(
        np.percentile(values, 97.5)
    )


def confusion_matrix(
    predictions: List[Dict],
    gt_by_image: Dict[int, List[Dict]],
    class_names: Sequence[str],
    id_to_name: Dict[int, str],
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Rows = GT class, cols = predicted class. Last col = rejected (no match)."""
    n = len(class_names)
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    mat = np.zeros((n, n + 1), dtype=int)

    preds_by_image: Dict[int, List[Dict]] = {}
    for p in predictions:
        preds_by_image.setdefault(int(p["image_id"]), []).append(p)

    for iid, gts in gt_by_image.items():
        preds = preds_by_image.get(iid, [])
        for g in gts:
            best_j, best_iou = -1, iou_threshold
            for j, p in enumerate(preds):
                iou = box_iou(
                    torch.tensor([g["bbox"]]), torch.tensor([p["bbox"]])
                ).item()
                if iou > best_iou:
                    best_iou, best_j = iou, j
            gt_name = id_to_name.get(int(g["category_id"]))
            row = name_to_idx.get(gt_name)
            if row is None:
                continue
            if best_j < 0:
                mat[row, n] += 1
            else:
                pred_name = id_to_name.get(int(preds[best_j]["category_id"]))
                col = name_to_idx.get(pred_name, n)
                mat[row, col] += 1
    return mat
