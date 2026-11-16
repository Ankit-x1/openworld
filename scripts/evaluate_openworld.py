#!/usr/bin/env python3
"""Final open-world evaluation (Phase 5).

Per-class AP50, base / novel / generalized AP50, bootstrap confidence intervals,
confusion matrix. Compares predictions against ground truth on fixed val-test IDs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.openworld.evaluation import (
    bootstrap_ci, build_gt, evaluate_predictions, group_by_image,
    per_class_ap, confusion_matrix,
)
from pycocotools.coco import COCO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--selected-categories", default="data/splits/selected_categories.json")
    ap.add_argument("--image-ids", default="data/splits/val_test_image_ids.json")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--bootstrap-rounds", type=int, default=200)
    ap.add_argument("--out-dir", default="outputs/metrics")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = json.load(open(args.selected_categories))
    base_names = [c["name"] for c in sel["base"]]
    novel_names = [c["name"] for c in sel["novel"]]
    name_to_orig = {c["name"]: int(c["id"]) for c in sel["base"] + sel["novel"]}
    all_names = base_names + novel_names

    predictions = json.load(open(args.predictions))
    coco = COCO(args.annotations)
    image_ids = json.load(open(args.image_ids))

    id_to_name = {int(c["id"]): c["name"] for c in coco.loadCats(coco.getCatIds())}

    metrics = evaluate_predictions(
        predictions, coco, image_ids, base_names, novel_names,
        name_to_orig, iou_threshold=args.iou_thr,
    )

    def metric_subset(ids_subset, mode):
        id_set = set(int(i) for i in ids_subset)
        filt = [p for p in predictions if int(p["image_id"]) in id_set]
        subset_ids = [int(i) for i in ids_subset]
        m = evaluate_predictions(
            filt, coco, subset_ids, base_names, novel_names, name_to_orig,
            iou_threshold=args.iou_thr,
        )
        return m[mode]

    def metric_novel_ap(ids_subset):
        return metric_subset(ids_subset, "novel_AP50")

    def metric_gen_ap(ids_subset):
        return metric_subset(ids_subset, "generalized_AP50")

    mean_novel, lo_novel, hi_novel = bootstrap_ci(
        metric_novel_ap, image_ids, n_rounds=args.bootstrap_rounds,
        seed=cfg_bootstrap_seed(),
    )
    mean_gen, lo_gen, hi_gen = bootstrap_ci(
        metric_gen_ap, image_ids, n_rounds=args.bootstrap_rounds,
        seed=cfg_bootstrap_seed(),
    )

    report = {
        "iou_threshold": args.iou_thr,
        "base_AP50": metrics["base_AP50"],
        "novel_AP50": metrics["novel_AP50"],
        "generalized_AP50": metrics["generalized_AP50"],
        "novel_AP50_bootstrap": {"mean": mean_novel, "lo": lo_novel, "hi": hi_novel},
        "generalized_AP50_bootstrap": {"mean": mean_gen, "lo": lo_gen, "hi": hi_gen},
        "per_class_AP50": metrics["per_class_AP50"],
        "num_images": metrics["num_images"],
        "num_predictions": metrics["num_predictions"],
    }
    with open(out_dir / "openworld_eval.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        f"IoU threshold : {args.iou_thr}",
        f"Images        : {metrics['num_images']}",
        f"Predictions   : {metrics['num_predictions']}",
        f"Base AP50     : {metrics['base_AP50']:.4f}",
        f"Novel AP50    : {metrics['novel_AP50']:.4f} "
        f"(bootstrap 95% CI [{lo_novel:.4f}, {hi_novel:.4f}])",
        f"Generalized   : {metrics['generalized_AP50']:.4f} "
        f"(bootstrap 95% CI [{lo_gen:.4f}, {hi_gen:.4f}])",
        "",
        "Per-class AP50:",
    ]
    for name in all_names:
        lines.append(f"  {name:12s} {metrics['per_class_AP50'][name]:.4f}")
    report_text = "\n".join(lines)
    print(report_text)
    with open(out_dir / "openworld_eval.txt", "w") as f:
        f.write(report_text)

    gt_by_image = build_gt(coco, image_ids)
    for p in predictions:
        p["name"] = id_to_name.get(int(p["category_id"]), str(p["category_id"]))
    for iid, gts in gt_by_image.items():
        for g in gts:
            g["name"] = id_to_name.get(int(g["category_id"]), str(g["category_id"]))

    mat = confusion_matrix(
        predictions, gt_by_image, all_names, id_to_name,
        iou_threshold=args.iou_thr,
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap="Blues")
    labels = list(all_names) + ["rejected"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(all_names)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(all_names, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Confusion matrix (IoU >= 0.5)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    np.save(out_dir / "confusion_matrix.npy", mat)
    print("Confusion matrix ->", out_dir / "confusion_matrix.png")


def cfg_bootstrap_seed():
    import yaml
    try:
        return yaml.safe_load(open("configs/config.yaml"))["eval"]["bootstrap_seed"]
    except Exception:
        return 42


if __name__ == "__main__":
    main()
