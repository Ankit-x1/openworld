#!/usr/bin/env python3
"""Phase 0 data preparation.

Reads the two COCO annotation JSONs, validates the 15 base + 5 novel category
protocol, and writes:
  data/annotations/*.stripped.json   slim COCO-format JSONs for the subset
  data/splits/*.json                 fixed train / val-tune / val-test image-ID manifests
  data/metadata/*.json               counts, co-occurrence, box sizes

Run once, then commit the manifests. Images are fetched separately with
scripts/fetch_images.py.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from pycocotools.coco import COCO

from src.openworld.categories import (
    BASE_CLASSES,
    NOVEL_CLASSES,
    resolve_categories,
)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def size_bucket(w, h):
    area = w * h
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def strip_annotations(coco, image_ids, keep_cat_ids, category_list):
    """Write a slim COCO JSON: selected images + selected annotations only."""
    images, annotations = [], []
    next_ann_id = 0
    for iid in image_ids:
        img = coco.loadImgs(int(iid))[0]
        images.append(
            {
                "id": int(iid),
                "file_name": img["file_name"],
                "width": int(img["width"]),
                "height": int(img["height"]),
                "coco_url": img.get("coco_url", ""),
            }
        )
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=int(iid), catIds=keep_cat_ids)):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": int(iid),
                    "category_id": int(ann["category_id"]),
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
            next_ann_id += 1
    return {"images": images, "annotations": annotations, "categories": category_list}


def build_categories_list(name_to_id, selected_names):
    return [
        {"id": int(name_to_id[name]), "name": name} for name in selected_names
    ]


def count_instances(coco, image_ids, cat_ids):
    inst = defaultdict(int)
    imgs = defaultdict(int)
    for iid in image_ids:
        seen = set()
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=int(iid), catIds=cat_ids)):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            cid = int(ann["category_id"])
            inst[cid] += 1
            seen.add(cid)
        for cid in seen:
            imgs[cid] += 1
    return inst, imgs


def main():
    ap = argparse.ArgumentParser(description="Prepare COCO manifests for openworld")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--train-annotations", default=None)
    ap.add_argument("--val-annotations", default=None)
    ap.add_argument("--max-train-images", type=int, default=None)
    ap.add_argument("--max-val-images", type=int, default=None)
    ap.add_argument("--output-dir", default="data")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    train_ann = args.train_annotations or cfg["data"]["train_annotations"]
    val_ann = args.val_annotations or cfg["data"]["val_annotations"]
    out = Path(args.output_dir)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(parents=True, exist_ok=True)

    # 1. Validate the 20 category names against the full training JSON
    base_ids, novel_ids, name_to_id = resolve_categories(train_ann)
    all_ids = {**base_ids, **novel_ids}
    label_map = {orig: i + 1 for i, orig in enumerate(sorted(all_ids.values()))}
    base_cat_ids = sorted(base_ids.values())
    all_cat_ids = sorted(set(all_ids.values()))

    # 2. Training images: contain at least one non-crowd base instance
    train_coco = COCO(train_ann)
    ann_ids = train_coco.getAnnIds(catIds=base_cat_ids)
    base_anns = train_coco.loadAnns(ann_ids)
    pos_image_ids = sorted(
        {int(a["image_id"]) for a in base_anns if int(a.get("iscrowd", 0)) == 0}
    )
    rng = random.Random(cfg["training"]["seed"])
    rng.shuffle(pos_image_ids)
    max_train = args.max_train_images or cfg["training"]["max_train_images"]
    train_ids = sorted(pos_image_ids[:max_train])

    # 3. Stripped training JSON: base annotations only, 15 base categories
    base_cat_list = build_categories_list(name_to_id, BASE_CLASSES)
    stripped_train = strip_annotations(train_coco, train_ids, base_cat_ids, base_cat_list)
    with open(out / "annotations" / "instances_train2017.stripped.json", "w") as f:
        json.dump(stripped_train, f, indent=2)

    # 4. Validation: all val images with at least one annotation of any of 20 classes
    val_coco = COCO(val_ann)
    val_ann_ids = val_coco.getAnnIds(catIds=all_cat_ids)
-    val_ids = sorted({int(a["image_id"]) for a in val_ann_ids})
    max_val = args.max_val_images or cfg["validation"]["max_val_images"]
    if len(val_ids) > max_val:
        rng.shuffle(val_ids)
        val_ids = sorted(val_ids[:max_val])

    tune_fraction = cfg["validation"]["tune_fraction"]
    n_tune = max(1, int(len(val_ids) * tune_fraction))
    rng2 = random.Random(cfg["validation"]["seed"])
    rng2.shuffle(val_ids)
    val_tune_ids = sorted(val_ids[:n_tune])
    val_test_ids = sorted(val_ids[n_tune:])

    # 5. Stripped validation JSONs: all 20 classes' annotations
    val_cat_list = build_categories_list(name_to_id, BASE_CLASSES + NOVEL_CLASSES)
    all_val_ids = val_tune_ids + val_test_ids
    all_val_anns = strip_annotations(val_coco, all_val_ids, all_cat_ids, val_cat_list)

    tune_imgs_set = set(val_tune_ids)
    test_imgs_set = set(val_test_ids)
    tune_anns = [a for a in all_val_anns["annotations"] if a["image_id"] in tune_imgs_set]
    test_anns = [a for a in all_val_anns["annotations"] if a["image_id"] in test_imgs_set]

    tune_images = [i for i in all_val_anns["images"] if i["id"] in tune_imgs_set]
    test_images = [i for i in all_val_anns["images"] if i["id"] in test_imgs_set]

    for name, imgs, anns in (
        ("instances_val2017.stripped.json", all_val_anns["images"], all_val_anns["annotations"]),
        ("instances_val_tune2017.stripped.json", tune_images, tune_anns),
        ("instances_val_test2017.stripped.json", test_images, test_anns),
    ):
        with open(out / "annotations" / name, "w") as f:
            json.dump({"images": imgs, "annotations": anns, "categories": val_cat_list}, f, indent=2)

    # 6. Manifests
    with open(out / "splits" / "train_image_ids.json", "w") as f:
        json.dump(train_ids, f, indent=2)
    with open(out / "splits" / "val_tune_image_ids.json", "w") as f:
        json.dump(val_tune_ids, f, indent=2)
    with open(out / "splits" / "val_test_image_ids.json", "w") as f:
        json.dump(val_test_ids, f, indent=2)

    with open(out / "splits" / "selected_categories.json", "w") as f:
        json.dump(
            {
                "base": [{"name": n, "id": int(base_ids[n])} for n in BASE_CLASSES],
                "novel": [{"name": n, "id": int(novel_ids[n])} for n in NOVEL_CLASSES],
            },
            f,
            indent=2,
        )

    # 7. Metadata / audit
    name_to_orig = {n: int(cid) for n, cid in all_ids.items()}
    train_inst, train_imgs = count_instances(train_coco, train_ids, base_cat_ids)
    tune_inst, tune_imgs = count_instances(val_coco, val_tune_ids, all_cat_ids)
    test_inst, test_imgs = count_instances(val_coco, val_test_ids, all_cat_ids)

    category_counts = {}
    for name in BASE_CLASSES + NOVEL_CLASSES:
        cid = name_to_orig[name]
        category_counts[name] = {
            "category_id": cid,
            "split": "base" if name in BASE_CLASSES else "novel",
            "train_instances": int(train_inst.get(cid, 0)),
            "train_images": int(train_imgs.get(cid, 0)),
            "val_tune_instances": int(tune_inst.get(cid, 0)),
            "val_tune_images": int(tune_imgs.get(cid, 0)),
            "val_test_instances": int(test_inst.get(cid, 0)),
            "val_test_images": int(test_imgs.get(cid, 0)),
        }
    with open(out / "metadata" / "category_counts.json", "w") as f:
        json.dump(category_counts, f, indent=2)

    co_occurrence = defaultdict(int)
    for iid in train_ids:
        anns = train_coco.loadAnns(train_coco.getAnnIds(imgIds=int(iid)))
        has_base = any(
            int(a["category_id"]) in base_ids.values()
            and int(a.get("iscrowd", 0)) == 0
            for a in anns
        )
        has_novel = any(
            int(a["category_id"]) in novel_ids.values()
            and int(a.get("iscrowd", 0)) == 0
            for a in anns
        )
        if has_base and has_novel:
            co_occurrence["images_with_base_and_novel"] += 1

    size_counts = {"small": 0, "medium": 0, "large": 0}
    for ann in stripped_train["annotations"]:
        size_counts[size_bucket(ann["bbox"][2], ann["bbox"][3])] += 1

    missing_images = []
    images_root = Path(args.output_dir) / "images"
    for info in stripped_train["images"]:
        if not (images_root / "train2017" / info["file_name"]).exists():
            missing_images.append(info["file_name"])

    split_metadata = {
        "seed": cfg["training"]["seed"],
        "num_train_images": len(train_ids),
        "num_val_tune_images": len(val_tune_ids),
        "num_val_test_images": len(val_test_ids),
        "val_tune_fraction": tune_fraction,
        "base_classes": BASE_CLASSES,
        "novel_classes": NOVEL_CLASSES,
        "note": "Base-only supervision: train images keep base annotations only; "
        "novel annotations in those images are dropped as supervision.",
    }
    with open(out / "splits" / "split_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)

    with open(out / "metadata" / "image_counts.json", "w") as f:
        json.dump(
            {
                "train": len(train_ids),
                "val_tune": len(val_tune_ids),
                "val_test": len(val_test_ids),
            },
            f,
            indent=2,
        )
    with open(out / "metadata" / "co_occurrence.json", "w") as f:
        json.dump(dict(co_occurrence), f, indent=2)
    with open(out / "metadata" / "download_manifest.json", "w") as f:
        json.dump(
            {
                "train_missing_images_sample": missing_images[:20],
                "train_missing_images_total": len(missing_images),
                "note": "Fetch images with scripts/fetch_images.py before training.",
            },
            f,
            indent=2,
        )

    print("===== Phase 0 complete =====")
    print(f"train images : {len(train_ids)}")
    print(f"val tune     : {len(val_tune_ids)}")
    print(f"val test     : {len(val_test_ids)}")
    print(f"categories   : {len(BASE_CLASSES)} base + {len(NOVEL_CLASSES)} novel")
    print(f"co-occurrence base+novel train images: {co_occurrence.get('images_with_base_and_novel', 0)}")
    print(f"train box sizes: {size_counts}")
    print(f"train images missing locally: {len(missing_images)}")


if __name__ == "__main__":
    main()
