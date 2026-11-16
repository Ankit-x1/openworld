"""Category protocol: 15 base + 5 novel COCO classes.

Base and novel names are drawn from the published OV-COCO seen/unseen splits
(OVR-CNN class lists). This is a custom 20-class subset, not the full
48-base / 17-novel benchmark, so results are not directly comparable to
full OV-COCO numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

BASE_CLASSES = [
    "person",
    "car",
    "chair",
    "bottle",
    "bicycle",
    "motorcycle",
    "truck",
    "horse",
    "sheep",
    "bench",
    "laptop",
    "tv",
    "book",
    "backpack",
    "bird",
]

NOVEL_CLASSES = ["bus", "dog", "elephant", "umbrella", "keyboard"]


def load_name_to_id(annotations_path: str | Path) -> Dict[str, int]:
    """Map COCO category name -> original COCO category id."""
    with open(annotations_path) as f:
        data = json.load(f)
    return {cat["name"]: int(cat["id"]) for cat in data["categories"]}


def resolve_categories(
    annotations_path: str | Path,
    base_classes: List[str] | None = None,
    novel_classes: List[str] | None = None,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Resolve the 20 names to original COCO category ids and validate them."""
    base_classes = BASE_CLASSES if base_classes is None else base_classes
    novel_classes = NOVEL_CLASSES if novel_classes is None else novel_classes
    name_to_id = load_name_to_id(annotations_path)

    missing = [n for n in base_classes + novel_classes if n not in name_to_id]
    if missing:
        raise ValueError(f"Categories not found in COCO annotations: {missing}")

    if set(base_classes) & set(novel_classes):
        raise ValueError("Base and novel classes must be disjoint")

    base_ids = {n: name_to_id[n] for n in base_classes}
    novel_ids = {n: name_to_id[n] for n in novel_classes}

    if set(base_ids.values()) & set(novel_ids.values()):
        raise ValueError("Base and novel category ids overlap")

    return base_ids, novel_ids, name_to_id


def label_map_for_base(
    annotations_path: str | Path,
) -> Tuple[Dict[int, int], Dict[str, int]]:
    """original COCO id -> 1..15 for the base classes, plus name -> original id."""
    base_ids, _, _ = resolve_categories(annotations_path, novel_classes=[])
    label_map = {orig: i + 1 for i, orig in enumerate(sorted(base_ids.values()))}
    return label_map, base_ids
