"""COCO dataset for the 15-class base detector.

Returns (image, target) with labels remapped to 1..15 (0 = background).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class RandomHorizontalFlip:
    """Horizontal flip that updates bounding boxes in place."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, target: Dict):
        if random.random() < self.p:
            w = image.width
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


class COCODataset(Dataset):
    def __init__(
        self,
        annotations_path: str | Path,
        images_root: str | Path,
        image_ids: Sequence[int],
        label_map: Dict[int, int],  # original COCO category id -> 1..15
        transforms: Callable | None = None,
    ):
        self.coco = COCO(str(annotations_path))
        self.images_root = Path(images_root)
        self.image_ids = [int(i) for i in image_ids]
        self.label_map = dict(label_map)
        self.cat_ids = sorted(self.label_map)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(image_id)[0]

        image = Image.open(self.images_root / img_info["file_name"]).convert("RGB")

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            labels.append(self.label_map[int(ann["category_id"])])
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(areas, dtype=torch.float32)
            if areas
            else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64)
            if iscrowd
            else torch.zeros((0,), dtype=torch.int64),
            "file_name": img_info["file_name"],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
