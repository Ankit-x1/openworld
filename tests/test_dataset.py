"""Smoke test for COCODataset (skips when images are not downloaded yet)."""
import os
import json

import pytest
import torch

from src.openworld.categories import label_map_for_base
from src.openworld.dataset import COCODataset, collate_fn

TRAIN_ANN = "data/annotations/instances_train2017.stripped.json"
TRAIN_IMAGES = "data/images/train2017"
TRAIN_IDS = "data/splits/train_image_ids.json"


@pytest.mark.skipif(
    not (os.path.exists(TRAIN_ANN) and os.path.isdir(TRAIN_IMAGES)),
    reason="stripped annotations and images not prepared yet",
)
def test_dataset_item():
    label_map, _ = label_map_for_base(TRAIN_ANN)
    all_ids = json.load(open(TRAIN_IDS))
    ds = COCODataset(TRAIN_ANN, TRAIN_IMAGES, all_ids[:2], label_map)
    image, target = ds[0]
    assert image.mode == "RGB"
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64
    assert target["labels"].numel() == 0 or (
        target["labels"].min() >= 1 and target["labels"].max() <= 15
    )
    batch = collate_fn([ds[0], ds[1]])
    assert len(batch) == 2
