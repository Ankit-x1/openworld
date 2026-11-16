"""Validate the category protocol: 20 names resolve to real COCO ids."""
import os
import pytest

from src.openworld.categories import (
    BASE_CLASSES,
    NOVEL_CLASSES,
    resolve_categories,
)

TRAIN_ANN = "data/annotations/instances_train2017.json"


@pytest.mark.skipif(not os.path.exists(TRAIN_ANN), reason="training JSON not present yet")
def test_resolve_20_categories():
    base_ids, novel_ids, name_to_id = resolve_categories(TRAIN_ANN)

    assert set(base_ids.keys()) == set(BASE_CLASSES)
    assert set(novel_ids.keys()) == set(NOVEL_CLASSES)
    assert len(base_ids) == 15
    assert len(novel_ids) == 5

    assert not (set(base_ids.values()) & set(novel_ids.values()))
    assert all(1 <= v <= 90 for v in base_ids.values())
    assert all(1 <= v <= 90 for v in novel_ids.values())


@pytest.mark.skipif(not os.path.exists(TRAIN_ANN), reason="training JSON not present yet")
def test_label_mapping_range():
    from src.openworld.categories import label_map_for_base
    label_map, _ = label_map_for_base(TRAIN_ANN)
    assert set(label_map.values()) == set(range(1, 16))
