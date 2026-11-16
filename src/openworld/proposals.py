"""RPN proposal extraction and proposal recall evaluation."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou, nms


def _cat(tensors, dim: int = 0) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


def extract_proposals(
    model: torch.nn.Module,
    images: List,
    targets: List | None = None,
    pre_nms_top_n: int = 2000,
    post_nms_top_n: int = 1000,
    nms_thresh: float = 0.7,
    score_thresh: float = 0.0,
    min_size: float = 1.0,
) -> List[Dict]:
    """Return RPN proposals in original image coordinates, with scores."""
    from torchvision.models.detection.rpn import concat_box_prediction_layers
    model.eval()
    original_sizes = [tuple(im.shape[-2:]) for im in images]
    with torch.no_grad():
        image_list, _ = model.transform(images, None)
        features = model.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        levels = list(features.values())
        logits, deltas = model.rpn.head(levels)
        anchors = model.rpn.anchor_generator(image_list, levels)
        counts = [x.shape[1] * x.shape[2] * x.shape[3] for x in logits]
        logits, deltas = concat_box_prediction_layers(logits, deltas)
        decoded = model.rpn.box_coder.decode(deltas.detach(), anchors).view(len(images), -1, 4)
        # Reuse torchvision's per-level top-k, clipping and NMS implementation.
        rpn = model.rpn
        old_pre, old_post, old_nms, old_score, old_min = (rpn._pre_nms_top_n['testing'],
            rpn._post_nms_top_n['testing'], rpn.nms_thresh, rpn.score_thresh, rpn.min_size)
        try:
            rpn._pre_nms_top_n['testing'] = pre_nms_top_n
            rpn._post_nms_top_n['testing'] = post_nms_top_n
            rpn.nms_thresh, rpn.score_thresh, rpn.min_size = nms_thresh, score_thresh, min_size
            boxes, scores = rpn.filter_proposals(decoded, logits, image_list.image_sizes, counts)
        finally:
            rpn._pre_nms_top_n['testing'], rpn._post_nms_top_n['testing'] = old_pre, old_post
            rpn.nms_thresh, rpn.score_thresh, rpn.min_size = old_nms, old_score, old_min
        # transform.postprocess rescales coordinates back to the input image.
        results = model.transform.postprocess(
            [{'boxes': b, 'scores': s} for b, s in zip(boxes, scores)],
            image_list.image_sizes, original_sizes)
        return [{'boxes': r['boxes'].cpu(), 'scores': r['scores'].cpu()} for r in results]


def clip_boxes_inline(boxes: torch.Tensor, size) -> torch.Tensor:
    h, w = int(size[0]), int(size[1])
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h)
    return boxes


def save_proposals(
    results: List[Dict], image_ids: List[int], out_dir: str | Path
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for iid, res in zip(image_ids, results):
        p = out_dir / f"{int(iid):012d}.npz"
        np.savez(p, boxes=res["boxes"].numpy(), scores=res["scores"].numpy())
        paths.append(p)
    return paths


def load_proposals(image_id: int, out_dir: str | Path) -> Dict:
    p = Path(out_dir) / f"{int(image_id):012d}.npz"
    data = np.load(p)
    return {"boxes": data["boxes"], "scores": data["scores"]}


def recall_at(
    gt_boxes: torch.Tensor, proposal_boxes: torch.Tensor, iou_threshold: float = 0.5
) -> float:
    """Fraction of GT boxes with max IoU >= threshold against proposals."""
    if gt_boxes.shape[0] == 0 or proposal_boxes.shape[0] == 0:
        return 0.0
    ious = box_iou(gt_boxes, proposal_boxes)
    max_iou = ious.max(dim=1).values
    return float((max_iou >= iou_threshold).float().mean().item())


def split_gt_by_class(
    coco, image_ids: List[int], base_ids: Dict[str, int], novel_ids: Dict[str, int]
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]]:
    """image_id -> list of GT xyxy boxes for base and novel classes."""
    base_gt: Dict[int, List[torch.Tensor]] = {}
    novel_gt: Dict[int, List[torch.Tensor]] = {}
    base_id_set = set(base_ids.values())
    novel_id_set = set(novel_ids.values())

    for iid in image_ids:
        ann_ids = coco.getAnnIds(imgIds=int(iid))
        anns = coco.loadAnns(ann_ids)
        b_boxes, n_boxes = [], []
        for ann in anns:
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            cid = int(ann["category_id"])
            box = torch.tensor([x, y, x + w, y + h])
            if cid in base_id_set:
                b_boxes.append(box)
            elif cid in novel_id_set:
                n_boxes.append(box)
        if b_boxes:
            base_gt[int(iid)] = b_boxes
        if n_boxes:
            novel_gt[int(iid)] = n_boxes
    return base_gt, novel_gt
