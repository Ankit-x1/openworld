"""Box geometry: IoU, NMS, encode/decode.

Conventions match torchvision so every routine can be tested against
torchvision.ops.box_iou, torchvision.ops.nms and torchvision BoxCoder.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torchvision.ops import box_iou as tv_box_iou
from torchvision.ops import nms as tv_nms


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x, y, x + w, y + h], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for boxes in xyxy format. Returns [N, M]."""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))

    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8)


def clip_boxes(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    h, w = int(size[0]), int(size[1])
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h)
    return boxes


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long)
    return tv_nms(boxes, scores, iou_threshold)


def encode(
    anchors: torch.Tensor, gt_boxes: torch.Tensor, weights=(10.0, 10.0, 5.0, 5.0)
) -> torch.Tensor:
    """Encode gt boxes relative to anchors (center form)."""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah

    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gcx = gt_boxes[:, 0] + 0.5 * gw
    gcy = gt_boxes[:, 1] + 0.5 * gh

    wx, wy, ww, wh = weights
    dx = (gcx - acx) / aw / wx
    dy = (gcy - acy) / ah / wy
    dw = torch.log(gw / aw) / ww
    dh = torch.log(gh / ah) / wh
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode(
    codes: torch.Tensor, anchors: torch.Tensor, weights=(10.0, 10.0, 5.0, 5.0)
) -> torch.Tensor:
    """Decode box-coder outputs back to xyxy boxes."""
    wx, wy, ww, wh = weights
    dx = codes[..., 0::4] * wx
    dy = codes[..., 1::4] * wy
    dw = codes[..., 2::4] * ww
    dh = codes[..., 3::4] * wh

    aw = anchors[..., 2] - anchors[..., 0]
    ah = anchors[..., 3] - anchors[..., 1]
    acx = anchors[..., 0] + 0.5 * aw
    acy = anchors[..., 1] + 0.5 * ah

    max_scale = math.log(1000.0 / 16.0)
    dw = dw.clamp(max=max_scale)
    dh = dh.clamp(max=max_scale)

    gw = torch.exp(dw) * aw
    gh = torch.exp(dh) * ah
    gcx = dx * aw + acx
    gcy = dy * ah + acy

    x1 = gcx - 0.5 * gw
    y1 = gcy - 0.5 * gh
    x2 = gcx + 0.5 * gw
    y2 = gcy + 0.5 * gh
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
