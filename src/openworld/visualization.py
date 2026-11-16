"""Draw ground-truth and predicted boxes; save side-by-side comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GT_COLOR = (0, 200, 0)
PRED_COLOR = (220, 30, 30)


def _load_font(size: int = 16):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(
    image: Image.Image,
    boxes: Sequence,
    labels: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    colors: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = _load_font()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        color = colors[i] if colors else GT_COLOR
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(labels[i])
        if scores is not None:
            label += f" {float(scores[i]):.2f}"
        tw = draw.textlength(label, font=font)
        th = 14
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
        draw.text((x1, y1 - th), label, fill=(255, 255, 255), font=font)
    return image


def side_by_side(
    gt_image: Image.Image,
    pred_image: Image.Image,
    out_path: str | Path,
    titles: Tuple[str, str] = ("Ground truth", "Prediction"),
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w1, h1 = gt_image.size
    w2, h2 = pred_image.size
    w = w1 + w2 + 40
    h = max(h1, h2) + 40
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(gt_image, (20, 20))
    canvas.paste(pred_image, (w1 + 40, 20))
    canvas.save(out_path)
    return out_path
