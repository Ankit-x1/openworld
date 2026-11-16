"""Frozen OpenCLIP region recognition.

Encodes image regions and compares them with class text embeddings.
Optionally fuses CLIP probabilities with RPN objectness scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PAD_COLOR = tuple(int(c * 255) for c in CLIP_MEAN)


def square_context_crop(
    image: Image.Image, box, expansion: float = 0.2, size: int = 224
) -> Image.Image:
    """Crop a square around a box with context expansion, preserving aspect."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * (1.0 + expansion)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1, ny1 = cx - side / 2.0, cy - side / 2.0
    nx2, ny2 = cx + side / 2.0, cy + side / 2.0

    crop = image.crop(
        (
            max(0, int(nx1)),
            max(0, int(ny1)),
            min(image.width, int(nx2)),
            min(image.height, int(ny2)),
        )
    )
    pad = Image.new("RGB", (int(round(side)), int(round(side))), PAD_COLOR)
    pad.paste(crop, (int(round(max(0, -nx1))), int(round(max(0, -ny1)))))
    return ImageOps.pad(pad, (size, size), method=Image.BICUBIC, color=PAD_COLOR)


def preprocess_region(image: Image.Image, size: int = 224) -> torch.Tensor:
    image = ImageOps.pad(image, (size, size), method=Image.BICUBIC, color=PAD_COLOR)
    t = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (t - mean) / std


class CLIPRecognizer:
    def __init__(
        self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"
    ):
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device

    def encode_text(self, class_names: Sequence[str], template: str = "a photo of a {}."):
        texts = self.tokenizer([template.format(n) for n in class_names]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(texts)
        return F.normalize(emb, dim=-1)

    def encode_regions(self, pil_images: List[Image.Image], batch_size: int = 64):
        tensors = [preprocess_region(im) for im in pil_images]
        batches = [
            torch.stack(tensors[i : i + batch_size])
            for i in range(0, len(tensors), batch_size)
        ]
        embeddings = []
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(self.device)
                emb = self.model.encode_image(batch)
                embeddings.append(F.normalize(emb, dim=-1).cpu())
        return torch.cat(embeddings, dim=0)

    def similarity(self, region_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        return region_emb @ text_emb.T

    def class_probs(self, sims: torch.Tensor) -> torch.Tensor:
        logit_scale = self.model.logit_scale.exp().to(sims.device)
        logits = sims * logit_scale
        return F.softmax(logits, dim=1)


def fused_scores(
    objectness: torch.Tensor,
    probs: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """S_ij = o_i^alpha * p_ij^(1-alpha). [R, C]"""
    o = objectness.clamp(min=1e-6).unsqueeze(1)
    p = probs.clamp(min=1e-6)
    return o.pow(alpha) * p.pow(1.0 - alpha)
