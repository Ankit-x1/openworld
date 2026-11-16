"""Box geometry tested against torchvision reference implementations."""
import torch
from torchvision.ops import box_iou as tv_box_iou
from torchvision.ops import nms as tv_nms
from torchvision.models.detection._utils import BoxCoder

from src.openworld.boxes import (
    box_iou,
    decode,
    encode,
    nms,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_iou_matches_torchvision():
    torch.manual_seed(0)
    b1 = torch.rand(20, 4) * 100
    b1[:, 2:] = b1[:, :2] + torch.rand(20, 2) * 50
    b2 = torch.rand(15, 4) * 100
    b2[:, 2:] = b2[:, :2] + torch.rand(15, 2) * 50
    a = box_iou(b1, b2)
    b = tv_box_iou(b1, b2)
    assert torch.allclose(a, b, atol=1e-5)

    assert box_iou(torch.zeros((0, 4)), b2).shape == (0, 15)


def test_nms_matches_torchvision():
    torch.manual_seed(1)
    boxes = torch.rand(30, 4) * 100
    boxes[:, 2:] = boxes[:, :2] + torch.rand(30, 2) * 30 + 5
    scores = torch.rand(30)
    a = nms(boxes, scores, 0.5)
    b = tv_nms(boxes, scores, 0.5)
    assert torch.equal(a, b)


def test_xyxy_xywh_roundtrip():
    boxes = torch.tensor([[10.0, 20.0, 110.0, 220.0], [0.0, 0.0, 5.0, 5.0]])
    assert torch.allclose(xywh_to_xyxy(xyxy_to_xywh(boxes)), boxes)


def test_encode_decode_roundtrip():
    anchors = torch.tensor([[0.0, 0.0, 100.0, 100.0],
                            [10.0, 10.0, 90.0, 110.0]])
    gt = torch.tensor([[20.0, 30.0, 80.0, 90.0],
                       [15.0, 15.0, 85.0, 105.0]])
    codes = encode(anchors, gt)
    coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
    expected_codes = coder.encode([gt], [anchors])[0]
    assert torch.allclose(codes, expected_codes, atol=1e-6)

    recovered = decode(codes, anchors)
    expected = coder.decode(codes, [anchors])
    assert recovered.shape == (len(anchors), 1, 4)
    assert torch.allclose(recovered, expected, atol=1e-4)
    assert torch.allclose(recovered[:, 0], gt, atol=1e-4)

    class_codes = torch.cat([codes, codes], dim=1)
    class_decoded = decode(class_codes, anchors)
    class_expected = coder.decode(class_codes, [anchors])
    assert torch.allclose(class_decoded, class_expected, atol=1e-4)
