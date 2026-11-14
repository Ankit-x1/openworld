
# openworld — open-vocabulary object detection

**What this project does.** It trains a Faster R-CNN detector to find objects (localization) in 15 base COCO categories, then uses a frozen OpenCLIP model to name objects in the regions it finds — including 5 novel categories the detector was never trained to classify (bus, dog, elephant, umbrella, keyboard).

**Why.** A normal detector can only name categories it saw during training. This project splits the problem into two parts:

1. **Localization:** Faster R-CNN finds where objects are, using only the 15 base categories for training.
2. **Recognition:** OpenCLIP looks at each found region and matches it against text descriptions of all 20 categories (base + novel).

The central question: how well can a detector trained on a limited set of base categories localize novel objects, and how much does CLIP-based region recognition improve their classification?

**Locked decisions:**

- Dataset: COCO 2017
- Categories: 15 base + 5 novel (custom OV-COCO subset)
- Detector: Faster R-CNN, ResNet-50 + FPN, ImageNet backbone only (no COCO detector pretraining)
- Recognition: frozen OpenCLIP ViT-B/32
- Runtime: Windows CPU (no GPU required)
