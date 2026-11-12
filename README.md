# openworld

Find objects using a detector trained on 15 categories, then ask frozen CLIP to name the regions using a 20-category vocabulary. The useful result is not just one accuracy number: it shows whether unseen objects were missed because it is not trained yet.

**Start with [the step-by-step local and Colab guide](docs/RUNBOOK.md).** The single notebook is a runner, not a second copy of the code. No Git commands, repository initialization, commits, image downloads, or model training were performed to prepare this package.

## What is included

There are five scripts: `prepare_data.py`, `fetch_images.py`, `train.py`, `extract_proposals.py`, and `evaluate.py`. Reusable code lives in `src/openworld/`. Settings live in `config.yaml`.

The pipeline is:

```text
Full COCO JSONs on your laptop → small fixed annotation files
                                          ↓
                              image downloads on Colab
                                          ↓
                     base-trained Faster R-CNN → proposed boxes
                                          ↓
                     frozen CLIP → 20-category recognition
```

Base categories: person, car, chair, bottle, bicycle, motorcycle, truck, horse, sheep, bench, laptop, tv, book, backpack, bird.

Unseen detector categories: bus, dog, elephant, umbrella, keyboard.

Training keeps images with at least one valid, non-crowd base object. It removes all non-base box labels. Unseen objects can still appear unlabeled in those images. The detector starts with an ImageNet backbone, not a COCO-pretrained detector. CLIP remains frozen. “Unseen” means unseen in the detector's box-label training, not unseen by CLIP or ImageNet pretraining.

All 5,000 validation images are kept, including images with no selected objects. A fixed seeded split assigns 25% to tuning and 75% to final testing. This is a custom 15/5 experiment, not the full published open-vocabulary COCO benchmark. The hardcoded class selection is not an automated verification of the published split.

## The three measurements

`evaluate.py detector` measures the original detector's base-class detections. `evaluate.py proposals` measures how often its proposed boxes cover base and unseen objects, at 100, 300, and 1,000 proposals. `evaluate.py gt` tests CLIP on known annotation boxes, separating recognition from localization. `evaluate.py open-vocab` tests the actual combined system.

Detection scores use one best CLIP label per region. They combine objectness and vocabulary-relative CLIP scores, apply a cosine rejection rule and remove repeated same-class boxes. These scores are rankings, not calibrated probabilities. Prompts, crop context and score choices are tuned only on `val_tune`; final CLIP commands require a saved settings file.

Mean accuracy across categories and box-overlap thresholds is written as `AP50_95`. `AP50` uses a box-overlap threshold of 0.5. A value of `0.25` means 25%, not 0.25%. `null` means that a class or size group has no evaluable objects. Known-box recognition is a diagnostic, not a guaranteed upper bound on detector AP.

## Scope deliberately kept small

No web demo, classical edge-detection exercise, anchor reimplementation, second detector, or distillation system. Those would add work before the central experiment is answered. The included optional comparison excludes training images containing a selected unseen category; it requires a separate preparation folder and a new detector run.

No trained checkpoints, real dataset files, prediction scores or performance claims are included. Unit tests use tiny synthetic fixtures, solely to check correctness. See `docs/VALIDATION.md` for what was actually tested and `docs/sources.md` for library references.
