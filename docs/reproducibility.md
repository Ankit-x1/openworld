# Reproduction and artifact notes

This guide covers the recorded Windows/CPU experiment. This working copy has
local COCO images, prepared annotations, checkpoints, and prediction files;
they are ignored by Git and are not available to someone cloning the tracked
repository. The local result files were used to verify the README.

## Set up a new Windows environment

Use the repository root as the working directory. Install the CPU builds of
PyTorch and torchvision separately; they are not listed in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install jupyter ipykernel pytest
```

The versions used for the completed training run were not recorded. These
commands are a setup recipe, not a way to recreate the exact historical
environment.

## Prepare only the selected images

The scripts need the original COCO 2017 instance annotation JSONs at
`data/annotations/instances_train2017.json` and
`data/annotations/instances_val2017.json` to create the stripped annotations
and image-ID manifests. The image archives are not needed. Obtain the
annotation files from the COCO distribution, place them at those paths, then
prepare the subset:

```powershell
python scripts\prepare_data.py --config configs\config.yaml
```

This writes files under `data/annotations`, `data/splits`, and `data/metadata`.
It can overwrite the existing manifests. For the completed experiment, use
the recorded manifests; do not regenerate or change the test split after
looking at its results.

Fetch only the 2,000 selected training images and the 2,000 selected
validation images (500 tuning + 1,500 test):

```powershell
python scripts\fetch_images.py --subset data\annotations\instances_train2017.stripped.json --dest data\images\train2017 --jobs 16
python scripts\fetch_images.py --subset data\annotations\instances_val_tune2017.stripped.json --dest data\images\val2017 --jobs 16
python scripts\fetch_images.py --subset data\annotations\instances_val_test2017.stripped.json --dest data\images\val2017 --jobs 16
```

Check each command's `failed=0` summary. Existing verified files are skipped.
This downloads selected photos, not all of COCO.

## Experiment stages

The following commands intentionally run training or inference. They are
included for a deliberate new run, not needed to read the published result
summary.

Train the detector on the selected base-only annotations:

```powershell
python scripts\train_detector.py --mode train --config configs\config.yaml --images-root data\images\train2017 --epochs 3 --batch-size 2 --lr 0.005 --checkpoint-dir outputs\checkpoints\full --log-file outputs\metrics\full_train_log.csv
```

CPU training takes many hours. The detector is initialized with ImageNet
backbone weights, not pretrained COCO detection weights.
The checkpoint loader uses `torch.load(..., weights_only=False)` because these
checkpoints also contain optimizer and random-state data. Only load a
checkpoint you trust; Python pickle loading can execute code.

Extract test proposals and run CLIP on the top 100 proposals per image:

```powershell
python scripts\extract_proposals.py --config configs\config.yaml --checkpoint outputs\checkpoints\full\epoch_3.pt --image-ids data\splits\val_test_image_ids.json --annotations data\annotations\instances_val_test2017.stripped.json --images-root data\images\val2017 --out-dir outputs\proposals\val_test --top-n 1000
python scripts\run_clip.py --mode proposals --config configs\config.yaml --image-ids data\splits\val_test_image_ids.json --annotations data\annotations\instances_val_test2017.stripped.json --images-root data\images\val2017 --proposals-dir outputs\proposals\val_test --top-k 100 --alpha 0.5 --expansion 0.2 --out-dir outputs\clip\val_test_proposals
```

`run_clip.py` accepts a `--checkpoint` option but does not use it to create or
score proposals. The proposal files are produced by `extract_proposals.py`;
their checkpoint provenance is not embedded in the `.npz` files.

To evaluate an existing prediction file without rerunning the detector or
CLIP, the CLI evaluator is:

```powershell
python scripts\evaluate_openworld.py --predictions outputs\clip\val_test_proposals\predictions.json --annotations data\annotations\instances_val_test2017.stripped.json --selected-categories data\splits\selected_categories.json --image-ids data\splits\val_test_image_ids.json --out-dir outputs\metrics
```

## Notebook cautions

- Do not use **Run All**. Notebook 02 trains; notebook 03 extracts proposals;
  notebook 04 runs CLIP; notebook 05 contains proposal extraction, CLIP, and a
  separate detector-only inference cell.
- In notebook 03, the first recall@K loop loads all saved proposals for each
  value of K and does not slice to K. Its saved 100/300/1000 values therefore
  repeat the full-proposal result and should not be reported as separate
  recall@K measurements. A later plot cell does slice proposals to K.
- Notebook 04 contains old evaluation-debug cells. One cell attempts to patch
  `src/openworld/evaluation.py` in place and no longer finds its old source
  line; the active evaluator already contains the category-filtering fix. The
  proposal commands also pass `--checkpoint`, which `run_clip.py` ignores.
- In notebook 05, cells 2 and 3 regenerate test proposals and CLIP
  predictions; cell 9 reruns the detector-only baseline. Skip them when using
  the saved files. Cell 1 only reports missing files and does not stop
  execution. Cells 4-7 evaluate saved predictions; cell 8 recreates the
  confusion matrix and example gallery from local images.
- `scripts/run_pipeline.bat` is not a results-viewing shortcut. It prepares
  data and starts a smoke training run; later stages are commented out, and
  some commented examples contain old checkpoint paths.

## Result definitions and provenance

- **GT-box CLIP recognition** supplies ground-truth boxes and measures
  classification accuracy over the stored regions. It does not evaluate
  localization and is not end-to-end detector AP.
- **RPN + CLIP** evaluates the class-labeled RPN proposals using the
  repository's custom class-aware 101-point AP at IoU strictly greater than
  0.5. The final JSON records 1,500 images and 150,000 predictions.
- **Detector-only** evaluates the 15 base classes with `pycocotools.COCOeval`.
  Its COCO AP values do not form a controlled comparison with the custom
  RPN + CLIP evaluator.
- Alpha 0.5 was selected on the 500-image tuning split before the 1,500-image
  test evaluation. The predictions are a list of records and do not carry
  alpha, checkpoint hash, or proposal provenance as metadata. The final
  metrics JSON says alpha is assumed from the run that created the
  predictions. The saved command/output history supports the recorded setting,
  but the prediction file alone cannot prove it.
- The image-bootstrap intervals use 200 resamples and seed 42. They quantify
  resampling uncertainty on this test set, not variation across training runs.
- The saved confusion matrix has actual classes as rows and predicted classes
  plus a `missed` column. It does not include unmatched false positives as a
  separate row, so it is a partial object-match summary, not a full
  classification confusion matrix.
