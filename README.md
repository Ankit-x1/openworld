# Openworld: Open-Vocabulary Object Detection on CPU

This project explores whether a detector trained on a limited set of object classes can find objects from classes it was not trained to classify. It combines:

1. **Faster R-CNN with a ResNet-50 + FPN backbone** to find candidate object regions.
2. **Frozen OpenCLIP ViT-B/32** to name those regions using text descriptions of 20 classes.

The experiment was run locally on **Windows using CPU**, without a GPU or Colab. The dataset is a **custom 20-class subset** using class membership from the OV-COCO seen/unseen split. It is **not the full 48-base / 17-novel benchmark**, so these results must not be presented as directly comparable to full OV-COCO benchmark numbers.

> **Status:** Three detector-training epochs, proposal extraction, CLIP recognition, and final test evaluation were completed. The numbers below are from the recorded run. Some earlier notebook cells and scripts had bugs; the fixes and limitations are documented below.

---

## 1. The experiment in one picture

```text
COCO images and annotations
         |
         +--> 15 base classes for detector training
         |       (novel labels are not used as training supervision)
         |
         +--> Faster R-CNN + ResNet-50 + FPN
         |       trained for 3 epochs on 2,000 selected training images
         |
         +--> RPN proposes regions in an image
         |
         +--> frozen OpenCLIP compares each region with 20 class prompts
         |
         +--> objectness + CLIP scores produce ranked class predictions
         |
         +--> fixed validation-test evaluation
```

### Base classes used for detector supervision

`person`, `car`, `chair`, `bottle`, `bicycle`, `motorcycle`, `truck`, `horse`, `sheep`, `bench`, `laptop`, `tv`, `book`, `backpack`, and `bird`.

### Novel classes

`bus`, `dog`, `elephant`, `umbrella`, and `keyboard`.

The detector has **16 outputs**: 15 foreground base classes plus background. CLIP considers all **20 class names** when it labels a proposed region.

### What “base-only” means here

The training subset retains selected images even when they contain novel objects. The stripped training annotations contain **only base-class annotations**; novel annotations are removed from training supervision. This does not mean novel objects were removed from the image pixels. Consequently, some novel objects may be treated as unlabelled/background during detector training—an important limitation of this protocol.

---

## 2. Model and settings

| Component | Recorded setting |
|---|---|
| Runtime | Windows, CPU |
| Detector | Faster R-CNN, ResNet-50 + feature pyramid network |
| Detector initialization | ImageNet backbone weights; no pretrained object-detection weights |
| Detector classes | 15 base classes + background |
| Detector training | 2,000 training images, 3 epochs, batch size 2, learning rate 0.005 |
| Training optimizer | SGD; momentum and weight decay from `configs/config.yaml` |
| CLIP | Frozen OpenCLIP `ViT-B-32-quickgelu`, `openai` pretrained weights |
| Default text prompt | `a photo of a {}.` |
| Crop context expansion | `0.2` in the recorded CLIP runs |
| Test proposals | Up to 1,000 RPN proposals saved per image; top 100 passed to CLIP |
| Fusion setting | `alpha = 0.5`, selected on val-tune before final test evaluation |
| Test evaluator | Custom 101-point class-aware AP at IoU strictly greater than 0.5 |
| Confidence intervals | 200 image-level bootstrap resamples, seed 42 |

CLIP was **not fine-tuned**. The final test run used the full-training checkpoint at:

```text
outputs/checkpoints/full/epoch_3.pt
```

### Region scoring

For each proposed region, the recorded implementation combines RPN objectness `o` and a CLIP class probability `p` using:

```text
score = o ** alpha * p ** (1 - alpha)
```

The script assigns a class to a region and saves a score and box. At `alpha = 0`, the score uses CLIP probability only; at `alpha = 0.5`, it combines both. **At `alpha = 1`, the class scores are tied if the implementation selects a class by taking the argmax of this formula**, because the CLIP probability is raised to zero. Therefore, the recorded `alpha = 1.0` result should not be interpreted as a meaningful CLIP-classification baseline without checking the implementation and tie handling.

---

## 3. Data and fixed splits

The project uses COCO 2017 annotation files already supplied locally. It does **not** need the full COCO training-image archive.

### Recorded subset

| Split | Images | Use |
|---|---:|---|
| Train | 2,000 | Train the 15-class detector |
| Val-tune | 500 | Compare CLIP settings and select alpha |
| Val-test | 1,500 | Final held-out evaluation |

The selected validation subset was split 25% / 75% with the configured seed. The split lists are saved in `data/splits/` and should stay fixed for this experiment.

**Do not use the test split to select prompts, crop padding, alpha, thresholds, checkpoints, or other settings.** If the test results lead to changes, document the follow-up experiment and use a new held-out evaluation rather than presenting the old test as untouched.

### Data layout

```text
openworld/
├── configs/
│   └── config.yaml
├── data/
│   ├── annotations/
│   │   ├── instances_train2017.json             # locally supplied original JSON
│   │   ├── instances_val2017.json               # locally supplied original JSON
│   │   ├── instances_train2017.stripped.json     # base-only training annotations
│   │   ├── instances_val_tune2017.stripped.json
│   │   └── instances_val_test2017.stripped.json
│   ├── images/
│   │   ├── train2017/
│   │   └── val2017/
│   ├── splits/
│   │   ├── train_image_ids.json
│   │   ├── val_tune_image_ids.json
│   │   ├── val_test_image_ids.json
│   │   ├── selected_categories.json
│   │   └── split_metadata.json
│   └── metadata/
├── notebooks/
├── scripts/
├── src/openworld/
├── tests/
└── outputs/
    ├── checkpoints/
    ├── proposals/
    ├── clip/
    ├── predictions/
    ├── metrics/
    └── analysis/
```

Image files, full annotation files, trained checkpoints, prediction outputs, and other large data are local artifacts. **Do not commit them to Git.** Check `.gitignore` carefully: a folder appearing in the repository tree does not guarantee it is ignored. Review changes yourself; no automatic commit is required.

---

## 4. Environment and setup

### Existing installation

If the environment and data are already present, **do not reinstall or redownload them**. Open PowerShell in the project directory and activate the existing environment:

```powershell
cd C:\Users\karki\Downloads\main_openworld\openworld
.\.venv\Scripts\Activate.ps1
```

Open the project in VS Code if it is not already open:

```powershell
code .
```

For notebooks, choose the project's **`.venv` Python kernel**. The first notebook cell should print the project root. A useful check for the notebook's Python is:

```python
import sys
print(sys.executable)
```

The recorded full-training command was run after `deactivate`, using `python` from the PowerShell environment rather than explicitly using `.venv`. Training completed successfully, but its exact Python and library versions were **not recorded**. Notebook runs shown later used the `.venv` path. For a future reproducible run, record the Python, PyTorch, torchvision, OpenCLIP, and pycocotools versions and use the same intended environment consistently.

### Fresh installation only

If setting up on a new Windows machine, create an environment and install CPU PyTorch plus the project dependencies. Do not do this over an existing working environment unless needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install jupyter ipykernel
```

If a notebook kernel is not available, install/register `ipykernel` in the intended environment and select it in VS Code. Installing a new environment is **not** a substitute for preserving the environment information from the completed run.

### First-use model downloads

The ImageNet backbone weights and OpenCLIP weights may download on first use. This is separate from downloading COCO images. The recorded CLIP run was corrected to use `ViT-B-32-quickgelu` with `openai` weights after a QuickGELU mismatch warning.

---

## 5. Preparing or reusing data

**For the completed experiment, reuse the existing split and annotation files.** Do not regenerate the split just to rerun a notebook.

If preparing a genuinely new experiment from the original annotation JSONs, the preparation script creates stripped annotation files, metadata, and fixed image-ID lists:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\config.yaml
```

This may overwrite prepared files. Back up and record the existing splits first if they need to be preserved. Never change the final test split after looking at the results and then describe it as the original held-out split.

### Fetch only selected images if missing

These commands fetch images listed in the prepared subset JSONs; existing verified images are skipped. Run only the subset that is actually missing:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_images.py --subset data\annotations\instances_train2017.stripped.json --dest data\images\train2017 --jobs 16
.\.venv\Scripts\python.exe scripts\fetch_images.py --subset data\annotations\instances_val_tune2017.stripped.json --dest data\images\val2017 --jobs 16
.\.venv\Scripts\python.exe scripts\fetch_images.py --subset data\annotations\instances_val_test2017.stripped.json --dest data\images\val2017 --jobs 16
```

Check the output for `failed=0`; the script can report failed downloads, so do not assume a run succeeded just because it ended. The completed validation-tune check reported `500` images available and `0` missing.

---

## 6. Notebook workflow

The notebook filenames are generally:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Inspect the class protocol, prepared annotations, splits, and sample images |
| `02_train_detector.ipynb` | Run a small training test, then train the base detector |
| `03_proposals_recall.ipynb` | Extract tuning-set proposals and inspect proposal recall |
| `04_clip_recognition.ipynb` | Run CLIP on ground-truth boxes and on proposals; compare settings on val-tune |
| Notebook 05 / `05_final_evaluation_updated.ipynb` | Evaluate saved predictions on the fixed val-test split, calculate intervals, and inspect failures |

The final notebook may have been saved under a different name after its cells were copied into the existing notebook. Use the one containing the **corrected final evaluation cells**.

### Important: do not blindly click “Run All”

The training, proposal-extraction, CLIP, and detector-only cells can take hours on CPU and may overwrite existing files. In the completed run, the expensive test proposal and CLIP outputs already exist. Reuse those files instead of rerunning those cells unless deliberately starting a new experiment. The updated final notebook checks for saved files and can evaluate them without rerunning CLIP.

Some original notebook cells had an outdated smoke-checkpoint path or referred to output files before they were created. Those old cells should not be used unchanged. Restart the kernel after editing imported Python files; saved notebook error output can remain visible until the affected cell is rerun or cleared.

### Notebook 01 — data exploration

Inspect the fixed class lists, selected images and annotations. On an existing run, reuse the prepared data. If the visualization cell cannot find an image, check that the appropriate `train2017` or `val2017` file is present; do not rebuild the whole dataset to fix a missing image.

### Notebook 02 — detector training

The small test used 200 training images for one epoch. It finished successfully and produced a smoke checkpoint under a timestamped folder in `outputs/checkpoints/smoke/`. This was a pipeline check, not the final detector.

The recorded full run used the following command, with the full checkpoint directory distinct from the smoke directory:

```powershell
python scripts\train_detector.py --mode train --config configs\config.yaml --images-root data\images\train2017 --epochs 3 --batch-size 2 --lr 0.005 --checkpoint-dir outputs\checkpoints\full --log-file outputs\metrics\full_train_log.csv
```

For a future run from the existing `.venv`, use `.\.venv\Scripts\python.exe` in place of `python`. **Running this command again retrains the detector and may overwrite existing checkpoint files.** On the recorded CPU, full training took many hours.

The recorded full run finished all three epochs and saved its checkpoints in `outputs/checkpoints/full/`.

### Notebook 03 — proposals

Use the **full** checkpoint, not the smoke checkpoint. The recorded test-proposal command was:

```powershell
.\.venv\Scripts\python.exe scripts\extract_proposals.py --config configs\config.yaml --checkpoint outputs\checkpoints\full\epoch_3.pt --image-ids data\splits\val_test_image_ids.json --annotations data\annotations\instances_val_test2017.stripped.json --images-root data\images\val2017 --out-dir outputs\proposals\val_test --top-n 1000
```

The completed run saved **1,500** test proposal files to `outputs/proposals/val_test/`. This command regenerates proposals, so do not run it again just to inspect the results.

Proposal recall is a separate localization analysis. No final proposal-recall values were provided in the recorded outputs summarized here. Verify the notebook slices proposals to the requested `K` for each recall@K calculation before reporting those numbers; an earlier notebook cell used all loaded proposals in its top-K loop.

### Notebook 04 — CLIP recognition on val-tune

1. **GT boxes + CLIP:** measures class recognition when localization is supplied by ground truth. This is not end-to-end detection AP.
2. **RPN proposals + CLIP:** labels proposed regions and evaluates settings on val-tune.
3. Compare settings on val-tune and choose one before test evaluation. The recorded choice was **alpha 0.5**.

The recorded Experiment B notebook cell contained an old `--checkpoint` argument pointing to a smoke path. The distributed `run_clip.py` version reads saved proposals from `--proposals-dir` and does not use that argument to generate the proposals. For reproducibility, verify the local script and the provenance of the proposal files rather than treating that stale argument as evidence of which detector was used.

### Notebook 05 — final test evaluation

The original notebook attempted to evaluate a prediction JSON before it existed, and its detector-only cell used `--image-ids` where the original training script expected `--val-ids`; it also had an outdated checkpoint path. The final evaluation was instead run using corrected cells that:

1. Verified the full checkpoint, test images, and saved predictions.
2. Used the 1,500 saved test proposal files and CLIP predictions generated with alpha 0.5.
3. Calculated class-aware custom AP50 and image-level bootstrap intervals.
4. Saved a confusion matrix and failure examples.
5. Ran a separate detector-only COCO evaluation.

Do **not** rerun the proposal or CLIP cells merely to obtain the metrics: the saved predictions can be evaluated again directly. Keep test settings fixed.

---

## 7. Recorded results

### 7.1 Ground-truth boxes + CLIP on val-tune

The GT-box recognition run processed **500 images** and saved **2,939 region predictions**. Boxes are supplied by the annotations, so the table below measures recognition given those boxes—not whether the full detector finds them.

| Class | GT boxes | Top-1 accuracy |
|---|---:|---:|
| person | 1,408 | 0.685 |
| bicycle | 51 | 0.647 |
| car | 258 | 0.655 |
| motorcycle | 69 | 0.870 |
| bus | 45 | 0.911 |
| truck | 48 | 0.646 |
| bench | 61 | 0.492 |
| bird | 67 | 0.537 |
| dog | 21 | 0.905 |
| horse | 38 | 0.816 |
| sheep | 51 | 0.863 |
| elephant | 24 | 1.000 |
| backpack | 56 | 0.625 |
| umbrella | 43 | 0.744 |
| bottle | 127 | 0.843 |
| chair | 306 | 0.402 |
| tv | 34 | 0.559 |
| laptop | 26 | 0.885 |
| keyboard | 22 | 0.455 |
| book | 184 | 0.685 |

These values use the stored `gt_category_id` and predicted `category_id`. A previous bug overwrote the ground-truth category; ensure both fields are present before trusting any regenerated GT-box accuracy calculation. In particular, the elephant value comes from only 24 boxes, so it should not be interpreted as proof of perfect general performance.

### 7.2 RPN + CLIP settings on val-tune

After correcting the class-aware AP calculation, the recorded tuning results were:

| Alpha | Base AP50 | Novel AP50 | Generalized AP50 |
|---:|---:|---:|---:|
| 0.0 | 0.037 | 0.023 | 0.034 |
| **0.5** | **0.038** | **0.027** | **0.035** |
| 1.0 | 0.007 | 0.000 | 0.006 |

These are fractions: for example, `0.035` corresponds to `3.5%`. Alpha 0.5 was selected on val-tune before final test evaluation. See the note above about class-score ties at alpha 1.0.

### 7.3 Final RPN + CLIP results on val-test

The final CLIP proposal run saved **150,000 predictions** across **1,500 test images**: 100 regions per image in this run.

| Metric | AP50 | 95% image-bootstrap interval |
|---|---:|---:|
| Base classes | 0.0314 (3.14%) | 0.0288–0.0401 |
| Novel classes | 0.0374 (3.74%) | 0.0282–0.0574 |
| Generalized: all 20 classes | 0.0329 (3.29%) | 0.0314–0.0429 |

The intervals were calculated using 200 image-level resamples with seed 42. They describe uncertainty from resampling this evaluation set conditional on the saved model and predictions; they **do not** measure variability across new training runs, other random seeds, or different datasets.

#### Per-class test AP50

| Class | Split | AP50 |
|---|---|---:|
| person | base | 0.054 |
| car | base | 0.035 |
| chair | base | 0.033 |
| bottle | base | 0.052 |
| bicycle | base | 0.035 |
| motorcycle | base | 0.022 |
| truck | base | 0.022 |
| horse | base | 0.060 |
| sheep | base | 0.020 |
| bench | base | 0.002 |
| laptop | base | 0.038 |
| tv | base | 0.017 |
| book | base | 0.012 |
| backpack | base | 0.020 |
| bird | base | 0.050 |
| bus | novel | 0.004 |
| dog | novel | 0.092 |
| elephant | novel | 0.086 |
| umbrella | novel | 0.004 |
| keyboard | novel | 0.002 |

These are low absolute AP values. They summarize the saved predictions under this particular protocol; they do not identify a single cause of the failures. Proposal quality, region classification, duplicate detections, sparse training, and other design choices may all matter and require separate analysis.

### 7.4 Detector-only baseline on val-test

The separately run **COCO-style detector evaluation** uses the 15 base categories. It produced:

| Metric | Result |
|---|---:|
| AP at IoU 0.50 | 0.073 (7.3%) |
| AP averaged over IoU 0.50–0.95 | 0.027 (2.7%) |
| AP at IoU 0.75 | 0.014 (1.4%) |
| Average recall, IoU 0.50–0.95, up to 100 detections | 0.082 (8.2%) |

This detector cannot directly classify the five novel classes as novel labels. The detector-only scores above use COCO evaluation; the RPN + CLIP AP50 scores use a **custom evaluator**, with different handling and conventions. They are **not a controlled, identical-metric comparison**. Do not claim from these tables alone that adding CLIP improved or worsened the detector. To make a fair comparison, evaluate comparable outputs on the same images, class set, matching rules, and metric implementation.

---

## 8. Understanding the metrics and figures

### Custom AP50

The final RPN + CLIP notebook ranks detections by score and computes interpolated AP at 101 recall points for each class. A correct prediction must match a ground-truth box of the **same class** with IoU strictly greater than `0.5`. The reported base, novel, and generalized values are means of their per-class values. The recorded implementation excludes crowd ground-truth boxes and assigns zero AP to classes without ground truth in a resample.

This is a project-specific evaluator, not a statement that the results were computed by the official COCO evaluation protocol. AP is not the same as simple classification accuracy.

### Detector-only COCO metrics

`AP50` is AP at IoU 0.50. `AP50–95` averages AP over multiple stricter IoU thresholds from 0.50 to 0.95. COCO recall figures also have specified IoU ranges and maximum detection counts. Consult the printed evaluator settings rather than treating every recall number as a percentage of all objects found at one threshold.

### Confusion matrix and gallery

The final notebook saves:

```text
outputs/metrics/confusion_matrix.png
outputs/analysis/gallery/
```

The matrix matches predictions to ground-truth objects by overlap. Rows are actual classes; columns are predicted classes. The last column is **missed**. The matrix does **not** include an extra row for unmatched false-positive predictions. The gallery contains up to 12 examples; they are examples in processing order, not a ranked or exhaustive error analysis. Green boxes mark ground truth and red boxes mark a matched prediction.

If `FigureCanvasAgg is non-interactive` appears, the plot may not display inside the notebook, but the notebook can still save the PNG. Open the saved image in VS Code to view it.

---

## 9. Outputs from the completed run

| File or folder | Description |
|---|---|
| `outputs/checkpoints/full/epoch_1.pt` through `epoch_3.pt` | Full-training detector checkpoints |
| `outputs/checkpoints/smoke/` | Small training-test checkpoint(s) |
| `outputs/metrics/full_train_log.csv` | Full-training log; inspect column names before interpreting box losses |
| `outputs/proposals/val_tune/` | Saved proposal files for tuning |
| `outputs/proposals/val_test/` | 1,500 saved test proposal files |
| `outputs/clip/val_tune_gt/predictions.json` | GT-box CLIP predictions on val-tune |
| `outputs/clip/val_tune_prop_a0.0/` | Tuning predictions at alpha 0.0 |
| `outputs/clip/val_tune_prop_a0.5/` | Tuning predictions at alpha 0.5 |
| `outputs/clip/val_tune_prop_a1.0/` | Tuning predictions at alpha 1.0; see tie caveat |
| `outputs/clip/val_test_proposals/predictions.json` | 150,000 saved final CLIP proposal predictions |
| `outputs/metrics/openworld_eval.json` | Final custom RPN + CLIP metrics and intervals |
| `outputs/predictions/detector_val_test.json` | Detector-only test predictions |
| `outputs/metrics/detector_baseline_val_test.json` | Detector-only COCO summary |
| `outputs/metrics/confusion_matrix.png` | Saved confusion matrix |
| `outputs/analysis/gallery/` | Failure examples |

These files may be overwritten if the corresponding cells are run again. Back them up before intentional regeneration.

---

## 10. Troubleshooting and fixes made during the run

### Training says the five novel categories are missing

The stripped training JSON deliberately has only the 15 base categories. The base label-map helper must resolve **base categories only**. In the corrected `src/openworld/categories.py`, empty lists are preserved using explicit `is None` checks, and `label_map_for_base` calls `resolve_categories(..., novel_classes=[])`.

After changing an imported module, restart the notebook kernel. Verify that the base label map contains 15 labels numbered 1–15. Do not add novel annotations to the training JSON to work around this error.

### Training looks for images in `data/images/`

The training JPEGs are under `data/images/train2017/`. Pass `--images-root data\images\train2017` when using the original training script. Validation images are under `data/images/val2017/`.

### `'str' object has no attribute 'to'` during training

The dataset target includes a filename string as well as tensors. The training loop should move only tensor values to the device; the filename is not a model target tensor.

### Box-flipping fix

A horizontal flip must swap the left and right x-coordinates. The corrected line in `src/openworld/dataset.py` is:

```python
boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
```

### Box-loss fields show zero in an older log

An earlier version logged keys named `loss_box_loss` and `loss_rpn_box_loss`, while torchvision returned `loss_box_reg` and `loss_rpn_box_reg`. The console showed nonzero box losses, so zeros under the mismatched log names should not be interpreted as measured zero loss. Verify the log columns against the training code before using them.

### `UnpicklingError` when loading your checkpoint

Newer PyTorch versions may default to loading weights only, while the project's checkpoint also stores optimizer and random-state information. The local fix was to use `weights_only=False` when loading the user's own checkpoint. **Only do this for a checkpoint you trust**, because loading a full checkpoint can execute code through Python pickle.

### Missing validation image

Train, val-tune, and val-test require different subsets. The existence of a `val2017` folder does not prove every needed image is present. Fetch the missing prepared subset only, verify `failed=0`, and check again.

### `'Image' object has no attribute 'to'` in proposal extraction

The dataset supplies PIL images. Proposal extraction must convert them to tensors before sending them to the model. The project also received a corrected proposal-extraction implementation that returns boxes in original image coordinates. Restart the notebook kernel after replacing these modules.

### Wrong AP from earlier evaluation code

An earlier evaluator could match a predicted class to ground truth from another class and could omit ground-truth objects in images with no predictions for that class. The fix filters ground truth by category across **all** evaluation images. A small test in which a class-1 prediction overlaps only a class-2 object must return `0.0` for class 1, not `1.0`. The recorded tuning scores and the final notebook results above were recalculated after that correction.

### Old notebook 05 errors

- `--image-ids` was not accepted by the original detector training/evaluation script, which used `--val-ids`.
- An old cell pointed at `outputs/checkpoints/epoch_3.pt` instead of the recorded `outputs/checkpoints/full/epoch_3.pt`.
- Evaluation and gallery cells attempted to open test CLIP predictions before the predictions had been generated.

Use the corrected final notebook and saved files instead of running those old cells unchanged.

### CLIP warnings

- **QuickGELU mismatch:** set the model to `ViT-B-32-quickgelu` with the recorded `openai` weights, and rerun the affected recognition predictions if necessary.
- **IProgress / ipywidgets warning:** affects notebook progress display, not necessarily the computation.
- **Hugging Face unauthenticated or Windows symlink warnings:** downloads/cache can still work; check the actual run output for failures.
- **Non-writable NumPy array warning:** was printed in the recorded CLIP run; the run continued. Review preprocessing if modifying the code rather than assuming every warning is harmless.
- **Tensor-to-scalar warning while training:** the recorded smoke run continued. A future code cleanup can use a detached tensor when logging, without changing the model update.

---

## 11. Limitations and next steps

- The 20 classes are a **custom subset**, not the full OV-COCO benchmark.
- The dataset and training run were deliberately small and CPU-limited. The recorded test AP values are low.
- Novel objects can be present in training images without novel labels.
- Ground-truth-box CLIP accuracy answers a different question from end-to-end detection AP.
- The custom RPN + CLIP evaluator and COCO detector evaluator differ, so a direct improvement claim is not justified by the two numbers alone.
- Alpha 1.0 requires special care because class scores tie under the recorded fusion formula.
- The final confusion matrix excludes unmatched false positives, and the gallery is illustrative rather than exhaustive.
- The exact software versions used by the full detector-training command were not recorded.
- The test results should be treated as final for the recorded protocol, not used to pick new settings.
- Classical CV demonstrations, a hand-built anchor implementation, Grounding DINO/OWLv2 reference implementations, a full crossed ablation matrix, and a web demo were outside this run.

For a future study, document the exact environment, validate proposal recall with correctly sliced `K`, compare methods with the same evaluation protocol, record additional seeds where feasible, and use an appropriate new holdout for any changes made after inspecting the final test.
