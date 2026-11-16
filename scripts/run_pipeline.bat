@echo off
rem OpenWorld pipeline runner for Windows CPU. Run from the repo root.
set CONFIG=configs/config.yaml

echo ============ [0] Prepare data manifests ============
python scripts/prepare_data.py --config %CONFIG%

rem ============ [1] Download images (run once, needs internet) ============
rem python scripts/fetch_images.py --subset data/annotations/instances_train2017.stripped.json --dest data/images/train2017 --jobs 32
rem python scripts/fetch_images.py --subset data/annotations/instances_val_test2017.stripped.json --dest data/images/val2017 --jobs 32

echo ============ [2] Smoke test: train on 200 images, 1 epoch ============
python scripts/train_detector.py --mode train --config %CONFIG% --limit-images 200 --epochs 1 --batch-size 2 --lr 0.005

rem ============ [3] Full training (uncomment when ready) ============
rem python scripts/train_detector.py --mode train --config %CONFIG% --epochs 3 --batch-size 2

rem ============ [4] Extract RPN proposals ============
rem python scripts/extract_proposals.py --config %CONFIG% --checkpoint outputs/checkpoints/epoch_3.pt --image-ids data/splits/val_test_image_ids.json --annotations data/annotations/instances_val_test2017.stripped.json --images-root data/images/val2017 --out-dir outputs/proposals/val_test --top-n 1000

rem ============ [5] CLIP recognition (ground-truth upper bound) ============
rem python scripts/run_clip.py --mode gt --config %CONFIG% --image-ids data/splits/val_test_image_ids.json --annotations data/annotations/instances_val_test2017.stripped.json --images-root data/images/val2017 --out-dir outputs/clip/val_test_gt

rem ============ [6] CLIP recognition (RPN proposals + fusion) ============
rem python scripts/extract_proposals.py --config %CONFIG% --checkpoint outputs/checkpoints/epoch_3.pt --image-ids data/splits/val_test_image_ids.json --annotations data/annotations/instances_val_test2017.stripped.json --images-root data/images/val2017 --out-dir outputs/proposals/val_test --top-n 1000
rem python scripts/run_clip.py --mode proposals --config %CONFIG% --checkpoint outputs/checkpoints/epoch_3.pt --image-ids data/splits/val_test_image_ids.json --annotations data/annotations/instances_val_test2017.stripped.json --images-root data/images/val2017 --proposals-dir outputs/proposals/val_test --top-k 100 --alpha 0.5 --out-dir outputs/clip/val_test_proposals

rem ============ [7] Final evaluation ============
rem python scripts/evaluate_openworld.py --predictions outputs/clip/val_test_proposals/predictions.json --annotations data/annotations/instances_val_test2017.stripped.json --selected-categories data/splits/selected_categories.json --image-ids data/splits/val_test_image_ids.json --out-dir outputs/metrics

echo Done.
