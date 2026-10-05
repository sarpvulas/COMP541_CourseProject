# Underwater image classification with a dual-stream ResNet-50 backbone (TOOD-style detection head unfinished)

PyTorch code for the COMP 541 (Deep Learning) course project at Koç University by Hüseyin Sarp Vulaş and Zeynep Aydın. It trains a single-label image classifier on the RUOD underwater dataset, using a two-branch ResNet-50 with edge enhancement and gated fusion. A TOOD-style detection head is included but is not finished or trained.

[![tests](https://github.com/sarpvulas/COMP541_CourseProject/actions/workflows/tests.yml/badge.svg)](https://github.com/sarpvulas/COMP541_CourseProject/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Companion repository: [sarpvulas/COMP541-Project](https://github.com/sarpvulas/COMP541-Project) (the UODReview platform snapshot). See [Relationship to COMP541-Project](#relationship-to-comp541-project).

## TL;DR

Underwater images are blurry and low in contrast, which makes standard detectors less reliable. This project feeds each image and an edge-enhanced grayscale copy into a two-branch ResNet-50, fuses the branches with learned gates, and classifies the image. A TOOD-style detection head was started but has no label assignment or loss, so the repository has no detection results and no measured accuracy yet.

## Results

TODO(sarp): add mAP results and the figure. The repository contains no measured results and no report. `main.py` trains a classifier, reports top-1 accuracy per epoch on a validation split, and reports test top-1 accuracy once at the end; no run's numbers are stored here. There are no detection metrics.

## What is in the repo

| Part | File | Status |
|------|------|--------|
| Illumination/grayscale module (IDM) | `modules/IDM.py` | Used by both pipelines |
| Edge enhancement module (IEEM) | `modules/IEEM.py` | Used by both pipelines (see Limitations) |
| Dual ResNet-50 (RGB branch and edge branch, ImageNet-initialised, weights not shared; ImageNet mean/std normalisation on the RGB branch) | `modules/dual_backbone.py` | Used by both pipelines |
| Gated fusion of the two branches (TAGFFM) per ResNet stage | `modules/TAGFFM.py` | Used by both pipelines; sizes hard-coded for 512x512 input |
| FPN (torchvision) | `modules/Pipeline*.py` | Used by both pipelines |
| Classification head | `MultiClassHead.py`, `modules/Pipeline_OnlyClassify.py` | Trained by `main.py` |
| Center crop, box filter and label rule shared by train and eval | `modules/crop.py` | Unit tested |
| TOOD-style head | `TOODHead.py`, `modules/Pipeline.py` | Forward pass only; not trained (see Limitations) |
| Anchors, IoU, matching, box decoding, GIoU | `modules/anchor_utils.py` | Unit tested |
| Anchor-based detection loss (cross-entropy + GIoU) | `losses.py` (`detection_loss`, `giou_regression_loss`) | Not used by `main.py`; the GIoU part is unit tested, the classification part is not a usable loss (see Limitations) |
| COCO-format data loading, fixed-seed train/validation split | `utils.py` | Used |
| Seeding | `modules/seed.py` | Used by `main.py` |

## Method

```mermaid
flowchart LR
    A[RGB image 512x512] --> B[IDM: learned grayscale x3]
    B --> C[IEEM: Laplacian + conv blocks]
    A --> D[ResNet-50, RGB branch]
    C --> E[ResNet-50, edge branch]
    D --> F[TAGFFM fusion, stages 1-4]
    E --> F
    F --> G[FPN, 5 levels]
    G --> H[Classification head: used in main.py]
    G --> I[TOOD-style head: Pipeline.py, not trained]
```

What `main.py` does: for each image it center-crops to 512x512 (`modules/crop.py`), keeps the boxes that are at least 50% inside the crop, and labels the image with the class whose remaining boxes cover the largest total area. The classification head produces per-scale class maps; they are averaged spatially and across scales, and trained with cross-entropy (the whole model is trained, not only the head). Training and evaluation use the same crop and label rule. The RGB image is scaled to [0, 1] and then normalised with the ImageNet mean and std inside `DualResNet50` (RGB branch only; the edge branch gets the Laplacian-derived features unchanged), so train, validation and test see identical inputs. 10% of the training images (fixed split seed 1234) are held out as a validation split. After each epoch it computes top-1 accuracy on that validation split, logs it to Weights & Biases, and saves a checkpoint to `models/checkpoints/` when validation accuracy is strictly higher than the best so far (an accuracy of 0 never saves). After the last epoch it reloads the best checkpoint and evaluates the test split once; the test split is never used to choose a checkpoint. If no epoch ever saved a checkpoint, the last-epoch weights are tested instead and the log says so. Images with no usable box in the crop are skipped, and the count is printed each epoch.

The TOOD-style head (`TOODHead.py`) is adapted from MMDetection 3.3.0 `tood_head.py` (Apache-2.0), which implements TOOD (Feng et al., ICCV 2021, https://arxiv.org/abs/2108.07755). The head layout, initialisation, task decomposition with layer attention and the classification alignment map come from there. Task-aligned label assignment and the TOOD losses are not implemented, so the head is not trained anywhere in this repository. The deformable-offset module is a registered submodule created once in `__init__` (it used to be rebuilt with random weights on every forward call); a test checks that its parameters are registered and receive gradients. Without `mmcv` the deformable sampling uses `torchvision.ops.deform_conv2d`; that path is what the tests exercise, and it was not compared numerically with the `mmcv` op.

## Tech stack

Python 3.12, PyTorch, torchvision (ResNet-50, FPN, `CocoDetection`), pycocotools, Weights & Biases, matplotlib. `mmcv` is optional and only used by the TOOD-style head (torchvision's op is the fallback).

## Getting the data

The code expects RUOD (Real-world Underwater Object Detection, 10 classes) in COCO format:

```
RUOD/
  train/                          # training images
  test/                           # test images
  annotations_json/
    instances_train.json
    instances_test.json
```

TODO(sarp): add the official RUOD download link and citation. Copy `modules/dataset_paths.example.txt` to `modules/dataset_paths.txt` (git-ignored) and set the four paths; you can also point `UOD_PATHS_FILE` at another file. The loader maps the sorted COCO category ids to class indices 0-9 (for ids 1 to 10 this is the old `category_id - 1` rule) and raises an error if an annotation file does not have exactly 10 categories. The RUOD files themselves have not been checked, since the dataset was not available.

## Quickstart

Tested on macOS, Python 3.12.2, CPU only.

```bash
git clone https://github.com/sarpvulas/COMP541_CourseProject.git
cd COMP541_CourseProject
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
python -m pytest -q          # 60 passed
```

Forward and backward pass of the classification pipeline on one random 512x512 image (downloads the ResNet-50 ImageNet weights on first run):

```bash
python - <<'EOF'
import torch
from modules.Pipeline_OnlyClassify import FullPipeline_OnlyClassify
from losses import single_label_classification_loss
m = FullPipeline_OnlyClassify(num_classes=10)
out = m(torch.rand(1, 3, 512, 512))
print([tuple(o.shape) for o in out])
loss = single_label_classification_loss(out, [{"labels": torch.tensor(3)}])
loss.backward()
EOF
```

`python main.py [--seed N]` needs RUOD and a Weights & Biases login (`WANDB_MODE=disabled` turns W&B off). It has not been run on RUOD.

Smoke test: `python scripts/run_smoke.py` builds a tiny synthetic COCO dataset in a temp folder (16 train and 10 test images of 640x640, flat gray, 1-3 random boxes each), runs `main.main()` for one epoch on CPU with W&B disabled, and prints the usual training and evaluation lines. With the default seed (0) it printed `Avg Loss: 2.3281 over 7 batches; skipped 0 images`, validation `top-1 accuracy: 0.0000 (0/2)` and test `top-1 accuracy: 0.3000 (3/10)`; a second run printed the same loss, and `UOD_SEED=1` printed 2.3233. The data is random, so these numbers only show that the loop runs and is repeatable on CPU. It needs the ResNet-50 ImageNet weights, so it is not part of CI.

Optional debug switches, all off by default: `UOD_DEBUG=1` (tensor stats and NaN checks in `modules/Debug.py` and in `TOODHead`), `UOD_DETECT_ANOMALY=1`, `UOD_WANDB_WATCH=1`. `UOD_DEBUG` alone does not make `IDM.forward` print; that also needs `debug=True` in the call.

## Reproducibility notes

- Settings in `modules/config.py`: batch size 2, learning rate 1e-4, 10 epochs, AdamW with weight decay 1e-4, StepLR (step 2, gamma 0.5).
- Seed: `python main.py --seed N` or `UOD_SEED=N` (default 0). It seeds Python, NumPy and PyTorch, the choice of the 500-image subsets, the training shuffle order, and the model's weight initialisation; `main.py` also sets the cuDNN deterministic flags. In the smoke test two CPU runs with the same seed gave the same training loss.
- Split: 10% of the (size-filtered) training images form the validation split, drawn with a separate fixed seed (1234), so it does not change with `--seed`. `utils.load_data` then takes a random subset of at most 500 images from each of the train, validation and test parts, and drops images smaller than 512 pixels on either side before any of this.
- Where determinism is not guaranteed: on a GPU, some CUDA backward kernels (adaptive average pooling and the FPN's upsampling) use atomic adds, so results can differ between runs even with the same seed; `torch.use_deterministic_algorithms` is not enabled because it raises on those kernels. Multi-GPU runs (`DataParallel`) are not covered. Different PyTorch versions or hardware can change results. Only CPU repeatability was checked. `num_workers` is 0, so there is no worker-seeding issue.
- Dependency versions used for the checks above are listed in `requirements.txt`.
- Changes from the earlier code that alter results: ImageNet normalisation was added, checkpoints are now chosen on a validation split and the test metric is reported once, the seed is fixed by default, the offset module of the TOOD-style head is now trainable (it is not used by `main.py`), and class indices come from the sorted COCO category ids (identical for ids 1 to 10). Any numbers from the earlier code are not comparable: its checkpoints and "best" accuracies were selected on test accuracy, so they are also optimistic.

## Limitations

- No detection and no results. The model solves a single-label task that is not a standard benchmark: the label is a heuristic (the class with the largest total box area inside the center crop), so accuracy is not comparable with published classification or detection numbers.
- The validation split is a random 10% of the training images, so images of the same scene or dive may appear in both train and validation if the dataset has such near-duplicates; this was not checked. Each split is also reduced to a 500-image subset by default.
- `modules/IEEM.py` concatenates the 12 learned feature channels with the 3 Laplacian channels and then keeps only the first three channels of the result, which come from the learned convolution branch. The raw Laplacian channels are not in the output; they influence it only through the input of the learned branch. The code and comments do not say which reduction to 3 channels was intended, so this is documented and left unchanged, not fixed.
- ImageNet normalisation is applied to the RGB branch only. The edge branch is fed Laplacian-derived features that have no ImageNet statistics, although its ResNet-50 is ImageNet-pretrained.
- Fixing IEEM later will change results. The output keeps only the first three channels of the concatenation, which are `f_l` channels, so the raw Laplacian channels never reach the output directly (they only enter through the input of the learned branch). The intent of the "Fuse features" step appears to be an output that depends on the Laplacian; the open question is how to reduce the 15 channels to 3. Candidate fixes are a learned 1x1 convolution from 15 to 3 channels, or `f_l[:, :3] + e_cat`. Neither is applied here.
- The TOOD-style head still has no label assignment or loss. Its offset module is now trainable, but nothing trains it. `modules/Pipeline.py` ran end to end in a manual check through the torchvision fallback (a 1x3x512x512 forward and backward pass, with a non-zero gradient reaching the offset module); it was not run with `mmcv` because `mmcv` was not installed.
- `losses.detection_loss` is unused and is not a finished loss: it treats every non-positive anchor as class 0 (there is no background class) and feeds probabilities to cross-entropy as logits. Its regression term now uses the matched-pair GIoU (`giou_regression_loss`, tested). The earlier call to `torchmetrics` GIoU was checked against torchmetrics 1.9.0, where it already returned the mean of the diagonal, so it was not wrong there; it is replaced to remove the dependency on that library's behaviour. `losses.py` also keeps its own `box_iou` and `match_anchors`, which duplicate `modules/anchor_utils.py` with different default thresholds.
- Batch size 2 with BatchNorm layers in train mode is noisy; this was not tuned or tested.
- TAGFFM hard-codes the 512x512 feature sizes of the four ResNet stages.
- CI was updated to install `wandb` and `pycocotools` (imported by the tests) and `matplotlib` (listed in `requirements.txt`, imported lazily by the debug plot) but the workflow has not been run on GitHub from this branch.

## Relationship to COMP541-Project

[COMP541-Project](https://github.com/sarpvulas/COMP541-Project) and this repository are two parts of one COMP 541 course project on underwater object detection, done by Hüseyin Sarp Vulaş and Zeynep Aydın. COMP541-Project was created on 2024-12-12; its first commit holds a snapshot of the UODReview platform (Long Chen et al., built on MMDetection). This repository was committed on 2025-01-27 and holds the course pipeline code (dual ResNet-50 backbone, IDM, IEEM, TAGFFM, a TOOD-style head). Comparing the two trees: no file is byte-identical; the shared file names (`config.py`, `train.py`, `utils.py`, `LICENSE`, `README.md`, `.gitignore`) have different contents and roles; the pipeline imports nothing from `mmdet` or `mmengine`, does not read the platform's configs, and the platform contains none of the pipeline's modules. The one reuse is that `TOODHead.py` is a rewritten derivative of the platform's `mmdet/models/dense_heads/tood_head.py` (MMDetection 3.3.0), and it can optionally call `mmcv.ops.deform_conv2d` from MMCV. Neither repository contains training logs, results tables or the course report.

## License and credits

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The repository is licensed Apache-2.0 because `TOODHead.py` is derived from [MMDetection](https://github.com/open-mmlab/mmdetection) (OpenMMLab, Apache-2.0). An earlier revision of this repository carried an MIT license; the owner should confirm the change.

- TOOD: Feng et al., "TOOD: Task-aligned One-stage Object Detection", ICCV 2021.
- MMDetection: task decomposition, head layout and initialisation in `TOODHead.py`.
- ResNet-50 weights and FPN from torchvision.
- Course project for COMP 541 Deep Learning, Koç University, by Hüseyin Sarp Vulaş and Zeynep Aydın.
- Companion repository: [sarpvulas/COMP541-Project](https://github.com/sarpvulas/COMP541-Project), a snapshot of the UODReview platform by Long Chen et al.

TODO(sarp): add a LinkedIn link.

Authors: Hüseyin Sarp Vulaş and Zeynep Aydın, Koç University, COMP 541.
