# Underwater image classification with a dual-stream ResNet-50 backbone (TOOD-style detection head unfinished)

PyTorch code for the COMP 541 course project at Koç University. It trains a single-label image classifier on the RUOD underwater dataset, using a two-branch ResNet-50 with edge enhancement and gated fusion. A TOOD-style detection head is included but is not finished or trained.

[![tests](https://github.com/sarpvulas/COMP541_CourseProject/actions/workflows/tests.yml/badge.svg)](https://github.com/sarpvulas/COMP541_CourseProject/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## TL;DR

Underwater images are blurry and low in contrast, which makes standard detectors less reliable. This project feeds each image and an edge-enhanced grayscale copy into a two-branch ResNet-50, fuses the branches with learned gates, and classifies the image. A TOOD-style detection head was started but has no label assignment or loss, so the repository has no detection results and no measured accuracy yet.

## Results

TODO(sarp): add mAP results and the figure. The repository contains no measured results. `main.py` trains a classifier and reports top-1 accuracy per epoch on the test split; no run's numbers are stored here. There are no detection metrics.

## What is in the repo

| Part | File | Status |
|------|------|--------|
| Illumination/grayscale module (IDM) | `modules/IDM.py` | Used by both pipelines |
| Edge enhancement module (IEEM) | `modules/IEEM.py` | Used by both pipelines (see Limitations) |
| Dual ResNet-50 (RGB branch and edge branch, ImageNet-initialised, weights not shared) | `modules/dual_backbone.py` | Used by both pipelines |
| Gated fusion of the two branches (TAGFFM) per ResNet stage | `modules/TAGFFM.py` | Used by both pipelines; sizes hard-coded for 512x512 input |
| FPN (torchvision) | `modules/Pipeline*.py` | Used by both pipelines |
| Classification head | `MultiClassHead.py`, `modules/Pipeline_OnlyClassify.py` | Trained by `main.py` |
| Center crop, box filter and label rule shared by train and eval | `modules/crop.py` | Unit tested |
| TOOD-style head | `TOODHead.py`, `modules/Pipeline.py` | Forward pass only; not trained (see Limitations) |
| Anchors, IoU, matching, box decoding, GIoU | `modules/anchor_utils.py` | Unit tested |
| Anchor-based detection loss (cross-entropy + GIoU) | `losses.py` (`detection_loss`) | Not used by `main.py` |
| COCO-format data loading | `utils.py` | Used |

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

What `main.py` does: for each image it center-crops to 512x512 (`modules/crop.py`), keeps the boxes that are at least 50% inside the crop, and labels the image with the class whose remaining boxes cover the largest total area. The classification head produces per-scale class maps; they are averaged spatially and across scales, and trained with cross-entropy (the whole model is trained, not only the head). Training and evaluation use the same crop and label rule. After each epoch it computes top-1 accuracy on the test split, logs it to Weights & Biases, and saves a checkpoint to `models/checkpoints/` when accuracy improves. Images with no usable box in the crop are skipped, and the count is printed each epoch.

The TOOD-style head (`TOODHead.py`) is adapted from MMDetection 3.3.0 `tood_head.py` (Apache-2.0), which implements TOOD (Feng et al., ICCV 2021, https://arxiv.org/abs/2108.07755). The head layout, initialisation, task decomposition with layer attention and the classification alignment map come from there. Task-aligned label assignment and the TOOD losses are not implemented, and the deformable-offset module is rebuilt with random weights on every forward call, so the head cannot be trained as written.

## Tech stack

Python 3.12, PyTorch, torchvision (ResNet-50, FPN, `CocoDetection`), pycocotools, Weights & Biases, matplotlib. `mmcv` is needed only for the TOOD-style head.

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

TODO(sarp): add the official RUOD download link and citation. Copy `modules/dataset_paths.example.txt` to `modules/dataset_paths.txt` (git-ignored) and set the four paths; you can also point `UOD_PATHS_FILE` at another file. The loader subtracts 1 from every COCO `category_id` to get class indices 0-9. That assumes RUOD ids run from 1 to 10; this has not been checked against the dataset files.

## Quickstart

Tested on macOS, Python 3.12.2, CPU only.

```bash
git clone https://github.com/sarpvulas/COMP541_CourseProject.git
cd COMP541_CourseProject
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
python -m pytest -q          # 39 passed
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

`python main.py` needs RUOD and a Weights & Biases login (`WANDB_MODE=disabled` turns W&B off). It has been run for one epoch on a small synthetic COCO dataset (8 train and 6 test images of 640x640, random boxes) on CPU, which only shows that the loop works; it has not been run on RUOD.

Optional debug switches, all off by default: `UOD_DEBUG=1` (print tensor stats and check for NaNs), `UOD_DETECT_ANOMALY=1`, `UOD_WANDB_WATCH=1`.

## Reproducibility notes

- Settings in `modules/config.py`: batch size 2, learning rate 1e-4, 10 epochs, AdamW with weight decay 1e-4, StepLR (step 2, gamma 0.5).
- `utils.load_data` uses a random 500-image subset of each split and drops images smaller than 512 pixels on either side. No random seed is set, so runs are not repeatable.
- Dependency versions used for the checks above are listed in `requirements.txt`.

## Limitations

- No detection and no results. The model solves a single-label task that is not a standard benchmark: the label is a heuristic (the class with the largest total box area inside the center crop), so accuracy is not comparable with published classification or detection numbers.
- Checkpoints are selected on the test split; there is no validation split, so the reported accuracy is optimistically biased.
- `modules/IEEM.py` concatenates 12 learned feature channels with the 3 Laplacian channels and then keeps only the first three channels of the result, which come from the learned convolution branch. The raw Laplacian channels are discarded, so "Laplacian edge enhancement" is only loosely accurate.
- No ImageNet mean/std normalisation of inputs, although the backbone is ImageNet-pretrained.
- No random seed.
- `utils.save_model` also saves a checkpoint whenever accuracy is exactly 0.
- The TOOD-style head has no label assignment or loss, and its offset module is rebuilt with random weights on every forward call. `modules/Pipeline.py` and `TOODHead.py` were not run because `mmcv` was not installed.
- `losses.detection_loss` averages a pairwise GIoU matrix instead of paired boxes and is not used by `main.py`.
- TAGFFM hard-codes the 512x512 feature sizes of the four ResNet stages.

## Related

[sarpvulas/COMP541-Project](https://github.com/sarpvulas/COMP541-Project) is a working snapshot of the UODReview platform by Long Chen et al. (built on MMDetection).

## License and credits

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The repository is licensed Apache-2.0 because `TOODHead.py` is derived from [MMDetection](https://github.com/open-mmlab/mmdetection) (OpenMMLab, Apache-2.0). An earlier revision of this repository carried an MIT license; the owner should confirm the change.

- TOOD: Feng et al., "TOOD: Task-aligned One-stage Object Detection", ICCV 2021.
- MMDetection: task decomposition, head layout and initialisation in `TOODHead.py`.
- ResNet-50 weights and FPN from torchvision.
- Course project for COMP 541 Deep Learning, Koç University.

TODO(sarp): confirm co-author credit (the companion repository names Zeynep Aydın as a co-author of the course project). TODO(sarp): add a LinkedIn link.

Author: Sarp Vulaş, Dubai. MSc Computational Finance, King's College London.
