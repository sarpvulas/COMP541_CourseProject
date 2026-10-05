"""One-epoch smoke test of main.py on a tiny synthetic COCO dataset (CPU, W&B disabled).

Builds 16 train and 10 test images (640x640, flat gray, 1-3 random boxes each, class ids 1-10)
in a temporary folder, points the code at them, and runs main.main() for one epoch.
It only shows that the train/eval loop runs end to end; the numbers mean nothing.
Needs the ResNet-50 ImageNet weights (downloaded on first run). Run from anywhere:
    python scripts/smoke_test.py
"""
import json, os, random, sys, tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["WANDB_MODE"] = "disabled"

rng = random.Random(0)
tmp = Path(tempfile.mkdtemp(prefix="uod_smoke_"))
lines = []
for split, n in (("train", 16), ("test", 10)):
    (tmp / split).mkdir()
    images, anns = [], []
    for i in range(1, n + 1):
        Image.new("RGB", (640, 640), (rng.randint(0, 255),) * 3).save(tmp / split / f"{i}.jpg")
        images.append({"id": i, "file_name": f"{i}.jpg", "width": 640, "height": 640})
        for _ in range(rng.randint(1, 3)):
            x, y, w, h = rng.randint(0, 500), rng.randint(0, 500), rng.randint(40, 140), rng.randint(40, 140)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": rng.randint(1, 10),
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    cats = [{"id": k, "name": str(k)} for k in range(1, 11)]
    (tmp / f"{split}.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    lines += [f"{split.upper()}_IMAGES_PATH={tmp / split}", f"{split.upper()}_ANNOTATIONS_PATH={tmp / f'{split}.json'}"]
(tmp / "paths.txt").write_text("\n".join(lines) + "\n")
os.environ["UOD_PATHS_FILE"] = str(tmp / "paths.txt")

import modules.config as config
config.EPOCHS = 1
import main
main.EPOCHS = 1
os.chdir(tmp)  # checkpoints go to <tmp>/models/checkpoints
main.main()
print(f"smoke test finished; data and checkpoint in {tmp}")
