# modules/config.py

# ===========================
# Dataset Paths
# ===========================

import os

# Function to load paths from a text file
def load_paths(file_path):
    paths = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                paths[key.strip()] = value.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Path configuration file '{file_path}' not found.")
    return paths

# Paths file lookup order: $UOD_PATHS_FILE, modules/dataset_paths.txt (git-ignored, yours),
# then modules/dataset_paths.example.txt (placeholders, so imports and tests work without data).
_HERE = os.path.dirname(__file__)
PATHS_FILE = os.environ.get("UOD_PATHS_FILE") or next(
    (p for p in (os.path.join(_HERE, "dataset_paths.txt"),
                 os.path.join(_HERE, "dataset_paths.example.txt")) if os.path.exists(p)),
    os.path.join(_HERE, "dataset_paths.txt"),
)
paths = load_paths(PATHS_FILE)

# Dataset Paths
TRAIN_IMAGES_PATH = paths.get("TRAIN_IMAGES_PATH", "")
TRAIN_ANNOTATIONS_PATH = paths.get("TRAIN_ANNOTATIONS_PATH", "")
TEST_IMAGES_PATH = paths.get("TEST_IMAGES_PATH", "")
TEST_ANNOTATIONS_PATH = paths.get("TEST_ANNOTATIONS_PATH", "")


# ===========================
# Training Parameters
# ===========================

BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 10
IMAGE_SIZE = (512, 512)  # (Width, Height)

# ===========================
# Seed and data split
# ===========================

# Training seed: data order, subset choice, weight initialisation. Override with the
# UOD_SEED environment variable or `python main.py --seed N`.
try:
    SEED = int(os.environ.get("UOD_SEED", "0"))
except ValueError:
    raise ValueError(
        f"UOD_SEED must be an integer, got {os.environ.get('UOD_SEED')!r}") from None

# A validation split is carved from the training set. Checkpoints are selected on it; the
# test split is evaluated once, on the selected checkpoint, at the end of training.
# The split uses its own fixed seed so it is the same for every training seed.
VAL_FRACTION = 0.1
VAL_SPLIT_SEED = 1234
SUBSET_SIZE = 500  # images drawn from each of the train, validation and test splits

# ===========================
# Anchor Generation Parameters
# ===========================

# Feature map sizes corresponding to different FPN levels.
# These should match the output feature maps from your backbone network.
FEATURE_MAP_SIZES = [
    (128, 128),  # Example: stride 4 (512 / 4)
    (64, 64),    # Example: stride 8 (512 / 8)
    (32, 32),    # Example: stride 16 (512 / 16)
    (16, 16),    # Example: stride 32 (512 / 32)
    (8, 8)       # Example: stride 64 (512 / 64)
]

# Anchor scales (in pixels). Adjust based on the dataset's object size distribution.
SCALES = [256, 512, 1024]
#SCALES = [32]

# Anchor aspect ratios. Defines the width to height ratio of the anchors.
ASPECT_RATIOS = [0.5, 1.0, 2.0]
#ASPECT_RATIOS = [1.0]

# ===========================
# Model Parameters
# ===========================

NUM_CLASSES = 10  # RUOD dataset has 10 classes

STRIDES = [4, 8, 16, 32, 64]

#STRIDES = [64]

# ===========================
# Debug switches (all off by default)
# ===========================

def _env_flag(name):
    return os.environ.get(name, "0").lower() in ("1", "true", "yes")

DETECT_ANOMALY = _env_flag("UOD_DETECT_ANOMALY")  # torch.autograd.set_detect_anomaly
WANDB_WATCH = _env_flag("UOD_WANDB_WATCH")        # wandb.watch(model, log="all")
# UOD_DEBUG=1 enables per-tensor printing and NaN checks in modules/Debug.py
