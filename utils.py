import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from modules.config import (
    TRAIN_IMAGES_PATH,
    TRAIN_ANNOTATIONS_PATH,
    TEST_IMAGES_PATH,
    TEST_ANNOTATIONS_PATH,
    BATCH_SIZE,
    NUM_CLASSES,
    SEED,
    SUBSET_SIZE,
    VAL_FRACTION,
    VAL_SPLIT_SEED,
)
from modules.seed import make_generator

# -------------------------------------------------------------------
# 1) CocoDetectionWithFilename
# -------------------------------------------------------------------
class CocoDetectionWithFilename(CocoDetection):
    """
    Extends torchvision.datasets.CocoDetection to also return the image's filename.
    Filters out images whose width or height is < min_size.

    Class indices: COCO category ids are mapped to 0..K-1 by their sorted order, and each
    returned annotation gets a "class_index" key. For ids 1..K this equals category_id - 1.
    """

    def __init__(self, root, annFile, transform=None, min_size=512):
        super().__init__(root, annFile, transform=transform)
        self.min_size = min_size

        cat_ids = sorted(self.coco.getCatIds())
        if len(cat_ids) != NUM_CLASSES:
            raise ValueError(
                f"{annFile} has {len(cat_ids)} categories but NUM_CLASSES is {NUM_CLASSES}"
            )
        self.cat_to_index = {cid: i for i, cid in enumerate(cat_ids)}

        # Filter self.ids so only images >= min_size in both dimensions
        valid_ids = []
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            w, h = img_info['width'], img_info['height']
            if w >= self.min_size and h >= self.min_size:
                valid_ids.append(img_id)

        self.ids = valid_ids  # Keep only valid image IDs

    def __getitem__(self, index):
        # Original getitem: returns (PIL image, ann_list), plus our added filename
        img, ann_list = super().__getitem__(index)
        for a in ann_list:
            if a["category_id"] not in self.cat_to_index:
                raise ValueError(
                    f"annotation {a.get('id')} of image {self.ids[index]} has category_id "
                    f"{a['category_id']}, which is not in the categories list "
                    f"{sorted(self.cat_to_index)}")
        ann_list = [dict(a, class_index=self.cat_to_index[a["category_id"]]) for a in ann_list]

        # Retrieve the filename
        img_id = self.ids[index]
        file_info = self.coco.loadImgs(img_id)[0]
        filename = file_info["file_name"]

        # Convert PIL to tensor if no transform
        if self.transform is None:
            img = ToTensor()(img)

        return img, ann_list, filename



# -------------------------------------------------------------------
# 2) detection_collate_fn (no resizing)
# -------------------------------------------------------------------
def detection_collate_fn(batch):
    """
    Collate function that:
      - Takes items of form (img_tensor, ann_list, filename)
      - Builds a dict with keys "image_id", "boxes", "labels" for each sample
      - Returns (images, targets, filenames) for the batch

    NOTE: No resizing or bounding box scaling is done here.
          Your train_model can do resizing if desired.
    """
    images = []
    targets = []
    filenames = []

    for (img, ann_list, fname) in batch:
        # Build "boxes" and "labels" from the raw annotation list
        boxes = []
        labels = []
        image_id = -1  # default if no ann

        if len(ann_list) > 0:
            image_id = ann_list[0]["image_id"]

        for ann in ann_list:
            # Each ann has "bbox" = [x, y, w, h], "category_id", etc.
            x, y, w, h = ann["bbox"]
            x2 = x + w
            y2 = y + h
            boxes.append([x, y, x2, y2])
            # 0-based class index from the dataset's sorted category ids
            labels.append(ann["class_index"])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target_dict = {
            "image_id": image_id,
            "boxes": boxes,
            "labels": labels
        }

        images.append(img)
        targets.append(target_dict)
        filenames.append(fname)

    return images, targets, filenames


# -------------------------------------------------------------------
# 3) load_data
# -------------------------------------------------------------------
def split_train_val(n, val_fraction=VAL_FRACTION, split_seed=VAL_SPLIT_SEED):
    """
    Split range(n) into (train_indices, val_indices) with a fixed-seed permutation.
    The validation part has round(n * val_fraction) items (at least 1 when n >= 2).
    """
    n_val = min(max(1, round(n * val_fraction)), n - 1) if n >= 2 else 0
    perm = torch.randperm(n, generator=make_generator(split_seed)).tolist()
    return perm[n_val:], perm[:n_val]


def _subset(dataset, size, generator):
    """A random subset of at most `size` items, drawn with `generator`."""
    idx = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, idx)


def load_data(batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE, use_subsets=True, seed=SEED,
              val_fraction=VAL_FRACTION):
    """
    Build (train_loader, val_loader, test_loader).

    The validation split is carved out of the training annotations with a fixed seed
    (VAL_SPLIT_SEED, val_fraction of the images), so it never overlaps the training part
    and is the same for every training seed. The test split is the dataset's own test
    split. With use_subsets, at most `subset_size` images are drawn from each of the three
    parts using `seed`. The training loader shuffles with a generator seeded from `seed`,
    so data order is repeatable. num_workers is 0, so no worker seeding is needed.
    """
    train_full = CocoDetectionWithFilename(
        root=TRAIN_IMAGES_PATH,
        annFile=TRAIN_ANNOTATIONS_PATH,
        transform=None,
        min_size=512  # <--- filter out images smaller than 512
    )

    # Images are cropped later by modules/crop.py, together with their boxes,
    # so that train and eval derive the label from the same crop.
    test_dataset = CocoDetectionWithFilename(
        root=TEST_IMAGES_PATH,
        annFile=TEST_ANNOTATIONS_PATH,
        transform=None,
        min_size=512  # <--- same for test
    )

    train_idx, val_idx = split_train_val(len(train_full), val_fraction)
    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(train_full, val_idx)

    if use_subsets:
        g = make_generator(seed)
        train_dataset = _subset(train_dataset, subset_size, g)
        val_dataset = _subset(val_dataset, subset_size, g)
        test_dataset = _subset(test_dataset, subset_size, g)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
        num_workers=0,
        pin_memory=pin,
        collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        collate_fn=detection_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        collate_fn=detection_collate_fn
    )

    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------
# 4) save_model
# -------------------------------------------------------------------
def save_model(model, epoch, accuracy, best_accuracy, save_path="models/checkpoints/best_model_epoch_{}.pth"):
    """
    Save the model if the accuracy improves (a strictly higher value, so 0 never saves).

    Args:
        model:         The PyTorch model
        epoch:         Current epoch number
        accuracy:      Current top-1 accuracy on the validation split
        best_accuracy: The best accuracy so far
        save_path:     Path pattern for saving
                       (e.g., "models/checkpoints/best_model_epoch_{}.pth")
    Returns:
        Possibly updated best_accuracy
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if accuracy > best_accuracy:
        final_save_path = save_path.format(epoch)
        # Unwrap DataParallel so the keys carry no "module." prefix.
        torch.save(getattr(model, "module", model).state_dict(), final_save_path)
        print(f"Model saved at epoch {epoch} with accuracy: {accuracy:.4f}")
        return accuracy
    return best_accuracy
