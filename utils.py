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
    IMAGE_SIZE,
)
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import CenterCrop

# -------------------------------------------------------------------
# 1) CocoDetectionWithFilename
# -------------------------------------------------------------------
class CocoDetectionWithFilename(CocoDetection):
    """
    Extends torchvision.datasets.CocoDetection to also return the image's filename.
    Filters out images whose width or height is < min_size.
    """

    def __init__(self, root, annFile, transform=None, min_size=512):
        super().__init__(root, annFile, transform=transform)
        self.min_size = min_size

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
            # Convert COCO 1-based category to 0-based if needed:
            cat_id = ann["category_id"] - 1
            labels.append(cat_id)

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
def load_data(batch_size=BATCH_SIZE, subset_size=500, use_subsets=True):
    train_dataset = CocoDetectionWithFilename(
        root=TRAIN_IMAGES_PATH,
        annFile=TRAIN_ANNOTATIONS_PATH,
        transform=None,
        min_size=512  # <--- filter out images smaller than 512
    )

    transform = Compose([
        CenterCrop(IMAGE_SIZE),
        ToTensor()
    ])
    test_dataset = CocoDetectionWithFilename(
        root=TEST_IMAGES_PATH,
        annFile=TEST_ANNOTATIONS_PATH,
        transform=transform,
        min_size=512  # <--- same for test
    )

    # Subset logic if use_subsets is True
    if use_subsets:
        subset_indices_train = torch.randperm(len(train_dataset))[:subset_size].tolist()  # Convert to list of integers
        train_dataset = Subset(train_dataset, subset_indices_train)

        subset_indices_test = torch.randperm(len(test_dataset))[:subset_size].tolist()  # Convert to list of integers
        test_dataset = Subset(test_dataset, subset_indices_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )

    return train_loader, test_loader


# -------------------------------------------------------------------
# 4) save_model
# -------------------------------------------------------------------
def save_model(model, epoch, accuracy, best_accuracy, save_path="models/checkpoints/best_model_epoch_{}.pth"):
    """
    Save the model if validation/test accuracy improves.

    Args:
        model:         The PyTorch model
        epoch:         Current epoch number
        accuracy:      Current val/test AP
        best_accuracy: The best AP so far
        save_path:     Path pattern for saving
                       (e.g., "models/checkpoints/best_model_epoch_{}.pth")
    Returns:
        Possibly updated best_accuracy
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if accuracy > best_accuracy or accuracy == 0:
        final_save_path = save_path.format(epoch)
        torch.save(model.state_dict(), final_save_path)
        print(f"Model saved at epoch {epoch} with accuracy: {accuracy:.4f}")
        return accuracy
    return best_accuracy
