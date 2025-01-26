import torch
import traceback
import wandb
from modules.visualization import visualize_annotations
from pycocotools.coco import COCO
from utils import save_model
from modules.evaluation import calculate_coco_metrics
from modules.config import NUM_CLASSES, IMAGE_SIZE
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import CenterCrop

def resize_and_align_boxes(image, target, coco_gt, image_size, device):
    """
    Resize (center-crop) the image to `image_size` and adjust GT boxes accordingly.
    Then zero-pad if the crop is smaller than image_size in any dimension.
    """
    _, original_height, original_width = image.shape
    crop_width, crop_height = image_size

    center_x = original_width // 2
    center_y = original_height // 2

    # Compute crop coordinates
    crop_x_min = max(center_x - crop_width // 2, 0)
    crop_x_max = min(center_x + crop_width // 2, original_width)
    crop_y_min = max(center_y - crop_height // 2, 0)
    crop_y_max = min(center_y + crop_height // 2, original_height)

    # Crop the image
    cropped_image = image[:, crop_y_min:crop_y_max, crop_x_min:crop_x_max].to(device)

    # Adjust bounding boxes
    adjusted_boxes = []
    for box in target["boxes"]:
        x_min, y_min, x_max, y_max = box

        # Compute overlap with the crop
        cropped_x_min = max(x_min, crop_x_min)
        cropped_y_min = max(y_min, crop_y_min)
        cropped_x_max = min(x_max, crop_x_max)
        cropped_y_max = min(y_max, crop_y_max)

        # Check the intersection area
        cropped_width = max(cropped_x_max - cropped_x_min, 0)
        cropped_height = max(cropped_y_max - cropped_y_min, 0)
        cropped_area = cropped_width * cropped_height

        # Compute the original box area
        original_width = max(x_max - x_min, 0)
        original_height = max(y_max - y_min, 0)
        original_area = original_width * original_height

        # Keep the box only if at least 50% of the area is inside the crop
        if cropped_area / original_area >= 0.5:
            new_x_min = cropped_x_min - crop_x_min
            new_y_min = cropped_y_min - crop_y_min
            new_x_max = cropped_x_max - crop_x_min
            new_y_max = cropped_y_max - crop_y_min

            # Ensure the box has a valid size
            if new_x_max > new_x_min and new_y_max > new_y_min:
                adjusted_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

    # If no valid boxes remain, return None for the target
    if len(adjusted_boxes) == 0:
        print(f"Warning: No valid targets after cropping for image_id {target['image_id']}.")
        return cropped_image, None

    # Update the target with adjusted boxes
    target["boxes"] = torch.tensor(adjusted_boxes, dtype=torch.float32, device=device)

    return cropped_image, target



import torch
import traceback
import wandb
from torch.optim.lr_scheduler import StepLR
from modules.visualization import visualize_annotations
from modules.evaluation import calculate_coco_metrics
from modules.config import IMAGE_SIZE, NUM_CLASSES
from torchvision.transforms import CenterCrop
import torch.nn.functional as F

def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_fn,  # We'll pass single_label_classification_loss or a custom function
    device,
    epochs,
    best_val_accuracy,
    coco_gt,
    coco_test_gt,
    anchors=None
):
    """
    Single-class classification training:
      - Each image can have multiple boxes with different labels => pick the label
        that has the largest total bounding box area in the image.
      - Then do standard cross-entropy with final [B, num_classes].
    """

    if anchors is None:
        raise ValueError("Anchors must be provided and cannot be None.")

    wandb.watch(model, log="all")
    lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    torch.autograd.set_detect_anomaly(True)

    def pick_largest_area_label(boxes, labels):
        """
        boxes: [N, 4] for all bounding boxes in the image
        labels: [N] each is the label ID for that box
        Return single int label that has the largest sum of box areas.
        """
        if boxes.size(0) == 0:
            return None  # or skip

        # w*h => area
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h  # shape [N]

        unique_labels = labels.unique()
        best_area = -1.0
        best_label = None
        for ulbl in unique_labels:
            mask = (labels == ulbl)  # shape [N], True for boxes with that label
            combined_area = areas[mask].sum().item()
            if combined_area > best_area:
                best_area = combined_area
                best_label = ulbl.item()

        return best_label

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        for batch_idx, (images, targets, filenames) in enumerate(train_loader):
            print(f"\nProcessing Batch {batch_idx + 1}/{len(train_loader)}")

            if batch_idx == 0 and epoch == 0:
                # optionally visualize
                pass

            processed_images = []
            final_labels = []

            for img, tgt in zip(images, targets):
                c_img, c_tgt = resize_and_align_boxes(
                    img, tgt, coco_gt, IMAGE_SIZE, device
                )
                if c_tgt is not None:
                    boxes = c_tgt["boxes"]      # shape [N,4]
                    label_ids = c_tgt["labels"] # shape [M]

                    # ---- MISMATCH CHECK HERE ----
                    if boxes.size(0) != label_ids.size(0):
                        # If we have mismatch => skip the image
                        print(f"[Warning] Mismatch: boxes={boxes.size(0)}, labels={label_ids.size(0)} => skip image.")
                        continue

                    single_label = pick_largest_area_label(boxes, label_ids)
                    if single_label is not None:
                        processed_images.append(c_img)
                        final_labels.append(single_label)
                    else:
                        # no label => skip
                        print("[Warning] No largest-area label => skip image.")
                else:
                    # c_tgt is None => no valid boxes
                    pass

            if len(processed_images) == 0:
                print(f"[train_model] No valid images => skip batch.")
                continue

            images_tensor = torch.stack(processed_images, dim=0).to(device)
            gt_labels = torch.tensor(final_labels, dtype=torch.long, device=device)
            # shape => [B]

            optimizer.zero_grad()

            try:
                cls_scores_list = model(images_tensor)
                # 4) We pass (cls_scores_list, targets) to your single_label_classification_loss,
                # but we must build a new 'targets_for_loss' that has each
                # item = {"labels": single_label} in a shape [1] or []
                # We'll do something like:
                tmp_targets = []
                for lbl in final_labels:
                    tmp_targets.append({"labels": torch.tensor(lbl, device=device)})

                loss = loss_fn(cls_scores_list, tmp_targets)
                # e.g. single_label_classification_loss

                loss.backward()
                optimizer.step()

            except RuntimeError as e:
                print(f"Error at batch {batch_idx}: {e}")
                traceback.print_exc()
                raise e

            epoch_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"[train_model] Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_loss": avg_epoch_loss})
        lr_scheduler.step()

        # Evaluate
        print(f"Evaluating on test dataset after Epoch {epoch+1}...")
        try:
            metrics = calculate_coco_metrics(
                model, test_loader, coco_test_gt, device, anchors=anchors,
                image_size=IMAGE_SIZE, visualize=True, save_visualizations=True
            )
            wandb.log({
                "AP": metrics["AP"],
                "AP50": metrics["AP50"],
                "accuracy": metrics["accuracy"]
            })
            best_val_accuracy = save_model(
                model, epoch+1, metrics.get("accuracy",0.0), best_val_accuracy
            )
            print(f"[train_model] Updated best_val_accuracy => {best_val_accuracy:.4f}")
        except Exception as e:
            print("[train_model] Error during evaluation:", e)
            traceback.print_exc()

    return best_val_accuracy



import torch.nn.functional as F

def resize_and_concatenate(cls_scores):
    """
    Resizes all tensors in cls_scores to the largest spatial resolution and concatenates along dim=1.

    Args:
        cls_scores (list[Tensor]): List of tensors with shapes [B, C, H, W].

    Returns:
        Tensor: Concatenated tensor with uniform spatial dimensions.
    """
    # Find the largest spatial resolution
    max_height = max(score.shape[2] for score in cls_scores)
    max_width = max(score.shape[3] for score in cls_scores)

    # Resize all tensors to the largest resolution
    resized_scores = [
        F.interpolate(score, size=(max_height, max_width), mode="bilinear", align_corners=False)
        for score in cls_scores
    ]

    # Concatenate along dim=1 (class dimension)
    return torch.cat(resized_scores, dim=1)
