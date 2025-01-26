import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from .anchor_utils import decode_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def denormalize_image(image, mean, std):
    """
    Denormalize an image tensor.

    Args:
        image (Tensor): Image tensor of shape (3, H, W).
        mean (list or ndarray): Mean values for each channel.
        std (list or ndarray): Standard deviation values for each channel.

    Returns:
        denormalized_image (ndarray): Denormalized image.
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return image


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

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

def calculate_coco_metrics(
    model,
    data_loader,
    coco_gt,
    device,
    anchors=None,
    image_size=(512, 512),
    visualize=True,
    num_visualizations=5,
    save_visualizations=True,
    save_dir="visualizations_CLS"
):
    """
    Single-Class (top-1) classification EVALUATION function.
    Uses the same "largest-area label" logic as training to unify multiple
    bounding-box labels into a single label for each test image.

    Steps:
      1) For each (images, targets, filenames) in data_loader:
         - Possibly unify images => [B, 3, H, W].
         - For each image i:
           a) unify its bounding boxes and labels => single label with largest box area
           b) skip if mismatch or no valid box
      2) Build a batch => pass to model => list of Tensors => each [B, num_classes, H_i, W_i]
      3) Spatial average each scale => [B, num_classes], average across scales => final => [B, num_classes]
      4) Argmax => predicted label => compare with single ground-truth largest-area label => accumulate correct/total
      5) Return placeholders (AP=0.0, etc.) + 'accuracy'

    Args:
        model: FullPipeline_OnlyClassify returning a list of Tensors [B, num_classes, H, W].
        data_loader: yields (images, targets, filenames).
            - images: [B, 3, H, W] or list of [C, H, W].
            - targets: each has 'boxes', 'labels' => shape mismatch is possible, we handle that.
        device: 'cuda' or 'cpu'.
        anchors: leftover from detection code, not used here.
        image_size: not used for cropping (assuming data loader already does center-crop).
        visualize, num_visualizations, save_visualizations, save_dir:
            optional logic for visualizing results.

    Returns:
        metrics (dict):
          {
            "AP":   0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APs":  0.0,
            "APm":  0.0,
            "APl":  0.0,
            "accuracy": top_1_accuracy
          }
    """

    # Helper: pick largest-area label
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

    model.eval()
    if save_visualizations and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    total_correct = 0
    total_samples = 0
    visualized_count = 0

    with torch.no_grad():
        for batch_idx, (images, targets, filenames) in enumerate(data_loader):
            # Possibly unify images => shape [B,3,H,W]
            if isinstance(images, list):
                images = torch.stack(images, dim=0)
            images = images.to(device)

            # 1) For each image => unify bounding boxes => pick largest-area label
            batch_boxes_labels = []
            valid_indices = []
            for i, tgt in enumerate(targets):
                boxes = tgt["boxes"]      # shape [N,4]
                label_ids = tgt["labels"] # shape [N]

                # Mismatch check
                if boxes.size(0) != label_ids.size(0):
                    print(f"[Eval] Mismatch: boxes={boxes.size(0)}, labels={label_ids.size(0)} => skip image.")
                    continue

                single_label = pick_largest_area_label(boxes, label_ids)
                if single_label is not None:
                    batch_boxes_labels.append(single_label)
                    valid_indices.append(i)
                else:
                    print(f"[Eval] No largest-area label => skip image index {i}.")

            if len(valid_indices) == 0:
                print(f"[Eval] Entire batch {batch_idx} had no valid images => skipping.")
                continue

            # 2) Filter images to only valid indices => shape [B_valid, 3, H, W]
            #    because we skip some images if mismatch or no largest label
            images_filtered = images[valid_indices]  # shape => [B_valid, 3, H, W]
            final_labels = torch.tensor(batch_boxes_labels, dtype=torch.long, device=device)

            # 3) Forward => model => list of scale Tensors
            cls_scores_list = model(images_filtered)

            # 4) Spatial average each scale => [B_valid, num_classes]
            per_scale_logits = []
            for scale_out in cls_scores_list:
                scale_avg = scale_out.mean(dim=[2, 3])  # => [B_valid, num_classes]
                per_scale_logits.append(scale_avg)
            final_logits = torch.stack(per_scale_logits, dim=0).mean(dim=0)  # => [B_valid, num_classes]

            # 5) Argmax => predicted label => shape [B_valid]
            preds = final_logits.argmax(dim=1)

            # 6) Compare => accumulate correct / total
            b_valid = final_labels.size(0)
            for idx in range(b_valid):
                # single GT label: final_labels[idx]
                # single pred label: preds[idx]
                if preds[idx].item() == final_labels[idx].item():
                    total_correct += 1
                total_samples += 1

                # Optional visualization
                if visualize and visualized_count < num_visualizations:
                    # Attempt to show image from filenames
                    # We have the valid_indices => map back
                    original_idx = valid_indices[idx]
                    if original_idx < len(filenames):
                        img_path = filenames[original_idx]
                    else:
                        img_path = None

                    if img_path and os.path.exists(img_path):
                        try:
                            pil_img = Image.open(img_path).convert("RGB")
                        except Exception:
                            pil_img = None
                    else:
                        pil_img = None

                    if pil_img:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(6,6))
                        plt.imshow(pil_img)
                        plt.axis('off')
                        plt.title(f"Pred: {preds[idx].item()} | GT: {final_labels[idx].item()}")
                        if save_visualizations:
                            out_path = os.path.join(save_dir,f"eval_batch{batch_idx}_idx{original_idx}.png")
                            plt.savefig(out_path,bbox_inches="tight")
                            print(f"[DEBUG] Saved => {out_path}")
                        plt.close()

                    visualized_count += 1

    # Final accuracy
    accuracy = total_correct / total_samples if total_samples>0 else 0.0
    print(f"[calculate_coco_metrics] Single-class top-1 accuracy: {accuracy:.4f}  ({total_correct}/{total_samples})")

    # Return placeholders for detection metrics plus accuracy
    return {
        "AP":   0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "APs":  0.0,
        "APm":  0.0,
        "APl":  0.0,
        "accuracy": accuracy
    }
