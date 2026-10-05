"""Center-crop and label selection shared by training and evaluation.

The classification label of an image is derived from the boxes that remain visible
in the center crop the model actually sees, so train and eval use the same rule.
"""
import torch


def crop_and_filter(image, target, image_size, min_visible=0.5):
    """
    Center-crop `image` (C, H, W) to `image_size` = (width, height) and clip the boxes.

    A box is kept only if at least `min_visible` of its area lies inside the crop.
    Boxes and labels are filtered with the same mask, so they stay aligned.

    Returns:
        (cropped_image, new_target) where new_target has "boxes", "labels" and
        "image_id", or (cropped_image, None) if no box survives.
    """
    _, height, width = image.shape
    crop_w, crop_h = image_size

    cx, cy = width // 2, height // 2
    x0 = max(cx - crop_w // 2, 0)
    x1 = min(cx + crop_w // 2, width)
    y0 = max(cy - crop_h // 2, 0)
    y1 = min(cy + crop_h // 2, height)

    cropped = image[:, y0:y1, x0:x1]

    boxes = target["boxes"].to(torch.float32)
    labels = target["labels"]
    if boxes.numel() == 0:
        return cropped, None

    clipped = torch.stack([
        boxes[:, 0].clamp(min=x0, max=x1),
        boxes[:, 1].clamp(min=y0, max=y1),
        boxes[:, 2].clamp(min=x0, max=x1),
        boxes[:, 3].clamp(min=y0, max=y1),
    ], dim=1)

    orig_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    clip_w = clipped[:, 2] - clipped[:, 0]
    clip_h = clipped[:, 3] - clipped[:, 1]
    clip_area = clip_w.clamp(min=0) * clip_h.clamp(min=0)

    keep = (orig_area > 0) & (clip_area / orig_area.clamp(min=1e-12) >= min_visible) & (clip_w > 0) & (clip_h > 0)
    if not keep.any():
        return cropped, None

    shift = torch.tensor([x0, y0, x0, y0], dtype=torch.float32)
    new_target = {
        "boxes": clipped[keep] - shift,
        "labels": labels[keep],
        "image_id": target.get("image_id", -1),
    }
    return cropped, new_target


def largest_area_label(boxes, labels):
    """Return the class id whose boxes cover the largest total area, or None if no boxes."""
    if boxes.size(0) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_label, best_area = None, -1.0
    for label in labels.unique():
        total = areas[labels == label].sum().item()
        if total > best_area:
            best_area, best_label = total, label.item()
    return best_label


def crop_and_label(image, target, image_size):
    """
    Crop the image, filter the boxes, and pick the single label.

    Returns (cropped_image, label) or (cropped_image, None) if no usable box remains.
    """
    cropped, new_target = crop_and_filter(image, target, image_size)
    if new_target is None:
        return cropped, None
    return cropped, largest_area_label(new_target["boxes"], new_target["labels"])


def build_batch(images, targets, image_size):
    """
    Crop each image and derive its single label from the boxes visible in the crop.

    Returns (image_tensor or None, label_tensor or None, n_skipped). Images with no usable
    box in the crop are skipped and counted.
    """
    kept_images, labels, skipped = [], [], 0
    for img, tgt in zip(images, targets):
        cropped, label = crop_and_label(img, tgt, image_size)
        if label is None:
            skipped += 1
            continue
        kept_images.append(cropped)
        labels.append(label)
    if not kept_images:
        return None, None, skipped
    return torch.stack(kept_images, dim=0), torch.tensor(labels, dtype=torch.long), skipped
