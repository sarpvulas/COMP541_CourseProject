import torch

from modules.crop import crop_and_label


def evaluate_top1(model, data_loader, device, image_size=(512, 512)):
    """
    Top-1 accuracy of the single-label classifier.

    The ground-truth label of each image is the class with the largest total box area inside
    the center crop (modules/crop.py), the same rule used in training. Images with no usable
    box in the crop are skipped and counted.

    Args:
        model: FullPipeline_OnlyClassify, returns a list of [B, num_classes, H, W] tensors.
        data_loader: yields (images, targets, filenames) with uncropped image tensors.
        device: torch device.
        image_size: (width, height) of the center crop.

    Returns:
        dict with "accuracy", "correct", "total" and "skipped".
    """
    model.eval()
    correct = total = skipped = 0

    with torch.no_grad():
        for images, targets, _ in data_loader:
            crops, labels = [], []
            for img, tgt in zip(images, targets):
                cropped, label = crop_and_label(img, tgt, image_size)
                if label is None:
                    skipped += 1
                    continue
                crops.append(cropped)
                labels.append(label)
            if not crops:
                continue

            batch = torch.stack(crops, dim=0).to(device)
            gt = torch.tensor(labels, dtype=torch.long, device=device)

            scale_logits = [out.mean(dim=[2, 3]) for out in model(batch)]
            preds = torch.stack(scale_logits, dim=0).mean(dim=0).argmax(dim=1)

            correct += (preds == gt).sum().item()
            total += gt.numel()

    accuracy = correct / total if total > 0 else 0.0
    print(f"[evaluate_top1] top-1 accuracy: {accuracy:.4f} ({correct}/{total}), skipped {skipped} images")
    return {"accuracy": accuracy, "correct": correct, "total": total, "skipped": skipped}
