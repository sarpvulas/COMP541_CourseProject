import traceback

import torch
import wandb
from torch.optim.lr_scheduler import StepLR

from modules.config import DETECT_ANOMALY, IMAGE_SIZE, WANDB_WATCH
from modules.crop import build_batch
from modules.evaluation import evaluate_top1
from utils import save_model


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    best_val_accuracy,
):
    """
    Single-label classification training. Each image gets one label: the class whose boxes
    cover the largest area inside the center crop (see modules/crop.py). Evaluation uses the
    same rule. Cross-entropy is applied to the scale-averaged logits.

    Debug options (off by default): UOD_DETECT_ANOMALY=1, UOD_WANDB_WATCH=1.
    """
    if WANDB_WATCH:
        wandb.watch(model, log="all")
    lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    if DETECT_ANOMALY:
        torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        trained_batches = 0
        skipped_images = 0
        skipped_batches = 0
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        for batch_idx, (images, targets, filenames) in enumerate(train_loader):
            images_tensor, gt_labels, skipped = build_batch(images, targets, IMAGE_SIZE)
            skipped_images += skipped
            if images_tensor is None:
                skipped_batches += 1
                continue

            images_tensor = images_tensor.to(device)
            gt_labels = gt_labels.to(device)

            optimizer.zero_grad()
            try:
                cls_scores_list = model(images_tensor)
                tmp_targets = [{"labels": lbl} for lbl in gt_labels]
                loss = loss_fn(cls_scores_list, tmp_targets)
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                print(f"Error at batch {batch_idx}: {e}")
                traceback.print_exc()
                raise

            epoch_loss += loss.item()
            trained_batches += 1
            wandb.log({"batch_loss": loss.item()})

        avg_epoch_loss = epoch_loss / max(trained_batches, 1)
        print(
            f"[train_model] Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_epoch_loss:.4f} over "
            f"{trained_batches} batches; skipped {skipped_images} images "
            f"({skipped_batches} whole batches) with no usable box in the crop"
        )
        wandb.log({"epoch_loss": avg_epoch_loss, "skipped_images": skipped_images})
        lr_scheduler.step()

        print(f"Evaluating on test dataset after Epoch {epoch+1}...")
        try:
            metrics = evaluate_top1(model, test_loader, device, image_size=IMAGE_SIZE)
            wandb.log({"accuracy": metrics["accuracy"]})
            best_val_accuracy = save_model(
                model, epoch + 1, metrics["accuracy"], best_val_accuracy
            )
            print(f"[train_model] Updated best_val_accuracy => {best_val_accuracy:.4f}")
        except Exception as e:
            print("[train_model] Error during evaluation:", e)
            traceback.print_exc()

    return best_val_accuracy
