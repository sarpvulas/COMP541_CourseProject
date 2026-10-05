import traceback

import torch
import wandb
from torch.optim.lr_scheduler import StepLR

from modules.config import DETECT_ANOMALY, IMAGE_SIZE, WANDB_WATCH
from modules.crop import build_batch
from modules.evaluation import evaluate_top1
from utils import save_model


CHECKPOINT_PATTERN = "models/checkpoints/best_model_epoch_{}.pth"


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    best_val_accuracy=0.0,
    checkpoint_pattern=CHECKPOINT_PATTERN,
):
    """
    Single-label classification training. Each image gets one label: the class whose boxes
    cover the largest area inside the center crop (see modules/crop.py). Evaluation uses the
    same rule. Cross-entropy is applied to the scale-averaged logits.

    Checkpoints are selected on `val_loader` (a split carved from the training set). After
    the last epoch the best checkpoint is reloaded and `test_loader` is evaluated once; the
    test split is never used to choose anything. If no epoch improved on
    `best_val_accuracy` (so nothing was saved), the final-epoch weights are tested instead.

    Returns a dict: "best_val_accuracy", "best_epoch" (None if nothing was saved),
    "test" (the evaluate_top1 result on the test split).

    Debug options (off by default): UOD_DETECT_ANOMALY=1, UOD_WANDB_WATCH=1.
    """
    if WANDB_WATCH:
        wandb.watch(model, log="all")
    best_epoch = None
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

        print(f"Evaluating on validation split after Epoch {epoch+1}...")
        metrics = evaluate_top1(model, val_loader, device, image_size=IMAGE_SIZE)
        wandb.log({"val_accuracy": metrics["accuracy"]})
        previous_best = best_val_accuracy
        best_val_accuracy = save_model(
            model, epoch + 1, metrics["accuracy"], best_val_accuracy, checkpoint_pattern
        )
        if best_val_accuracy > previous_best:
            best_epoch = epoch + 1
        print(f"[train_model] best_val_accuracy => {best_val_accuracy:.4f}")

    # Test split: once, on the checkpoint chosen by validation accuracy.
    if best_epoch is not None:
        state = torch.load(checkpoint_pattern.format(best_epoch), map_location=device)
        getattr(model, "module", model).load_state_dict(state)
        print(f"Final test evaluation with the epoch-{best_epoch} checkpoint (best validation)")
    else:
        print("No checkpoint was saved (validation accuracy never improved); "
              "final test evaluation uses the last-epoch weights")
    test_metrics = evaluate_top1(model, test_loader, device, image_size=IMAGE_SIZE)
    wandb.log({"test_accuracy": test_metrics["accuracy"]})

    return {"best_val_accuracy": best_val_accuracy, "best_epoch": best_epoch, "test": test_metrics}
