import argparse

import wandb
import torch
import torch.nn as nn
from utils import load_data
from modules.Pipeline_OnlyClassify import FullPipeline_OnlyClassify
from modules.seed import set_seed
from train import train_model
from modules.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    IMAGE_SIZE,
    NUM_CLASSES,
    SEED,
    VAL_FRACTION,
    VAL_SPLIT_SEED,
)
from losses import single_label_classification_loss


def main(seed=None):
    """Train on the training part, select checkpoints on validation, test once at the end."""
    seed = SEED if seed is None else seed
    set_seed(seed)  # before the model is built, so weight initialisation is seeded too
    wandb.init(project="uod_image_classification", config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
        "seed": seed,
        "val_fraction": VAL_FRACTION,
        "val_split_seed": VAL_SPLIT_SEED,
    })
    print(f"Initializing the training pipeline (seed {seed})...")
    print("Loading datasets...")
    train_loader, val_loader, test_loader = load_data(batch_size=BATCH_SIZE, seed=seed)
    print(f"Train: {len(train_loader)} batches, validation: {len(val_loader)}, "
          f"test: {len(test_loader)}")

    print("Initializing the FullPipeline_OnlyClassify model...")
    model = FullPipeline_OnlyClassify(num_classes=NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    print(f"Optimizer configured: {optimizer}")

    print("Commencing training...")
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=single_label_classification_loss,
        device=device,
        epochs=EPOCHS,
    )
    test = result["test"]
    print("Training completed successfully.")
    print(f"Best validation accuracy: {result['best_val_accuracy']:.4f} "
          f"(epoch {result['best_epoch']})")
    print(f"Test accuracy (evaluated once, on the selected weights): {test['accuracy']:.4f} "
          f"({test['correct']}/{test['total']})")
    wandb.finish()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: UOD_SEED, else 0)")
    main(seed=parser.parse_args().seed)
