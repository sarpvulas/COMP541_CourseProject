import wandb
import torch
import torch.nn as nn
from utils import load_data
from modules.Pipeline_OnlyClassify import FullPipeline_OnlyClassify
from train import train_model
from modules.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    IMAGE_SIZE,
    NUM_CLASSES
)
from losses import single_label_classification_loss
import traceback



def main():
    wandb.init(project="uod_image_classification", config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE
    })
    print("Initializing the training pipeline...")
    torch.cuda.empty_cache()
    # Load training and testing datasets
    print("Loading datasets...")
    try:
        train_loader, test_loader = load_data(batch_size=BATCH_SIZE)
        print(f"Training loader prepared with {len(train_loader)} batches.")
        print(f"Testing loader prepared with {len(test_loader)} batches.")
    except Exception as e:
        print("Error loading datasets:")
        traceback.print_exc()
        return

    # Initialize the detection model
    print("Initializing the FullPipeline_OnlyClassify model...")
    try:
        model = FullPipeline_OnlyClassify(num_classes=NUM_CLASSES)
        print("Model initialized successfully.")
    except Exception as e:
        print("Error initializing the model:")
        traceback.print_exc()
        return

    # Move model to device
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "mps"
    print(f"Using device: {device}")
    model.to(device)

    # Optimizer
    print("Setting up the optimizer...")
    try:

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)


        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )

        print(f"Optimizer configured: {optimizer}")
    except Exception as e:
        print("Error setting up the optimizer:")
        traceback.print_exc()
        return

    best_val_accuracy = 0.0
    print("Commencing training...")
    try:
        best_val_accuracy = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=single_label_classification_loss,  # The revised one
            device=device,
            epochs=EPOCHS,
            best_val_accuracy=best_val_accuracy,
        )
    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()
    else:
        print("Training completed successfully.")

    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()
