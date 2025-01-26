import wandb
import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from utils import load_data
from modules.anchor_utils import generate_all_anchors
from modules.Pipeline_OnlyClassify import FullPipeline_OnlyClassify
from train import train_model
from pycocotools.coco import COCO
from modules.config import (
    TRAIN_IMAGES_PATH,
    TEST_IMAGES_PATH,
    TRAIN_ANNOTATIONS_PATH,
    TEST_ANNOTATIONS_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    IMAGE_SIZE,
    FEATURE_MAP_SIZES,
    STRIDES,
    SCALES,
    ASPECT_RATIOS,
    NUM_CLASSES
)
from losses import single_label_classification_loss
import traceback
from torchvision.transforms import Compose, ToTensor



def main():
    wandb.init(project="uod_object_detection", config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE
    })
    print("Initializing the training pipeline...")
    torch.cuda.empty_cache()
    transform = Compose([
        CenterCrop(IMAGE_SIZE),
        ToTensor()
    ])
    print(f"Applied transformations: {transform}")

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

    # Initialize COCO ground truth for evaluation
    print("Loading COCO ground truth annotations...")
    try:
        coco_gt = COCO(TRAIN_ANNOTATIONS_PATH)
        print("COCO ground truth loaded successfully.")
        coco_test_gt = COCO(TEST_ANNOTATIONS_PATH)
    except Exception as e:
        print("Error loading COCO ground truth annotations:")
        traceback.print_exc()
        return

    # Generate anchors
    print("Generating anchors...")
    try:
        device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        anchors = generate_all_anchors(
            feature_map_sizes=FEATURE_MAP_SIZES,
            scales=SCALES,
            aspect_ratios=ASPECT_RATIOS,
            strides=STRIDES,
            device=device_
        )
        print(f"Total anchors generated: {anchors.shape[0]}")
        print(f"Anchors are on device: {anchors.device}")
    except Exception as e:
        print("Error generating anchors:")
        traceback.print_exc()
        return

    # Initialize the detection model
    print("Initializing the FullPipeline detection model...")
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
            coco_gt=coco_gt,
            coco_test_gt=coco_test_gt,
            anchors=anchors
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
