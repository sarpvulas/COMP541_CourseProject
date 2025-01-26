import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
import torch

def visualize_annotations(images, targets, anchors, coco_gt, save_dir=None, original_sizes=None):
    """
    Visualize images with ground truth boxes, labels, and anchors.

    Args:
        images (list[torch.Tensor]): List of image tensors to visualize.
        targets (list[dict]): List of target dictionaries containing bounding boxes and labels.
        anchors (torch.Tensor): Tensor of anchors for the current image.
        coco_gt (COCO): COCO object for retrieving image metadata.
        save_dir (str, optional): Directory to save visualizations. If None, images are only displayed.
        original_sizes (list[tuple], optional): List of tuples (width, height) for the original image sizes.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for idx, (image, target) in enumerate(zip(images, targets)):
        # Determine the original size of the image
        if original_sizes:
            original_width, original_height = original_sizes[idx]
        else:
            _, original_height, original_width = image.shape

        # Convert the image to PIL format for display
        pil_image = to_pil_image(image.cpu())

        # Create a matplotlib figure with aspect ratio matching the image
        fig, ax = plt.subplots(1, figsize=(original_width / 100, original_height / 100))
        ax.imshow(pil_image)

        # Plot ground truth boxes and labels
        for box, label in zip(target["boxes"].cpu().numpy(), target["labels"].cpu().numpy()):
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, str(label), color='g', fontsize=10, weight='bold')

        # Adjust axis limits to match the image size
        ax.set_xlim(0, original_width)
        ax.set_ylim(original_height, 0)  # Invert the y-axis to match image coordinates

        # Get image filename from COCO
        image_id = target["image_id"]
        image_filename = coco_gt.imgs[image_id]["file_name"]

        # Set the title
        ax.set_title(image_filename, fontsize=12)

        # Save or display the visualization
        if save_dir:
            save_path = os.path.join(save_dir, f"{os.path.splitext(image_filename)[0]}_visualization.png")
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

        # Close the figure to free memory
        plt.close(fig)
