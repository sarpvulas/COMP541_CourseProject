import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import os
from modules.anchor_utils import decode_boxes
from torchmetrics.functional.detection.giou import generalized_intersection_over_union
import torch.nn.functional as F

import matplotlib.patches as patches


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)
    return iou

def match_anchors(anchors, gt_boxes, iou_threshold_pos=0.5, iou_threshold_neg=0.4):
    device = anchors.device
    gt_boxes = gt_boxes.to(device)

    if gt_boxes.numel() == 0:
        pos_inds = torch.tensor([], dtype=torch.long, device=device)
        neg_inds = torch.arange(anchors.size(0), dtype=torch.long, device=device)
        matched_gt_inds = torch.tensor([], dtype=torch.long, device=device)
        return pos_inds, neg_inds, matched_gt_inds

    iou_mat = box_iou(anchors, gt_boxes)
    max_iou_vals, max_iou_idx = iou_mat.max(dim=1)

    pos_mask = max_iou_vals >= iou_threshold_pos
    neg_mask = max_iou_vals < iou_threshold_neg

    pos_inds = pos_mask.nonzero(as_tuple=True)[0]
    neg_inds = neg_mask.nonzero(as_tuple=True)[0]
    matched_gt_inds = max_iou_idx[pos_mask]

    return pos_inds, neg_inds, matched_gt_inds


def visualize_and_save_debug_image(image, gt_boxes, anchors, output_dir, batch_idx, img_name="debug_image"):
    """
    Visualize and save debug images showing anchors and ground truth boxes.

    Args:
        image (Tensor): Tensor of shape (C, H, W), typically in range [0, 1].
        gt_boxes (Tensor): Tensor of shape (N, 4) containing ground truth boxes.
        anchors (Tensor): Tensor of shape (M, 4) containing anchors.
        output_dir (str): Directory to save the debug images.
        batch_idx (int): Current batch index.
        img_name (str): Optional image name or identifier for saving files.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DEBUG] Created debug directory: {output_dir}")
    else:
        print(f"[DEBUG] Debug directory exists: {output_dir}")

    try:
        # Convert image tensor to numpy
        if image.ndim != 3 or image.shape[0] != 3:
            print(f"[Batch {batch_idx}] Invalid image tensor shape: {image.shape}. Skipping plot.")
            return
        image_np = image.permute(1, 2, 0).cpu().numpy()

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image_np)
        ax.set_title(f"Batch {batch_idx}, {img_name}")

        # Plot ground truth boxes (green)
        for box in gt_boxes:
            x1, y1, x2, y2 = box.cpu().numpy()
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor="green", facecolor="none"
            )
            ax.add_patch(rect)

        # Plot anchors (blue, limit to 100 for readability)
        for box in anchors[:100]:
            x1, y1, x2, y2 = box.cpu().numpy()
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=1, edgecolor="blue", facecolor="none", alpha=0.3
            )
            ax.add_patch(rect)

        # Save plot
        output_path = os.path.join(output_dir, f"batch_{batch_idx}_{img_name}.png")
        plt.savefig(output_path)
        print(f"[DEBUG] Debug plot saved at: {output_path}")

        # Show the plot inline
        plt.show()

    except Exception as e:
        print(f"[Batch {batch_idx}] Failed to create or save debug plot: {e}")
    finally:
        # Close the figure to free memory
        plt.close(fig)


def detection_loss(cls_scores, bbox_preds, targets, anchors,
                   iou_threshold_pos=0.5, iou_threshold_neg=0.4):
    batch_size = cls_scores.size(0)

    if len(targets) != batch_size:
        print(f"[detection_loss] Warning: Batch size mismatch - Predictions ({batch_size}) vs Targets ({len(targets)}).")
        batch_size = min(batch_size, len(targets))
        cls_scores = cls_scores[:batch_size]
        bbox_preds = bbox_preds[:batch_size]

    total_cls_loss = torch.tensor(0.0, device=cls_scores.device)
    total_reg_loss = torch.tensor(0.0, device=cls_scores.device)

    cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    for b in range(batch_size):
        print(f"[Batch {b + 1}/{batch_size}] Processing")
        cls_scores_b = torch.clamp(cls_scores[b], min=-1e6, max=1e6)
        bbox_preds_b = bbox_preds[b]

        gt_boxes_b = targets[b].get('boxes', torch.empty(0, 4, device=cls_scores.device))
        gt_labels_b = targets[b].get('labels', torch.empty(0, device=cls_scores.device))

        if gt_boxes_b.numel() == 0 or gt_labels_b.numel() == 0:
            print(f"[Batch {b + 1}] No ground truth, skipping.")
            continue

        gt_labels_b = gt_labels_b.to(cls_scores.device)
        gt_boxes_b = gt_boxes_b.to(cls_scores.device)

        pos_inds, neg_inds, matched_gt_inds = match_anchors(
            anchors, gt_boxes_b, iou_threshold_pos, iou_threshold_neg
        )

        print(f"[Batch {b + 1}] Positive anchors: {len(pos_inds)}, Negative anchors: {len(neg_inds)}")

        if len(pos_inds) == 0:
            print(f"[Batch {b + 1}] No positive anchors, saving and visualizing debug plot.")
            img_tensor = targets[b].get("image", None)
            if img_tensor is not None:
                img_name = targets[b].get("filename", f"image_{b}")
                visualize_and_save_debug_image(
                    img_tensor,
                    gt_boxes_b,
                    anchors,
                    output_dir="./debug_plots",
                    batch_idx=b,
                    img_name=img_name,
                )
            else:
                print(f"[Batch {b + 1}] No image tensor available for debug plotting.")

        target_labels = torch.zeros(cls_scores_b.size(0), dtype=torch.long, device=cls_scores.device)
        if pos_inds.numel() > 0:
            matched_labels = gt_labels_b[matched_gt_inds]
            target_labels[pos_inds] = matched_labels

        cls_loss = cls_loss_fn(cls_scores_b, target_labels)
        print(f"[Batch {b + 1}] Classification loss: {cls_loss.item()}")

        if pos_inds.numel() > 0:
            reg_targets = gt_boxes_b[matched_gt_inds]
            reg_preds = bbox_preds_b[pos_inds]
            decoded_boxes = decode_boxes(reg_preds, anchors[pos_inds])

            valid_mask = (decoded_boxes[:, 2] > decoded_boxes[:, 0]) & (decoded_boxes[:, 3] > decoded_boxes[:, 1])
            if valid_mask.any():
                decoded_boxes = decoded_boxes[valid_mask]
                reg_targets = reg_targets[valid_mask]

                print(f"[Batch {b + 1}] Decoded boxes: {decoded_boxes.shape}, Regression targets: {reg_targets.shape}")

                giou = generalized_intersection_over_union(decoded_boxes, reg_targets)
                reg_loss = (1 - giou).mean()
                print(f"[Batch {b + 1}] Regression loss: {reg_loss.item()}")
            else:
                reg_loss = torch.tensor(0.0, device=cls_scores.device)
                print(f"[Batch {b + 1}] No valid regression targets, skipping.")
        else:
            reg_loss = torch.tensor(0.0, device=cls_scores.device)
            print(f"[Batch {b + 1}] No positive anchors, skipping regression loss computation.")

        total_cls_loss += cls_loss
        total_reg_loss += reg_loss

        print(f"[Batch {b + 1}] Accumulated cls loss: {total_cls_loss.item()}, reg loss: {total_reg_loss.item()}")

    print(f"Final cls loss: {total_cls_loss.item()}, reg loss: {total_reg_loss.item()}")
    return total_cls_loss, total_reg_loss

import torch.nn.functional as F
import torch
import torch.nn as nn

def single_label_classification_loss(cls_scores, targets):
    # 1) Gather the scale-averaged logits
    per_scale_logits = []
    for scale_out in cls_scores:
        # shape => [B, num_classes, H, W]
        # Spatial average => [B, num_classes]
        scale_avg = scale_out.mean(dim=[2, 3])
        per_scale_logits.append(scale_avg)

    # 2) Average across scales => final => [B, num_classes]
    final_logits = torch.stack(per_scale_logits, dim=0).mean(dim=0)

    # 3) Build ground-truth label vector => shape [B]
    batch_size = final_logits.size(0)
    labels_list = []
    for b in range(batch_size):
        # Each targets[b]["labels"] is a single-label or shape-[1] tensor
        label_tensor = targets[b]["labels"]
        # Squeeze if shape is [1]
        if label_tensor.dim() > 0 and label_tensor.numel() == 1:
            label_tensor = label_tensor.squeeze()
        labels_list.append(label_tensor.long())
    gt_labels = torch.stack(labels_list, dim=0)  # => [B]

    # 4) Cross-entropy
    ce_loss_fn = nn.CrossEntropyLoss()
    loss = ce_loss_fn(final_logits, gt_labels)

    return loss