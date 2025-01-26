# anchor_utils.py

import torch
import math

def generate_anchors(fm_size, scales, aspect_ratios, stride):
    """
    Generate anchors for a single feature map.

    Args:
        fm_size (tuple): (height, width) of the feature map.
        scales (list): List of scales (in pixels).
        aspect_ratios (list): List of aspect ratios (width/height).
        stride (int): Stride of the feature map relative to the input image.

    Returns:
        anchors (Tensor): Shape (N, 4), where N is the number of anchors.
                          Each anchor is in [x1, y1, x2, y2] format.
    """
    fm_height, fm_width = fm_size

    # Handle empty feature map
    if fm_height == 0 or fm_width == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    # Validate aspect ratios
    for ratio in aspect_ratios:
        if ratio <= 0:
            raise ValueError(f"Aspect ratio must be positive, got {ratio}")

    # Generate grid centers
    device = torch.device('cpu')
    shift_x = (torch.arange(0, fm_width, device=device) + 0.5) * stride
    shift_y = (torch.arange(0, fm_height, device=device) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)

    centers = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  # (K,4)

    # Generate anchor boxes centered at (0,0)
    anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            w = scale * math.sqrt(ratio)
            h = scale / math.sqrt(ratio)
            anchor = torch.tensor([-w/2, -h/2, w/2, h/2], dtype=torch.float32)
            anchors.append(anchor)
    anchors = torch.stack(anchors, dim=0)  # (A,4)

    # Shift anchors to all grid centers
    K = centers.size(0)
    A = anchors.size(0)
    anchors = anchors.view(1, A, 4) + centers.view(K, 1, 4)
    anchors = anchors.view(-1, 4)  # (K*A,4)

    return anchors

def generate_all_anchors(feature_map_sizes, scales, aspect_ratios, strides, device):
    """
    Generate anchors for all feature maps.

    Args:
        feature_map_sizes (list of tuples): List of (height, width) for each feature map.
        scales (list): List of scales (in pixels).
        aspect_ratios (list): List of aspect ratios (width/height).
        strides (list): List of strides for each feature map.
        device (torch.device): Device to place the anchors on.

    Returns:
        all_anchors (Tensor): Shape (total_anchors, 4), all anchors across feature maps.
    """
    if not (len(feature_map_sizes) == len(strides)):
        raise ValueError("feature_map_sizes and strides must have the same length.")

    all_anchors = []
    for idx, (fm_size, stride) in enumerate(zip(feature_map_sizes, strides)):
        fm_h, fm_w = fm_size
        if fm_h == 0 or fm_w == 0:
            # Handle empty feature map
            anchors = torch.empty((0, 4), dtype=torch.float32, device=device)
        else:
            anchors = generate_anchors(fm_size, scales, aspect_ratios, stride)
            anchors = anchors.to(device)
        all_anchors.append(anchors)
    if all_anchors:
        all_anchors = torch.cat(all_anchors, dim=0)  # (total_anchors,4)
    else:
        all_anchors = torch.empty((0, 4), dtype=torch.float32, device=device)
    return all_anchors




def box_area(boxes):
    """
    Compute the area of a set of boxes in [x1, y1, x2, y2] format.

    Args:
        boxes (Tensor): Shape (N, 4).

    Returns:
        area (Tensor): Shape (N,), area of each box.
    """
    assert boxes.ndim == 2 and boxes.size(1) == 4, "Boxes must have shape (N, 4)"
    width = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area = width * height
    return area

def intersection_area(boxes1, boxes2):
    """
    Compute pairwise intersection area between two sets of boxes.

    Args:
        boxes1 (Tensor): Shape (A, 4).
        boxes2 (Tensor): Shape (B, 4).

    Returns:
        inter (Tensor): Shape (A, B), intersection area for all pairs.
    """
    A = boxes1.size(0)
    B = boxes2.size(0)

    if A == 0 or B == 0:
        return torch.zeros((A, B), dtype=boxes1.dtype, device=boxes1.device)

    # Expand coordinates for broadcast
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3].unsqueeze(0))

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    return inter

def box_iou(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of boxes in [x1, y1, x2, y2] format.

    Args:
        boxes1 (Tensor): Shape (A, 4).
        boxes2 (Tensor): Shape (B, 4).

    Returns:
        ious (Tensor): Shape (A, B) of IoU values.
    """
    area1 = box_area(boxes1)  # (A,)
    area2 = box_area(boxes2)  # (B,)

    inter = intersection_area(boxes1, boxes2)  # (A, B)
    union = area1[:, None] + area2.unsqueeze(0) - inter
    ious = inter / union.clamp(min=1e-6)
    return ious

def match_anchors(anchors, gt_boxes, iou_threshold_pos=0.4, iou_threshold_neg=0.3):
    """
    Match anchors to ground-truth boxes based on IoU thresholds.

    Args:
        anchors (Tensor): Shape (A, 4).
        gt_boxes (Tensor): Shape (G, 4).
        iou_threshold_pos (float): IoU above which an anchor is assigned as positive.
        iou_threshold_neg (float): IoU below which an anchor is assigned as negative.

    Returns:
        pos_inds (Tensor): Indices of anchors assigned as positive.
        neg_inds (Tensor): Indices of anchors assigned as negative.
        matched_gt_inds (Tensor): For each pos anchor, which GT index it matches.
    """
    device = anchors.device
    anchors = anchors.to(device)
    gt_boxes = gt_boxes.to(device)

    if gt_boxes.numel() == 0:
        # No ground truth boxes, all anchors are negative
        pos_inds = torch.tensor([], dtype=torch.long, device=device)
        neg_inds = torch.arange(0, anchors.size(0), dtype=torch.long, device=device)
        matched_gt_inds = torch.tensor([], dtype=torch.long, device=device)
        return pos_inds, neg_inds, matched_gt_inds

    iou_mat = box_iou(anchors, gt_boxes)  # (A, G)

    max_iou_vals, max_iou_idx = iou_mat.max(dim=1)  # (A,), (A,)

    pos_mask = max_iou_vals >= iou_threshold_pos
    neg_mask = max_iou_vals < iou_threshold_neg

    pos_inds = pos_mask.nonzero(as_tuple=True)[0]
    neg_inds = neg_mask.nonzero(as_tuple=True)[0]
    matched_gt_inds = max_iou_idx[pos_mask]

    return pos_inds, neg_inds, matched_gt_inds

def decode_boxes(bbox_preds, anchors, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
    """
    Decode bounding box deltas into [x1, y1, x2, y2] boxes.

    Args:
        bbox_preds (Tensor): Shape (A, 4), predicted deltas.
        anchors (Tensor): Shape (A, 4), anchor boxes.
        means (tuple): Means for bounding box regression.
        stds (tuple): Standard deviations for bounding box regression.

    Returns:
        decoded (Tensor): Shape (A, 4), decoded bounding boxes.
    """
    assert bbox_preds.size(0) == anchors.size(0), "bbox_preds and anchors must have the same number of elements."
    
    anchor_x1 = anchors[:, 0]
    anchor_y1 = anchors[:, 1]
    anchor_x2 = anchors[:, 2]
    anchor_y2 = anchors[:, 3]

    anchor_w = (anchor_x2 - anchor_x1).clamp(min=1e-6)
    anchor_h = (anchor_y2 - anchor_y1).clamp(min=1e-6)
    anchor_cx = anchor_x1 + 0.5 * anchor_w
    anchor_cy = anchor_y1 + 0.5 * anchor_h

    dx = bbox_preds[:, 0] * stds[0] + means[0]
    dy = bbox_preds[:, 1] * stds[1] + means[1]
    dw = bbox_preds[:, 2] * stds[2] + means[2]
    dh = bbox_preds[:, 3] * stds[3] + means[3]

    pred_cx = dx * anchor_w + anchor_cx
    pred_cy = dy * anchor_h + anchor_cy
    pred_w = anchor_w * torch.exp(dw)
    pred_h = anchor_h * torch.exp(dh)

    decoded_x1 = pred_cx - 0.5 * pred_w
    decoded_y1 = pred_cy - 0.5 * pred_h
    decoded_x2 = pred_cx + 0.5 * pred_w
    decoded_y2 = pred_cy + 0.5 * pred_h

    decoded = torch.stack([decoded_x1, decoded_y1, decoded_x2, decoded_y2], dim=-1)
    return decoded

