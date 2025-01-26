# test_anchor_utils.py
import math
import unittest
import torch

from modules.anchor_utils import (
    generate_anchors,
    generate_all_anchors,
    box_area,
    intersection_area,
    box_iou,
    match_anchors,
    decode_boxes,
    giou
)


class TestAnchorUtils(unittest.TestCase):
    def setUp(self):
        """
        Set up common variables for the tests.
        """
        self.device = torch.device('cpu')  # Use CPU for testing
        self.image_size = (512, 512)
        self.feature_map_sizes = [
            (128, 128),  # stride 4
            (64, 64),  # stride 8
            (32, 32),  # stride 16
            (16, 16),  # stride 32
            (8, 8)  # stride 64
        ]
        self.strides = [4, 8, 16, 32, 64]
        self.scales = [32, 64, 128]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.anchors = generate_all_anchors(
            feature_map_sizes=self.feature_map_sizes,
            scales=self.scales,
            aspect_ratios=self.aspect_ratios,
            strides=self.strides,
            device=self.device
        )

    def test_generate_anchors(self):
        """
        Test the generate_anchors function for a single feature map.
        """
        fm_size = (2, 2)  # Small feature map for easy verification
        scales = [32]
        aspect_ratios = [1.0]
        stride = 256  # Assuming image size 512x512

        expected_anchors = torch.tensor([
            [128 - 16, 128 - 16, 128 + 16, 128 + 16],  # Center at (128,128)
            [384 - 16, 128 - 16, 384 + 16, 128 + 16],  # Center at (384,128)
            [128 - 16, 384 - 16, 128 + 16, 384 + 16],  # Center at (128,384)
            [384 - 16, 384 - 16, 384 + 16, 384 + 16]  # Center at (384,384)
        ], dtype=torch.float32)

        generated_anchors = generate_anchors(fm_size, scales, aspect_ratios, stride)
        self.assertEqual(generated_anchors.shape, expected_anchors.shape)
        self.assertTrue(torch.allclose(generated_anchors, expected_anchors))

    def test_generate_all_anchors(self):
        """
        Test the generate_all_anchors function for multiple feature maps.
        """
        total_expected_anchors = 0
        for fm_size, stride in zip(self.feature_map_sizes, self.strides):
            total_expected_anchors += fm_size[0] * fm_size[1] * len(self.scales) * len(self.aspect_ratios)

        self.assertEqual(self.anchors.shape[0], total_expected_anchors)
        self.assertEqual(self.anchors.shape[1], 4)

    def test_box_area(self):
        """
        Test the box_area function.
        """
        boxes = torch.tensor([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [10, 10, 20, 20],
            [0, 0, 0, 0],  # Zero area
            [15, 15, 10, 10]  # Invalid box (x2 < x1, y2 < y1)
        ], dtype=torch.float32)

        expected_areas = torch.tensor([
            100.0,
            100.0,
            100.0,
            0.0,
            0.0  # Area should be clamped to 0
        ], dtype=torch.float32)

        computed_areas = box_area(boxes)
        self.assertTrue(torch.allclose(computed_areas, expected_areas))

    def test_intersection_area(self):
        """
        Test the intersection_area function.
        """
        boxes1 = torch.tensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20]
        ], dtype=torch.float32)

        boxes2 = torch.tensor([
            [5, 5, 15, 15],
            [0, 0, 10, 10]
        ], dtype=torch.float32)

        expected_inter = torch.tensor([
            [25.0, 100.0],  # Intersection of first box with both boxes
            [25.0, 0.0]  # Intersection of second box with both boxes
        ], dtype=torch.float32)

        computed_inter = intersection_area(boxes1, boxes2)
        self.assertTrue(torch.allclose(computed_inter, expected_inter))

    def test_box_iou(self):
        """
        Test the box_iou function.
        """
        boxes1 = torch.tensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20]
        ], dtype=torch.float32)

        boxes2 = torch.tensor([
            [5, 5, 15, 15],
            [0, 0, 10, 10]
        ], dtype=torch.float32)

        computed_iou = box_iou(boxes1, boxes2)
        expected_iou = torch.tensor([
            [25 / (100 + 100 - 25), 100 / (100 + 100 - 100)],  # [0.25, 1.0]
            [25 / (100 + 100 - 25), 0 / (100 + 100 - 0)]  # [0.25, 0.0]
        ], dtype=torch.float32)

        self.assertTrue(torch.allclose(computed_iou, expected_iou, atol=1e-4))

    def test_match_anchors_no_gt(self):
        """
        Test the match_anchors function when there are no ground truth boxes.
        """
        anchors = torch.tensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20]
        ], dtype=torch.float32)
        gt_boxes = torch.empty((0, 4), dtype=torch.float32)

        pos_inds, neg_inds, matched_gt_inds = match_anchors(anchors, gt_boxes)

        self.assertEqual(pos_inds.numel(), 0)
        self.assertTrue(torch.equal(neg_inds, torch.tensor([0, 1], dtype=torch.long)))
        self.assertEqual(matched_gt_inds.numel(), 0)

    def test_match_anchors_with_gt(self):
        """
        Test the match_anchors function with ground truth boxes.
        """
        anchors = torch.tensor([
            [0, 0, 10, 10],  # Perfect overlap with GT1
            [10, 10, 20, 20],  # Low overlap with GT1
            [5, 5, 15, 15],  # Perfect overlap with GT2
            [20, 20, 30, 30]  # No overlap
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [0, 0, 10, 10],  # GT1
            [5, 5, 15, 15]  # GT2
        ], dtype=torch.float32)

        pos_inds, neg_inds, matched_gt_inds = match_anchors(anchors, gt_boxes, iou_threshold_pos=0.5,
                                                            iou_threshold_neg=0.4)

        # Expected:
        # Anchor 0: IoU with GT1 = 1.0 (positive)
        # Anchor 1: IoU with GT1 = 0.25, GT2 = 0.0 (negative)
        # Anchor 2: IoU with GT1 = 0.25, GT2 = 1.0 (positive)
        # Anchor 3: IoU with GT1 = 0.0, GT2 = 0.0 (negative)

        expected_pos_inds = torch.tensor([0, 2], dtype=torch.long)
        expected_neg_inds = torch.tensor([1, 3], dtype=torch.long)
        expected_matched_gt_inds = torch.tensor([0, 1], dtype=torch.long)

        self.assertTrue(torch.equal(pos_inds, expected_pos_inds))
        self.assertTrue(torch.equal(neg_inds, expected_neg_inds))
        self.assertTrue(torch.equal(matched_gt_inds, expected_matched_gt_inds))

    def test_decode_boxes(self):
        """
        Test the decode_boxes function.
        """
        bbox_preds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],  # No change
            [0.1, 0.1, 0.1, 0.1],  # Small shift and scale
            [-0.1, -0.1, -0.1, -0.1]  # Small reverse shift and scale
        ], dtype=torch.float32)

        anchors = torch.tensor([
            [50, 50, 100, 100],
            [150, 150, 200, 200],
            [250, 250, 300, 300]
        ], dtype=torch.float32)

        decoded = decode_boxes(bbox_preds, anchors)

        # Manually compute expected decoded boxes
        # For bbox_preds = [0,0,0,0], decoded = same as anchors
        expected_decoded = anchors.clone()

        # For bbox_preds = [0.1,0.1,0.1,0.1]
        # dx=0.1, dy=0.1, dw=0.1, dh=0.1
        # anchor_w = 50
        # anchor_h = 50
        # anchor_cx = 75
        # anchor_cy = 75
        # pred_cx = 0.1 * 50 + 75 = 80
        # pred_cy = 0.1 * 50 + 75 = 80
        # pred_w = 50 * exp(0.1) ≈ 50 * 1.10517 ≈ 55.2585
        # pred_h = 50 * exp(0.1) ≈ 55.2585
        # decoded_x1 = 80 - 55.2585/2 ≈ 80 - 27.62925 ≈ 52.37075
        # decoded_y1 = same ≈ 52.37075
        # decoded_x2 = 80 + 27.62925 ≈ 107.62925
        # decoded_y2 = same ≈ 107.62925
        expected_decoded[1] = torch.tensor([52.3708, 52.3708, 107.6292, 107.6292], dtype=torch.float32)

        # For bbox_preds = [-0.1,-0.1,-0.1,-0.1]
        # dx=-0.1, dy=-0.1, dw=-0.1, dh=-0.1
        # anchor_w = 50
        # anchor_h = 50
        # anchor_cx = 275
        # anchor_cy = 275
        # pred_cx = -0.1 * 50 + 275 = 270
        # pred_cy = -0.1 * 50 + 275 = 270
        # pred_w = 50 * exp(-0.1) ≈ 50 * 0.904837 ≈ 45.24185
        # pred_h = 50 * exp(-0.1) ≈ 45.24185
        # decoded_x1 = 270 - 45.24185/2 ≈ 270 - 22.620925 ≈ 247.379075
        # decoded_y1 = same ≈ 247.379075
        # decoded_x2 = 270 + 22.620925 ≈ 292.620925
        # decoded_y2 = same ≈ 292.620925
        expected_decoded[2] = torch.tensor([247.3791, 247.3791, 292.6209, 292.6209], dtype=torch.float32)

        self.assertTrue(torch.allclose(decoded[0], expected_decoded[0], atol=1e-4))
        self.assertTrue(torch.allclose(decoded[1], expected_decoded[1], atol=1e-4))
        self.assertTrue(torch.allclose(decoded[2], expected_decoded[2], atol=1e-4))

    def test_giou(self):
        """
        Test the giou function.
        """
        pred_boxes = torch.tensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [5, 5, 15, 15],
            [0, 0, 10, 10],
            [0, 0, 5, 5]
        ], dtype=torch.float32)

        computed_giou = giou(pred_boxes, gt_boxes)

        # Manually compute GIoU for each pair
        # Pair 1: pred [0,0,10,10], gt [5,5,15,15]
        # IoU = 25 / (100 + 100 - 25) = 0.25
        # Enclosing box: [0,0,15,15], area = 225
        # Union = 175
        # GIoU = 0.25 - (225 - 175)/225 = 0.25 - 50/225 = 0.25 - 0.2222 ≈ 0.0278

        # Pair 2: pred [10,10,20,20], gt [0,0,10,10]
        # IoU = 0
        # Enclosing box: [0,0,20,20], area = 400
        # Union = 200
        # GIoU = 0 - (400 - 200)/400 = 0 - 0.5 = -0.5

        # Pair 3: pred [5,5,15,15], gt [0,0,5,5]
        # IoU = 0
        # Enclosing box: [0,0,15,15], area = 225
        # Union = 150
        # GIoU = 0 - (225 - 150)/225 = 0 - 75/225 = -0.3333

        expected_giou = torch.tensor([
            0.0278,
            -0.5,
            -0.3333
        ], dtype=torch.float32)

        self.assertTrue(torch.allclose(computed_giou, expected_giou, atol=1e-3))

    def test_giou_no_overlap(self):
        """
        Test GIoU when there is no overlap between boxes.
        """
        pred_boxes = torch.tensor([
            [0, 0, 10, 10],
            [20, 20, 30, 30]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [15, 15, 25, 25],
            [35, 35, 45, 45]
        ], dtype=torch.float32)

        computed_giou = giou(pred_boxes, gt_boxes)

        # Manually compute expected GIoU
        # For first pair:
        # IoU = 0
        # Enclosing box: [0,0,25,25], area = 625
        # Union = 100 + 100 = 200
        # GIoU = 0 - (625 - 200)/625 = 0 - 425/625 = -0.68

        # For second pair:
        # IoU = 0
        # Enclosing box: [20,20,45,45], area = 625
        # Union = 100 + 100 = 200
        # GIoU = 0 - (625 - 200)/625 = -0.68

        expected_giou = torch.tensor([-0.68, -0.68], dtype=torch.float32)

        self.assertTrue(torch.allclose(computed_giou, expected_giou, atol=1e-2))

    def test_giou_partial_overlap(self):
        """
        Test GIoU with partial overlap.
        """
        pred_boxes = torch.tensor([
            [0, 0, 10, 10]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [5, 5, 15, 15]
        ], dtype=torch.float32)

        computed_giou = giou(pred_boxes, gt_boxes)

        # Manually compute GIoU
        # IoU = 25 / (100 + 100 - 25) = 0.25
        # Enclosing box: [0,0,15,15], area = 225
        # Union = 175
        # GIoU = 0.25 - (225 - 175)/225 = 0.25 - 50/225 ≈ 0.0278

        expected_giou = torch.tensor([0.0278], dtype=torch.float32)

        self.assertTrue(torch.allclose(computed_giou, expected_giou, atol=1e-3))

    def test_giou_perfect_overlap(self):
        """
        Test GIoU when boxes perfectly overlap.
        """
        pred_boxes = torch.tensor([
            [10, 10, 20, 20]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [10, 10, 20, 20]
        ], dtype=torch.float32)

        computed_giou = giou(pred_boxes, gt_boxes)

        # Perfect overlap: IoU = 1.0
        # Enclosing box same as boxes, GIoU = 1.0
        expected_giou = torch.tensor([1.0], dtype=torch.float32)

        self.assertTrue(torch.allclose(computed_giou, expected_giou, atol=1e-4))

    def test_giou_large_enclosing_box(self):
        """
        Test GIoU with a large enclosing box.
        """
        pred_boxes = torch.tensor([
            [0, 0, 10, 10]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [20, 20, 30, 30]
        ], dtype=torch.float32)

        computed_giou = giou(pred_boxes, gt_boxes)

        # IoU = 0
        # Enclosing box: [0,0,30,30], area = 900
        # Union = 100 + 100 = 200
        # GIoU = 0 - (900 - 200)/900 = -700/900 ≈ -0.7778

        expected_giou = torch.tensor([-0.7778], dtype=torch.float32)

        self.assertTrue(torch.allclose(computed_giou, expected_giou, atol=1e-4))

    def test_giou_batch_processing(self):
        """
        Test the giou function with batched processing.
        """
        pred_boxes = torch.tensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [0, 0, 10, 10]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [5, 5, 15, 15],
            [0, 0, 10, 10],
            [0, 0, 5, 5],
            [5, 5, 15, 15]
        ], dtype=torch.float32)

        # Compute GIoU without batching
        computed_giou = giou(pred_boxes, gt_boxes)

        # Compute GIoU with batching
        computed_giou_batched = giou(pred_boxes, gt_boxes, batched=True, batch_size=2)

        self.assertTrue(torch.allclose(computed_giou, computed_giou_batched, atol=1e-6))

    def test_generate_anchors_multiple_scales_aspect_ratios(self):
        """
        Test generate_anchors with multiple scales and aspect ratios.
        """
        fm_size = (1, 1)  # Single location
        scales = [32, 64]
        aspect_ratios = [0.5, 1.0, 2.0]
        stride = 256  # Assuming image size 512x512

        expected_num_anchors = len(scales) * len(aspect_ratios)
        generated_anchors = generate_anchors(fm_size, scales, aspect_ratios, stride)

        self.assertEqual(generated_anchors.shape[0], expected_num_anchors)
        self.assertEqual(generated_anchors.shape[1], 4)

        # Verify coordinates
        cx, cy = stride * 0.5, stride * 0.5  # (128, 128)
        expected_anchors = []
        for scale in scales:
            for ratio in aspect_ratios:
                w = scale * math.sqrt(ratio)
                h = scale / math.sqrt(ratio)
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                expected_anchors.append([x1, y1, x2, y2])

        expected_anchors = torch.tensor(expected_anchors, dtype=torch.float32)
        self.assertTrue(torch.allclose(generated_anchors, expected_anchors, atol=1e-4))

    def test_generate_all_anchors_multiple_feature_maps(self):
        """
        Test generate_all_anchors with multiple feature maps.
        """
        fm_sizes = [
            (2, 2),  # stride 256
            (1, 1)  # stride 512
        ]
        strides = [256, 512]
        scales = [32]
        aspect_ratios = [1.0]
        image_size = (512, 512)
        device = torch.device('cpu')

        anchors = generate_all_anchors(fm_sizes, scales, aspect_ratios, strides, device)

        # For (2,2): 2x2 locations * 1 scale * 1 ratio = 4 anchors
        # For (1,1): 1x1 locations * 1 scale * 1 ratio = 1 anchor
        expected_num_anchors = 4 + 1
        self.assertEqual(anchors.shape[0], expected_num_anchors)
        self.assertEqual(anchors.shape[1], 4)

        # Verify coordinates
        # For (2,2) with stride 256:
        # Centers at (128,128), (384,128), (128,384), (384,384)
        # Each with scale 32 and ratio 1.0: width=32, height=32
        expected_anchors_fm1 = torch.tensor([
            [128 - 16, 128 - 16, 128 + 16, 128 + 16],
            [384 - 16, 128 - 16, 384 + 16, 128 + 16],
            [128 - 16, 384 - 16, 128 + 16, 384 + 16],
            [384 - 16, 384 - 16, 384 + 16, 384 + 16]
        ], dtype=torch.float32)

        # For (1,1) with stride 512:
        # Center at (256,256)
        # Anchor with scale 32 and ratio 1.0: width=32, height=32
        expected_anchors_fm2 = torch.tensor([
            [256 - 16, 256 - 16, 256 + 16, 256 + 16]
        ], dtype=torch.float32)

        expected_anchors = torch.cat([expected_anchors_fm1, expected_anchors_fm2], dim=0)
        self.assertTrue(torch.allclose(anchors, expected_anchors, atol=1e-4))

    def test_generate_all_anchors_empty_feature_map(self):
        """
        Test generate_all_anchors with one empty feature map.
        """
        fm_sizes = [
            (2, 2),  # stride 256
            (0, 0),  # Empty feature map
            (1, 1)  # stride 512
        ]
        strides = [256, 256, 512]
        scales = [32]
        aspect_ratios = [1.0]
        image_size = (512, 512)
        device = torch.device('cpu')

        anchors = generate_all_anchors(fm_sizes, scales, aspect_ratios, strides, device)

        # For (2,2): 4 anchors
        # For (0,0): 0 anchors
        # For (1,1): 1 anchor
        expected_num_anchors = 4 + 0 + 1
        self.assertEqual(anchors.shape[0], expected_num_anchors)
        self.assertEqual(anchors.shape[1], 4)

    def test_generate_all_anchors_large_feature_map(self):
        """
        Test generate_all_anchors with a large feature map.
        """
        fm_sizes = [
            (100, 100)  # stride 5
        ]
        strides = [5]
        scales = [32]
        aspect_ratios = [1.0]
        image_size = (500, 500)
        device = torch.device('cpu')

        anchors = generate_all_anchors(fm_sizes, scales, aspect_ratios, strides, device)

        expected_num_anchors = 100 * 100 * len(scales) * len(aspect_ratios)
        self.assertEqual(anchors.shape[0], expected_num_anchors)
        self.assertEqual(anchors.shape[1], 4)

    def test_generate_anchors_zero_scales(self):
        """
        Test generate_anchors with zero scales.
        """
        fm_size = (2, 2)
        scales = [0]
        aspect_ratios = [1.0]
        stride = 256

        generated_anchors = generate_anchors(fm_size, scales, aspect_ratios, stride)
        expected_anchors = torch.tensor([
            [128 - 0, 128 - 0, 128 + 0, 128 + 0],
            [384 - 0, 128 - 0, 384 + 0, 128 + 0],
            [128 - 0, 384 - 0, 128 + 0, 384 + 0],
            [384 - 0, 384 - 0, 384 + 0, 384 + 0]
        ], dtype=torch.float32)

        self.assertTrue(torch.allclose(generated_anchors, expected_anchors))

    def test_generate_anchors_negative_aspect_ratio(self):
        """
        Test generate_anchors with negative aspect ratios.
        """
        fm_size = (1, 1)
        scales = [32]
        aspect_ratios = [-1.0]  # Invalid aspect ratio
        stride = 256

        with self.assertRaises(ValueError):
            generate_anchors(fm_size, scales, aspect_ratios, stride)

    def test_generate_anchors_non_square_stride(self):
        """
        Test generate_anchors with non-square stride.
        """
        fm_size = (1, 2)  # height=1, width=2
        strides = [256, 128]  # Non-square strides
        # Since generate_anchors only accepts one stride per feature map, need to adjust the function
        # Here, for simplicity, assume stride is same for both dimensions
        stride = 256  # Take first stride

        scales = [32]
        aspect_ratios = [1.0]

        expected_anchors = torch.tensor([
            [128 - 16, 256 - 16, 128 + 16, 256 + 16],
            [384 - 16, 256 - 16, 384 + 16, 256 + 16]
        ], dtype=torch.float32)

        generated_anchors = generate_anchors(fm_size, scales, aspect_ratios, stride)
        self.assertEqual(generated_anchors.shape[0], 2)
        self.assertTrue(torch.allclose(generated_anchors, expected_anchors))

    def test_decode_boxes_invalid_input(self):
        """
        Test decode_boxes with mismatched number of anchors and bbox_preds.
        """
        bbox_preds = torch.zeros((2, 4), dtype=torch.float32)
        anchors = torch.zeros((3, 4), dtype=torch.float32)

        with self.assertRaises(AssertionError):
            decode_boxes(bbox_preds, anchors)

    def test_giou_invalid_input(self):
        """
        Test giou with mismatched number of pred and gt boxes.
        """
        pred_boxes = torch.zeros((2, 4), dtype=torch.float32)
        gt_boxes = torch.zeros((3, 4), dtype=torch.float32)

        with self.assertRaises(ValueError):
            giou(pred_boxes, gt_boxes)

    def test_match_anchors_all_positive(self):
        """
        Test match_anchors where all anchors are positive.
        """
        anchors = torch.tensor([
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [0, 0, 10, 10]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [0, 0, 10, 10]
        ], dtype=torch.float32)

        pos_inds, neg_inds, matched_gt_inds = match_anchors(anchors, gt_boxes, iou_threshold_pos=0.5,
                                                            iou_threshold_neg=0.4)

        expected_pos_inds = torch.tensor([0, 1, 2], dtype=torch.long)
        expected_neg_inds = torch.tensor([], dtype=torch.long)
        expected_matched_gt_inds = torch.tensor([0, 0, 0], dtype=torch.long)

        self.assertTrue(torch.equal(pos_inds, expected_pos_inds))
        self.assertEqual(neg_inds.numel(), 0)
        self.assertTrue(torch.equal(matched_gt_inds, expected_matched_gt_inds))

    def test_match_anchors_all_negative(self):
        """
        Test match_anchors where all anchors are negative.
        """
        anchors = torch.tensor([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [60, 60, 70, 70]
        ], dtype=torch.float32)

        pos_inds, neg_inds, matched_gt_inds = match_anchors(anchors, gt_boxes, iou_threshold_pos=0.5,
                                                            iou_threshold_neg=0.4)

        expected_pos_inds = torch.tensor([], dtype=torch.long)
        expected_neg_inds = torch.tensor([0, 1, 2], dtype=torch.long)
        expected_matched_gt_inds = torch.tensor([], dtype=torch.long)

        self.assertEqual(pos_inds.numel(), 0)
        self.assertTrue(torch.equal(neg_inds, torch.tensor([0, 1, 2], dtype=torch.long)))
        self.assertEqual(matched_gt_inds.numel(), 0)

    def test_match_anchors_multiple_gt_boxes(self):
        """
        Test match_anchors with multiple ground truth boxes.
        """
        anchors = torch.tensor([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [10, 10, 20, 20],
            [15, 15, 25, 25]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15]
        ], dtype=torch.float32)

        pos_inds, neg_inds, matched_gt_inds = match_anchors(anchors, gt_boxes, iou_threshold_pos=0.5,
                                                            iou_threshold_neg=0.4)

        # Expected:
        # Anchor 0: IoU with GT1=1.0 → positive
        # Anchor 1: IoU with GT1=0.25, GT3=0.25 → negative
        # Anchor 2: IoU with GT2=1.0 → positive
        # Anchor 3: IoU with GT2=0.25 → negative
        expected_pos_inds = torch.tensor([0, 2], dtype=torch.long)
        expected_neg_inds = torch.tensor([1, 3], dtype=torch.long)
        expected_matched_gt_inds = torch.tensor([0, 1], dtype=torch.long)

        self.assertTrue(torch.equal(pos_inds, expected_pos_inds))
        self.assertTrue(torch.equal(neg_inds, expected_neg_inds))
        self.assertTrue(torch.equal(matched_gt_inds, expected_matched_gt_inds))

    def test_match_anchors_multiple_matches(self):
        """
        Test match_anchors where multiple anchors match the same GT box.
        """
        anchors = torch.tensor([
            [0, 0, 10, 10],
            [1, 1, 9, 9],
            [2, 2, 8, 8],
            [10, 10, 20, 20]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20]
        ], dtype=torch.float32)

        pos_inds, neg_inds, matched_gt_inds = match_anchors(anchors, gt_boxes, iou_threshold_pos=0.5,
                                                            iou_threshold_neg=0.4)

        # Expected:
        # Anchors 0,1,2: IoU with GT1 >= 0.5 → positive
        # Anchor 3: IoU with GT2 = 1.0 → positive
        expected_pos_inds = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        expected_neg_inds = torch.tensor([], dtype=torch.long)
        expected_matched_gt_inds = torch.tensor([0, 0, 0, 1], dtype=torch.long)

        self.assertTrue(torch.equal(pos_inds, expected_pos_inds))
        self.assertEqual(neg_inds.numel(), 0)
        self.assertTrue(torch.equal(matched_gt_inds, expected_matched_gt_inds))

    def test_match_anchors_thresholds(self):
        """
        Test match_anchors with different IoU thresholds.
        """
        anchors = torch.tensor([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [10, 10, 20, 20],
            [15, 15, 25, 25]
        ], dtype=torch.float32)

        gt_boxes = torch.tensor([
            [5, 5, 15, 15]
        ], dtype=torch.float32)

        # Test with high IoU threshold
        pos_inds, neg_inds, matched_gt_inds = match_anchors(anchors, gt_boxes, iou_threshold_pos=0.6,
                                                            iou_threshold_neg=0.4)
        # Expected:
        # Anchor 0: IoU with GT = 25 / (100 + 100 - 25) = 0.25 < 0.6 → negative
        # Anchor 1: IoU = 100 / (100 + 100 - 100) = 1.0 >= 0.6 → positive
        # Anchor 2: IoU = 25 / (100 + 100 - 25) = 0.25 < 0.6 → negative
        # Anchor 3: IoU = 0 < 0.6 → negative
        expected_pos_inds = torch.tensor([1], dtype=torch.long)
        expected_neg_inds = torch.tensor([0, 2, 3], dtype=torch.long)
        expected_matched_gt_inds = torch.tensor([0], dtype=torch.long)

        self.assertTrue(torch.equal(pos_inds, expected_pos_inds))
        self.assertTrue(torch.equal(neg_inds, expected_neg_inds))
        self.assertTrue(torch.equal(matched_gt_inds, expected_matched_gt_inds))

        # Test with lower IoU threshold
        pos_inds, neg_inds, matched_gt_inds = match_anchors(anchors, gt_boxes, iou_threshold_pos=0.2,
                                                            iou_threshold_neg=0.1)
        # Expected:
        # Anchor 0: IoU = 0.25 >= 0.2 → positive
        # Anchor 1: IoU = 1.0 >= 0.2 → positive
        # Anchor 2: IoU = 0.25 >= 0.2 → positive
        # Anchor 3: IoU = 0 < 0.2 → negative
        expected_pos_inds = torch.tensor([0, 1, 2], dtype=torch.long)
        expected_neg_inds = torch.tensor([3], dtype=torch.long)
        expected_matched_gt_inds = torch.tensor([0, 0, 0], dtype=torch.long)

        self.assertTrue(torch.equal(pos_inds, expected_pos_inds))
        self.assertTrue(torch.equal(neg_inds, expected_neg_inds))
        self.assertTrue(torch.equal(matched_gt_inds, expected_matched_gt_inds))

    def test_generate_anchors_multiple_feature_maps_large(self):
        """
        Test generate_all_anchors with multiple feature maps, including a large feature map.
        """
        fm_sizes = [
            (2, 2),  # stride 256
            (100, 100),  # stride 5
            (1, 1)  # stride 512
        ]
        strides = [256, 5, 512]
        scales = [32]
        aspect_ratios = [1.0]
        image_size = (500, 500)  # Adjusted for stride=5
        device = torch.device('cpu')

        anchors = generate_all_anchors(fm_sizes, scales, aspect_ratios, strides, device)

        # For (2,2): 4 anchors
        # For (100,100): 10000 anchors
        # For (1,1): 1 anchor
        expected_num_anchors = 4 + 10000 + 1
        self.assertEqual(anchors.shape[0], expected_num_anchors)
        self.assertEqual(anchors.shape[1], 4)

    # Add more tests as needed...


if __name__ == '__main__':
    unittest.main()
