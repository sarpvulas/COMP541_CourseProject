import io
import contextlib
import unittest

import torch
from torchvision.ops import generalized_box_iou

from modules.anchor_utils import giou
from modules.crop import build_batch, crop_and_filter, crop_and_label, largest_area_label
from modules.Debug import Debug
from modules.IDM import IDM

SIZE = (512, 512)


def make_target(boxes, labels, image_id=1):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "image_id": image_id,
    }


class TestCropAndLabel(unittest.TestCase):
    def setUp(self):
        # 1000x1000 image: the center crop is x,y in [244, 756]
        self.image = torch.zeros(3, 1000, 1000)

    def test_largest_box_mostly_outside_crop_does_not_win(self):
        # class 7: huge box (0..400) but only a sliver inside the crop -> dropped (< 50% visible)
        # class 2: small box fully inside the crop
        target = make_target([[0, 0, 400, 400], [300, 300, 400, 400]], [7, 2])
        # without the crop, class 7 would win by area
        self.assertEqual(largest_area_label(target["boxes"], target["labels"]), 7)
        cropped, label = crop_and_label(self.image, target, SIZE)
        self.assertEqual(cropped.shape, (3, 512, 512))
        self.assertEqual(label, 2)

    def test_boxes_and_labels_are_filtered_together(self):
        target = make_target(
            [[0, 0, 100, 100], [300, 300, 400, 400], [500, 500, 600, 600]], [1, 2, 3]
        )
        _, new = crop_and_filter(self.image, target, SIZE)
        self.assertEqual(new["boxes"].shape[0], new["labels"].shape[0])
        self.assertEqual(new["labels"].tolist(), [2, 3])
        # coordinates are shifted into the crop frame (offset 244)
        self.assertTrue(torch.allclose(new["boxes"][0], torch.tensor([56.0, 56.0, 156.0, 156.0])))

    def test_mostly_outside_box_dropped_partial_box_kept(self):
        # box 200..300 has 56 of 100 px inside on each axis -> 31% visible -> dropped
        target = make_target([[200, 200, 300, 300], [200, 300, 400, 400]], [1, 2])
        _, new = crop_and_filter(self.image, target, SIZE)
        self.assertEqual(new["labels"].tolist(), [2])

    def test_no_visible_box_returns_none(self):
        target = make_target([[0, 0, 50, 50]], [4])
        cropped, label = crop_and_label(self.image, target, SIZE)
        self.assertIsNone(label)
        self.assertEqual(cropped.shape, (3, 512, 512))

    def test_empty_target(self):
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long), "image_id": -1}
        _, label = crop_and_label(self.image, target, SIZE)
        self.assertIsNone(label)

    def test_build_batch_counts_skipped_images(self):
        good = make_target([[300, 300, 400, 400]], [5])
        bad = make_target([[0, 0, 50, 50]], [6])
        images = [self.image, self.image, self.image]
        batch, labels, skipped = build_batch(images, [good, bad, good], SIZE)
        self.assertEqual(batch.shape, (2, 3, 512, 512))
        self.assertEqual(labels.tolist(), [5, 5])
        self.assertEqual(skipped, 1)

    def test_build_batch_all_skipped(self):
        bad = make_target([[0, 0, 50, 50]], [6])
        batch, labels, skipped = build_batch([self.image], [bad], SIZE)
        self.assertIsNone(batch)
        self.assertIsNone(labels)
        self.assertEqual(skipped, 1)


class TestGiouTinyBoxes(unittest.TestCase):
    def test_tiny_identical_boxes_match_torchvision(self):
        boxes = torch.tensor([[0.0, 0.0, 1e-4, 1e-4], [1.0, 1.0, 1.0002, 1.0002]])
        ours = giou(boxes, boxes)
        ref = torch.diag(generalized_box_iou(boxes, boxes))
        self.assertTrue(torch.allclose(ours, ref, atol=1e-3))
        self.assertTrue(torch.allclose(ours, torch.ones(2), atol=1e-3))

    def test_tiny_disjoint_boxes_match_torchvision(self):
        a = torch.tensor([[0.0, 0.0, 1e-4, 1e-4]])
        b = torch.tensor([[2e-4, 2e-4, 3e-4, 3e-4]])
        ref = torch.diag(generalized_box_iou(a, b))
        self.assertTrue(torch.allclose(giou(a, b), ref, atol=1e-3))


class TestDebugOffByDefault(unittest.TestCase):
    def test_debug_helper_is_silent_by_default(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Debug(exit_on_nan=True).debug_tensor(torch.tensor([float("nan")]), desc="x")
        self.assertEqual(buf.getvalue(), "")

    def test_idm_forward_is_silent_by_default(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            IDM()(torch.rand(1, 3, 64, 64))
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
