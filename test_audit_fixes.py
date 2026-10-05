"""Tests for the fixes made in the audit PR: offset module, normalisation, seeding, split,
class indices and the GIoU regression loss."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image
from torchvision.models import resnet50
from torchvision.ops import generalized_box_iou

import modules.dual_backbone as dual_backbone
from losses import giou_regression_loss
from modules.normalize import IMAGENET_MEAN, IMAGENET_STD, ImageNetNormalize
from modules.seed import make_generator, set_seed
from TOODHead import TOODHead
from utils import CocoDetectionWithFilename, detection_collate_fn, split_train_val


class TestTOODOffsetModule(unittest.TestCase):
    """Pure torch: the head falls back to torchvision's deform_conv2d when mmcv is missing."""

    def make_head(self):
        torch.manual_seed(0)
        return TOODHead(in_channels=8, num_classes=3, num_anchors=2, stacked_convs=2,
                        feat_channels=16)

    def test_offset_module_is_registered(self):
        head = self.make_head()
        names = [n for n, _ in head.named_parameters()]
        self.assertTrue(any(n.startswith("reg_offset_module") for n in names), names)

    def test_offset_module_is_not_rebuilt_per_forward(self):
        head = self.make_head()
        feats = [torch.randn(1, 8, 8, 8)]
        before = [p.detach().clone() for p in head.reg_offset_module.parameters()]
        with torch.no_grad():
            first = head(feats)[1][0]
            second = head(feats)[1][0]
        after = list(head.reg_offset_module.parameters())
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(b, a))
        self.assertTrue(torch.equal(first, second))  # same weights -> same output

    def test_offset_module_receives_gradients(self):
        head = self.make_head()
        feats = [torch.randn(2, 8, 8, 8), torch.randn(2, 8, 4, 4)]
        cls_scores, reg_preds = head(feats)
        sum(r.sum() for r in reg_preds).backward()
        grads = [p.grad for p in head.reg_offset_module.parameters()]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))


class TestImageNetNormalize(unittest.TestCase):
    def test_values(self):
        x = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).expand(1, 3, 2, 2).clone()
        self.assertTrue(torch.allclose(ImageNetNormalize()(x), torch.zeros_like(x), atol=1e-6))
        x = torch.ones(1, 3, 2, 2)
        expected = torch.tensor([(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
        self.assertTrue(torch.allclose(ImageNetNormalize()(x)[0, :, 0, 0], expected))

    def test_no_new_checkpoint_keys(self):
        self.assertEqual(list(ImageNetNormalize().state_dict().keys()), [])

    def test_rgb_branch_is_normalised_and_gray_branch_is_not(self):
        # Random ResNets instead of downloading pretrained weights.
        with mock.patch.object(dual_backbone, "resnet50", lambda weights=None: resnet50(weights=None)):
            net = dual_backbone.DualResNet50()
        seen = {}
        net.rgb_layer1.register_forward_hook(lambda m, i, o: seen.__setitem__("rgb", i[0]))
        net.gray_layer1.register_forward_hook(lambda m, i, o: seen.__setitem__("gray", i[0]))
        x = torch.rand(1, 3, 64, 64)
        net.eval()
        with torch.no_grad():
            net(x, x)
            expected_rgb = net.rgb_conv1(ImageNetNormalize()(x))
            expected_gray = net.gray_conv1(x)
        self.assertTrue(torch.allclose(seen["rgb"], expected_rgb, atol=1e-5))
        self.assertTrue(torch.allclose(seen["gray"], expected_gray, atol=1e-5))


class TestSeedAndSplit(unittest.TestCase):
    def test_same_seed_same_permutation_and_weights(self):
        a = torch.randperm(50, generator=make_generator(7))
        b = torch.randperm(50, generator=make_generator(7))
        c = torch.randperm(50, generator=make_generator(8))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        set_seed(3)
        w1 = torch.nn.Linear(4, 4).weight.clone()
        set_seed(3)
        w2 = torch.nn.Linear(4, 4).weight.clone()
        self.assertTrue(torch.equal(w1, w2))

    def test_split_is_disjoint_complete_fixed_and_sized(self):
        train, val = split_train_val(100, val_fraction=0.1, split_seed=1234)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(train), 90)
        self.assertEqual(sorted(train + val), list(range(100)))
        self.assertEqual((train, val), split_train_val(100, val_fraction=0.1, split_seed=1234))
        self.assertNotEqual(val, split_train_val(100, val_fraction=0.1, split_seed=1)[1])

    def test_split_edge_cases(self):
        self.assertEqual(split_train_val(1), ([0], []))
        train, val = split_train_val(2, val_fraction=0.9)
        self.assertEqual((len(train), len(val)), (1, 1))


def write_coco(root, category_ids):
    root = Path(root)
    (root / "img").mkdir()
    Image.new("RGB", (600, 600)).save(root / "img" / "1.jpg")
    cats = [{"id": c, "name": str(c)} for c in category_ids]
    anns = [{"id": i + 1, "image_id": 1, "category_id": c, "bbox": [10, 10, 100, 100],
             "area": 10000, "iscrowd": 0} for i, c in enumerate(category_ids)]
    ann = {"images": [{"id": 1, "file_name": "1.jpg", "width": 600, "height": 600}],
           "annotations": anns, "categories": cats}
    (root / "ann.json").write_text(json.dumps(ann))
    return str(root / "img"), str(root / "ann.json")


class TestClassIndices(unittest.TestCase):
    def labels(self, category_ids):
        with tempfile.TemporaryDirectory() as d:
            img, ann = write_coco(d, category_ids)
            ds = CocoDetectionWithFilename(img, ann)
            _, targets, _ = detection_collate_fn([ds[0]])
        return targets[0]["labels"].tolist()

    def test_ids_1_to_10_match_old_rule(self):
        self.assertEqual(self.labels(list(range(1, 11))), list(range(10)))

    def test_ids_0_to_9_are_not_shifted_out_of_range(self):
        # the old rule (category_id - 1) would give -1 for id 0
        self.assertEqual(self.labels(list(range(10))), list(range(10)))

    def test_wrong_category_count_raises(self):
        with tempfile.TemporaryDirectory() as d:
            img, ann = write_coco(d, [1, 2, 3])
            with self.assertRaises(ValueError):
                CocoDetectionWithFilename(img, ann)


class TestGiouRegressionLoss(unittest.TestCase):
    def test_matches_torchvision_diagonal_not_full_matrix(self):
        pred = torch.tensor([[0., 0, 10, 10], [0, 0, 5, 5], [2, 2, 8, 9]])
        gt = torch.tensor([[0., 0, 10, 10], [20, 20, 30, 30], [3, 1, 9, 8]])
        loss = giou_regression_loss(pred, gt)
        matrix = generalized_box_iou(pred, gt)
        self.assertAlmostEqual(loss.item(), (1 - matrix.diag()).mean().item(), places=5)
        self.assertNotAlmostEqual(loss.item(), (1 - matrix).mean().item(), places=3)

    def test_perfect_match_is_zero_and_gradient_flows(self):
        box = torch.tensor([[1., 1, 5, 5]], requires_grad=True)
        loss = giou_regression_loss(box, box.detach().clone())
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        giou_regression_loss(box, torch.tensor([[3., 3, 9, 9]])).backward()
        self.assertTrue(box.grad.abs().sum() > 0)

    def test_empty_is_zero(self):
        self.assertEqual(giou_regression_loss(torch.zeros(0, 4), torch.zeros(0, 4)).item(), 0.0)


class CountingLoader:
    """A one-batch loader that records how many times it was iterated."""

    def __init__(self, label):
        img = torch.rand(3, 520, 520)
        tgt = {"boxes": torch.tensor([[250., 250, 350, 350]]), "labels": torch.tensor([label]),
               "image_id": 1}
        self.batch = ([img, img.clone()], [tgt, dict(tgt)], ["a", "b"])
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return iter([self.batch])

    def __len__(self):
        return 1


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 1)

    def forward(self, x):
        return [self.conv(x)]


class TestTrainModelSelection(unittest.TestCase):
    def test_test_split_is_evaluated_once_after_training(self):
        import train as train_module
        from losses import single_label_classification_loss

        torch.manual_seed(0)
        model = TinyClassifier()
        train_l, val_l, test_l = CountingLoader(3), CountingLoader(3), CountingLoader(3)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(train_module, "wandb"):
            pattern = os.path.join(d, "ck_{}.pth")
            result = train_module.train_model(
                model, train_l, val_l, test_l, opt, single_label_classification_loss,
                torch.device("cpu"), epochs=3, checkpoint_pattern=pattern)
            saved = sorted(os.listdir(d))
        self.assertEqual(train_l.passes, 3)
        self.assertEqual(val_l.passes, 3)   # once per epoch: used for selection
        self.assertEqual(test_l.passes, 1)  # once, at the end
        self.assertIsNotNone(result["best_epoch"])
        self.assertIn(f"ck_{result['best_epoch']}.pth", saved)
        self.assertEqual(result["test"]["total"], 2)


class TestBestCheckpointIsTested(unittest.TestCase):
    def test_test_split_sees_the_best_validation_epoch_weights(self):
        """Validation accuracy peaks at epoch 2 of 3; the test run must use epoch-2 weights."""
        import train as train_module
        from losses import single_label_classification_loss

        torch.manual_seed(0)
        model = TinyClassifier()
        val_l, test_l = CountingLoader(3), CountingLoader(3)
        train_l = CountingLoader(3)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        val_accs = iter([0.2, 0.9, 0.5])
        val_weights, test_weights = [], []

        def fake_eval(m, loader, device, image_size=None):
            w = m.conv.weight.detach().clone()
            if loader is val_l:
                val_weights.append(w)
                return {"accuracy": next(val_accs), "correct": 0, "total": 1, "skipped": 0}
            test_weights.append(w)
            return {"accuracy": 0.0, "correct": 0, "total": 1, "skipped": 0}

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(train_module, "wandb"), \
                mock.patch.object(train_module, "evaluate_top1", fake_eval):
            result = train_module.train_model(
                model, train_l, val_l, test_l, opt, single_label_classification_loss,
                torch.device("cpu"), epochs=3, checkpoint_pattern=os.path.join(d, "ck_{}.pth"))
            saved = sorted(os.listdir(d))
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_val_accuracy"], 0.9)
        self.assertEqual(saved, ["ck_1.pth", "ck_2.pth"])  # epoch 3 (0.5) did not save
        self.assertFalse(torch.equal(val_weights[1], val_weights[2]))  # weights moved after epoch 2
        self.assertEqual(len(test_weights), 1)
        self.assertTrue(torch.equal(test_weights[0], val_weights[1]))


class TestInputValidation(unittest.TestCase):
    def test_non_integer_seed_message(self):
        import importlib
        import modules.config as config
        with mock.patch.dict(os.environ, {"UOD_SEED": "abc"}):
            with self.assertRaisesRegex(ValueError, "UOD_SEED must be an integer"):
                importlib.reload(config)
        importlib.reload(config)

    def test_unknown_category_id_names_id_and_image(self):
        with tempfile.TemporaryDirectory() as d:
            img, ann = write_coco(d, list(range(1, 11)))
            data = json.loads(Path(ann).read_text())
            data["annotations"][0]["category_id"] = 99
            Path(ann).write_text(json.dumps(data))
            ds = CocoDetectionWithFilename(img, ann)
            with self.assertRaisesRegex(ValueError, "category_id 99"):
                ds[0]

    def test_set_seed_leaves_cudnn_alone_by_default(self):
        before = torch.backends.cudnn.deterministic
        set_seed(1)
        self.assertEqual(torch.backends.cudnn.deterministic, before)


if __name__ == "__main__":
    unittest.main()
