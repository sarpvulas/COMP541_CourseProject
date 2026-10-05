import os
import tempfile
import unittest

import torch

from utils import save_model


class TestSaveModel(unittest.TestCase):
    def test_saves_only_on_strict_improvement(self):
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as d:
            pattern = os.path.join(d, "ckpt", "m_{}.pth")
            self.assertEqual(save_model(model, 1, 0.0, 0.0, pattern), 0.0)
            self.assertFalse(os.path.exists(os.path.join(d, "ckpt", "m_1.pth")))
            self.assertEqual(save_model(model, 2, 0.5, 0.0, pattern), 0.5)
            self.assertTrue(os.path.exists(os.path.join(d, "ckpt", "m_2.pth")))
            self.assertEqual(save_model(model, 3, 0.4, 0.5, pattern), 0.5)
            self.assertFalse(os.path.exists(os.path.join(d, "ckpt", "m_3.pth")))


if __name__ == "__main__":
    unittest.main()
