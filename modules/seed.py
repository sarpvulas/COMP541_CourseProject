"""Seeding helpers. One seed drives Python, NumPy, PyTorch and the data order."""
import random

import numpy as np
import torch


def set_seed(seed, deterministic=False):
    """
    Seed Python, NumPy and PyTorch (CPU and CUDA). With `deterministic=True` also set the
    global cuDNN flags (deterministic kernels, benchmark off); they are off by default so
    that calling this in tests does not change global state.

    This makes data order, subset choice and weight initialisation repeatable. It does not
    make a GPU run bit-identical: some CUDA backward kernels (for example adaptive average
    pooling and bilinear/nearest upsampling in the FPN) use atomic adds, whose summation
    order can change between runs. `torch.use_deterministic_algorithms` is not enabled
    because it raises on those kernels. Multi-GPU `DataParallel` runs are also not covered.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed):
    """A torch.Generator seeded with `seed`, for shuffling and subset selection."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
