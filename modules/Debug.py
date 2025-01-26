import torch

class Debug:
    """
    A helper class that provides methods to inspect tensors for shape,
    min/max values, and NaNs. You can configure whether to raise an
    exception when NaNs are found.
    """

    def __init__(self, exit_on_nan: bool = False):
        """
        Args:
            exit_on_nan (bool): If True, raise RuntimeError when NaNs are detected.
                                Otherwise, just print a debug warning.
        """
        self.exit_on_nan = exit_on_nan

    def debug_tensor(self, tensor_or_tensors, desc="tensor"):
        """
        Print shape, min, max, and detect NaNs for a single tensor or list/tuple of tensors.

        Args:
            tensor_or_tensors (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]):
                The tensor(s) to inspect.
            desc (str): A descriptor/name for logging.
        """
        if isinstance(tensor_or_tensors, (list, tuple)):
            # Handle multiple tensors
            for i, t in enumerate(tensor_or_tensors):
                self._inspect_one_tensor(t, f"{desc}[{i}]")
        else:
            # Single tensor
            self._inspect_one_tensor(tensor_or_tensors, desc)

    def _inspect_one_tensor(self, t, desc):
        """
        Private helper that checks a single tensor for shape, min, max, and NaNs.
        """
        if not isinstance(t, torch.Tensor):
            print(f"[DEBUG] {desc} is not a tensor: {type(t)}")
            return

        if torch.isnan(t).any():
            print(f"[DEBUG] Detected NaNs in {desc}, shape={t.shape}, "
                  f"min={t.min()}, max={t.max()}")
            if self.exit_on_nan:
                raise RuntimeError(f"NaNs found in {desc}!")
        else:
            print(f"[DEBUG] {desc} shape={t.shape}, min={t.min()}, max={t.max()}")
