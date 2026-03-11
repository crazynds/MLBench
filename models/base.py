"""Base class for all model handlers"""

import abc
import torch
from pathlib import Path


class BaseModelHandler(abc.ABC):
    MODELS_DIR = Path("data/models")
    DATASETS_DIR = Path("data/datasets")

    def __init__(self, device, precision: str, batch_size: int, logger):
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.logger = logger
        self.model = None
        self._sample_batch = None

        # Resolve dtype
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(precision, torch.float32)

    @abc.abstractmethod
    def load(self):
        """Load model weights onto device"""
        ...

    @abc.abstractmethod
    def prepare_data(self):
        """Load or generate a sample batch for inference"""
        ...

    @abc.abstractmethod
    def run_inference(self):
        """Run one inference pass (no grad)"""
        ...

    def cleanup(self):
        """Free GPU memory"""
        del self.model
        self.model = None
        del self._sample_batch
        self._sample_batch = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _to_device(self, tensor):
        return tensor.to(self.device, dtype=self.dtype)

    def _autocast_context(self):
        """Returns an autocast context appropriate for the device/precision."""
        if str(self.device) == "cpu":
            if self.dtype == torch.bfloat16:
                return torch.autocast("cpu", dtype=torch.bfloat16)
            return torch.no_grad()  # no autocast for fp32 on CPU
        # CUDA
        if self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=self.dtype)
        return torch.no_grad()