"""ResNet50 handler — Image Classification"""

import torch
import torchvision.models as models
from models.base import BaseModelHandler


class ResNet50Handler(BaseModelHandler):
    def load(self):
        self.logger.info("  Loading ResNet50 (torchvision pretrained)...")
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        self.model.eval()
        self.model.to(self.device)
        if self.dtype in (torch.float16, torch.bfloat16):
            self.model = self.model.to(dtype=self.dtype)
        self.logger.info("  ✅ ResNet50 loaded")

    def prepare_data(self):
        self._batches = [
            torch.randn(self.batch_size, 3, 224, 224, device=self.device, dtype=self.dtype)
            for _ in range(self.NUM_PRECOMPUTED_BATCHES)
        ]
        self.logger.info(f"  ✅ {self.NUM_PRECOMPUTED_BATCHES} batches ready: {tuple(self._batches[0].shape)}")

    def run_inference(self):
        batch = self._batches[self._next_batch_idx()]
        with torch.no_grad():
            with self._autocast_context():
                _ = self.model(batch)
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()
