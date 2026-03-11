"""RetinaNet handler — Object Detection"""

import torch
import torchvision.models.detection as det_models
from models.base import BaseModelHandler


class RetinaNetHandler(BaseModelHandler):
    def load(self):
        self.logger.info("  Loading RetinaNet (torchvision pretrained)...")
        weights = det_models.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = det_models.retinanet_resnet50_fpn_v2(weights=weights)
        self.model.eval()
        self.model.to(self.device)
        if self.dtype == torch.float16:
            self.logger.warning("  ⚠️  RetinaNet in fp16 may produce NaN — using fp32")
            self.dtype = torch.float32
        self.logger.info("  ✅ RetinaNet loaded")

    def prepare_data(self):
        self._batches = [
            [
                torch.randn(3, 640, 640, dtype=torch.float32, device=self.device)
                for _ in range(self.batch_size)
            ]
            for _ in range(self.NUM_PRECOMPUTED_BATCHES)
        ]
        self.logger.info(f"  ✅ {self.NUM_PRECOMPUTED_BATCHES} batches ready: {self.batch_size}x [3, 640, 640]")

    def run_inference(self):
        batch = self._batches[self._next_batch_idx()]
        with torch.no_grad():
            _ = self.model(batch)
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()
