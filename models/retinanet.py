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
        # Note: detection models are kept in fp32 for stability unless specified
        if self.dtype == torch.float16:
            self.logger.warning("  ⚠️  RetinaNet in fp16 may produce NaN — using fp32")
            self.dtype = torch.float32
        self.logger.info("  ✅ RetinaNet loaded")

    def prepare_data(self):
        # RetinaNet expects list of [C, H, W] float tensors
        self._sample_batch = [
            torch.randn(3, 640, 640, dtype=torch.float32, device=self.device)
            for _ in range(self.batch_size)
        ]
        self.logger.info(f"  ✅ Sample batch ready: {self.batch_size}x [3, 640, 640]")

    def run_inference(self):
        with torch.no_grad():
            _ = self.model(self._sample_batch)
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()