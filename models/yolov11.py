"""YOLOv11 handler — Object Detection via ultralytics"""

import torch
import numpy as np
from pathlib import Path
from models.base import BaseModelHandler

YOLO_WEIGHTS_PATH = Path("data/models/yolov11/yolo11n.pt")


class YOLOv11Handler(BaseModelHandler):
    def load(self):
        self.logger.info("  Loading YOLOv11n (ultralytics)...")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics>=8.3")

        weights = str(YOLO_WEIGHTS_PATH) if YOLO_WEIGHTS_PATH.exists() else "yolo11n.pt"
        self.model = YOLO(weights)

        device_str = str(self.device)
        self._yolo_device = device_str
        self.logger.info(f"  ✅ YOLOv11n loaded on {device_str}")

    def prepare_data(self):
        self._batches = [
            [
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(self.batch_size)
            ]
            for _ in range(self.NUM_PRECOMPUTED_BATCHES)
        ]
        self.logger.info(f"  ✅ {self.NUM_PRECOMPUTED_BATCHES} batches ready: {self.batch_size}x [640, 640, 3]")

    def run_inference(self):
        batch = self._batches[self._next_batch_idx()]
        _ = self.model.predict(
            batch,
            device=self._yolo_device,
            verbose=False,
            half=(self.dtype == torch.float16),
        )
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()
