"""Model handlers registry"""

from models.base import BaseModelHandler
from models.resnet50 import ResNet50Handler
from models.retinanet import RetinaNetHandler
from models.yolov11 import YOLOv11Handler
from models.stable_diffusion import StableDiffusionHandler
from models.dlrm_v2 import DLRMv2Handler
from models.whisper import WhisperHandler

_REGISTRY = {
    "resnet50": ResNet50Handler,
    "retinanet": RetinaNetHandler,
    "yolov11": YOLOv11Handler,
    "stable_diffusion": StableDiffusionHandler,
    "dlrm_v2": DLRMv2Handler,
    "whisper": WhisperHandler,
}


def get_model_handler(model_name: str, **kwargs) -> BaseModelHandler:
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[model_name](**kwargs)