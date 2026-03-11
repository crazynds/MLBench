"""
Model and dataset downloader.
Checks existence before downloading.
"""

import os
import shutil
import urllib.request
from pathlib import Path

MODELS_DIR = Path("data/models")
DATASETS_DIR = Path("data/datasets")

# ── Configuration per model ───────────────────────────────────────────────────
# Each entry: {model_dir, hf_repo (optional), dataset_dir, dataset_url or hf_dataset}
MODEL_CONFIG = {
    "resnet50": {
        "model_dir": MODELS_DIR / "resnet50",
        "hf_repo": None,           # loaded via torchvision, no download needed
        "dataset_dir": DATASETS_DIR / "imagenet_sample",
        "dataset_source": "hf",
        "hf_dataset": "imagenet-1k-sample",  # synthetic fallback handled in handler
        "note": "Uses torchvision pretrained weights + synthetic ImageNet-like data",
    },
    "retinanet": {
        "model_dir": MODELS_DIR / "retinanet",
        "hf_repo": None,           # torchvision
        "dataset_dir": DATASETS_DIR / "coco_sample",
        "dataset_source": "synthetic",
        "note": "Uses torchvision pretrained weights + synthetic COCO-like data",
    },
    "yolov11": {
        "model_dir": MODELS_DIR / "yolov11",
        "hf_repo": "ultralytics/assets",
        "model_file": "yolo11n.pt",
        "dataset_dir": DATASETS_DIR / "coco_sample",
        "dataset_source": "synthetic",
        "note": "Downloads YOLOv11n weights from ultralytics",
    },
    "stable_diffusion": {
        "model_dir": MODELS_DIR / "stable_diffusion",
        "hf_repo": "runwayml/stable-diffusion-v1-5",
        "dataset_dir": DATASETS_DIR / "sd_prompts",
        "dataset_source": "builtin",
        "note": "Downloads SD v1.5 from HuggingFace (requires ~4GB)",
    },
    "dlrm_v2": {
        "model_dir": MODELS_DIR / "dlrm_v2",
        "hf_repo": None,           # built from torchrec or custom
        "dataset_dir": DATASETS_DIR / "criteo_sample",
        "dataset_source": "synthetic",
        "note": "Uses synthetic Criteo-like tabular data",
    },
    "whisper": {
        "model_dir": MODELS_DIR / "whisper",
        "hf_repo": "openai/whisper-base",
        "dataset_dir": DATASETS_DIR / "librispeech_sample",
        "dataset_source": "hf_audio",
        "note": "Downloads Whisper-base + LibriSpeech sample",
    },
}


def ensure_model_and_dataset(model_name: str, logger):
    cfg = MODEL_CONFIG[model_name]
    logger.info(f"📁 Checking model files for {model_name}...")
    logger.info(f"   Note: {cfg['note']}")

    model_dir: Path = cfg["model_dir"]
    dataset_dir: Path = cfg["dataset_dir"]

    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    _ensure_model(model_name, cfg, logger)
    _ensure_dataset(model_name, cfg, logger)


def _ensure_model(model_name: str, cfg: dict, logger):
    model_dir: Path = cfg["model_dir"]
    sentinel = model_dir / ".downloaded"

    if sentinel.exists():
        logger.info(f"   ✅ Model already downloaded: {model_dir}")
        return

    hf_repo = cfg.get("hf_repo")

    if hf_repo is None:
        # torchvision handles download at runtime
        logger.info(f"   ℹ️  Model will be loaded via torchvision/torchrec at runtime")
        sentinel.touch()
        return

    logger.info(f"   ⬇️  Downloading model from HuggingFace: {hf_repo}")

    if model_name == "yolov11":
        _download_yolo(cfg, model_dir, logger)
    elif model_name == "stable_diffusion":
        _download_hf_pipeline(hf_repo, model_dir, logger)
    elif model_name == "whisper":
        _download_hf_model(hf_repo, model_dir, logger)
    else:
        logger.info(f"   ℹ️  Model will be loaded at runtime from {hf_repo}")

    sentinel.touch()
    logger.info(f"   ✅ Model ready: {model_dir}")


def _ensure_dataset(model_name: str, cfg: dict, logger):
    dataset_dir: Path = cfg["dataset_dir"]
    sentinel = dataset_dir / ".downloaded"

    if sentinel.exists():
        logger.info(f"   ✅ Dataset already present: {dataset_dir}")
        return

    source = cfg.get("dataset_source", "synthetic")
    logger.info(f"   ⬇️  Preparing dataset ({source}): {dataset_dir}")

    if source == "synthetic":
        _create_synthetic_marker(dataset_dir, model_name)
    elif source == "builtin":
        _create_builtin_marker(dataset_dir, model_name)
    elif source == "hf_audio":
        _download_hf_audio_sample(dataset_dir, logger)
    else:
        _create_synthetic_marker(dataset_dir, model_name)

    sentinel.touch()
    logger.info(f"   ✅ Dataset ready: {dataset_dir}")


def _download_yolo(cfg: dict, model_dir: Path, logger):
    model_file = cfg.get("model_file", "yolo11n.pt")
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_file}"
    dest = model_dir / model_file
    if not dest.exists():
        logger.info(f"   Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook(logger))
        except Exception as e:
            logger.warning(f"   ⚠️  Could not download YOLO weights: {e}. Will use random init.")


def _download_hf_pipeline(repo: str, model_dir: Path, logger):
    try:
        from diffusers import StableDiffusionPipeline
        logger.info(f"   Downloading diffusers pipeline {repo} → {model_dir} ...")
        pipe = StableDiffusionPipeline.from_pretrained(repo, cache_dir=str(model_dir))
        logger.info("   Pipeline cached.")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not pre-download SD pipeline: {e}. Will download at load time.")


def _download_hf_model(repo: str, model_dir: Path, logger):
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info(f"   Downloading Whisper {repo} → {model_dir} ...")
        WhisperProcessor.from_pretrained(repo, cache_dir=str(model_dir))
        WhisperForConditionalGeneration.from_pretrained(repo, cache_dir=str(model_dir))
        logger.info("   Whisper cached.")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not pre-download Whisper: {e}. Will download at load time.")


def _download_hf_audio_sample(dataset_dir: Path, logger):
    """Download a small LibriSpeech sample via datasets library."""
    try:
        from datasets import load_dataset
        logger.info("   Downloading LibriSpeech sample (librispeech_asr 'clean' split 10 samples)...")
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_demo",
            "clean",
            split="validation",
            cache_dir=str(dataset_dir),
            trust_remote_code=True,
        )
        logger.info(f"   Downloaded {len(ds)} audio samples.")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not download LibriSpeech sample: {e}. Will use synthetic audio.")
        _create_synthetic_marker(dataset_dir, "whisper")


def _create_synthetic_marker(dataset_dir: Path, model_name: str):
    info = dataset_dir / "dataset_info.txt"
    info.write_text(f"Synthetic dataset for {model_name}. Data generated at runtime.\n")


def _create_builtin_marker(dataset_dir: Path, model_name: str):
    info = dataset_dir / "dataset_info.txt"
    info.write_text(f"Built-in prompts dataset for {model_name}.\n")


def _progress_hook(logger):
    last_pct = [0]
    def hook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = int(count * block_size * 100 / total_size)
        if pct >= last_pct[0] + 10:
            logger.info(f"     {pct}%...")
            last_pct[0] = pct
    return hook