"""Whisper handler — Speech Recognition"""

import torch
import numpy as np
from pathlib import Path
from models.base import BaseModelHandler

WHISPER_CACHE_DIR = Path("data/models/whisper")
WHISPER_REPO = "openai/whisper-base"
AUDIO_DATASET_DIR = Path("data/datasets/librispeech_sample")

SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 5  # seconds of synthetic or real audio per sample


class WhisperHandler(BaseModelHandler):
    def load(self):
        self.logger.info("  Loading Whisper-base (transformers)...")
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers>=4.36")

        self.processor = WhisperProcessor.from_pretrained(
            WHISPER_REPO,
            cache_dir=str(WHISPER_CACHE_DIR),
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_REPO,
            cache_dir=str(WHISPER_CACHE_DIR),
            torch_dtype=self.dtype,
        )
        self.model.eval()
        self.model.to(self.device)
        self.logger.info("  ✅ Whisper-base loaded")

    def prepare_data(self):
        """Precompute NUM_PRECOMPUTED_BATCHES mel feature batches."""
        real_audio = self._load_real_audio()

        self._batches = []
        for i in range(self.NUM_PRECOMPUTED_BATCHES):
            if real_audio:
                # Cycle through real samples with a different offset per batch
                offset = (i * self.batch_size) % len(real_audio)
                indices = [(offset + j) % len(real_audio) for j in range(self.batch_size)]
                samples = [real_audio[k] for k in indices]
            else:
                samples = self._generate_synthetic_audio()
            self._batches.append(self._encode_audio(samples))

        source = "real audio" if real_audio else "synthetic audio"
        self.logger.info(f"  ✅ {self.NUM_PRECOMPUTED_BATCHES} batches ready ({source}): {tuple(self._batches[0].shape)}")

    def _load_real_audio(self):
        """Try to load real audio from HF dataset cache."""
        try:
            from datasets import load_dataset
            sentinel = AUDIO_DATASET_DIR / ".downloaded"
            if not sentinel.exists():
                return []

            try:
                ds = load_dataset(
                    "hf-internal-testing/librispeech_asr_demo",
                    "clean",
                    split="validation",
                    cache_dir=str(AUDIO_DATASET_DIR),
                    trust_remote_code=True,
                )
                samples = []
                for i in range(len(ds)):
                    audio = ds[i]["audio"]
                    arr = np.array(audio["array"], dtype=np.float32)
                    sr = audio["sampling_rate"]
                    if sr != SAMPLE_RATE:
                        factor = SAMPLE_RATE / sr
                        new_len = int(len(arr) * factor)
                        arr = np.interp(
                            np.linspace(0, len(arr) - 1, new_len),
                            np.arange(len(arr)),
                            arr,
                        ).astype(np.float32)
                    target_len = SAMPLE_RATE * AUDIO_DURATION_SEC
                    if len(arr) > target_len:
                        arr = arr[:target_len]
                    else:
                        arr = np.pad(arr, (0, target_len - len(arr)))
                    samples.append(arr)
                self.logger.info(f"  ✅ Loaded {len(samples)} real audio samples")
                return samples
            except Exception:
                return []
        except ImportError:
            return []

    def _generate_synthetic_audio(self):
        """Generate synthetic speech-like audio (band-limited noise)."""
        samples = []
        target_len = SAMPLE_RATE * AUDIO_DURATION_SEC
        for _ in range(self.batch_size):
            t = np.linspace(0, AUDIO_DURATION_SEC, target_len)
            audio = sum(
                0.1 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))
                for freq in [200, 400, 800, 1600]
            )
            audio = audio.astype(np.float32)
            audio = audio / (np.abs(audio).max() + 1e-8) * 0.5
            samples.append(audio)
        return samples

    def _encode_audio(self, audio_samples: list):
        inputs = self.processor(
            audio_samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return inputs.input_features.to(self.device, dtype=self.dtype)

    def run_inference(self):
        features = self._batches[self._next_batch_idx()]
        with torch.no_grad():
            _ = self.model.generate(
                features,
                max_new_tokens=128,
                language="en",
                task="transcribe",
            )
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()

    def cleanup(self):
        if hasattr(self, "processor"):
            del self.processor
        super().cleanup()
