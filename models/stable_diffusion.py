"""Stable Diffusion v1.5 handler — Text-to-Image Generation"""

import torch
from pathlib import Path
from models.base import BaseModelHandler

SD_CACHE_DIR = Path("data/models/stable_diffusion")
SD_REPO = "runwayml/stable-diffusion-v1-5"

SAMPLE_PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "cinematic shot of a futuristic city at night, neon lights, rain",
    "a beautiful landscape with mountains and a lake, sunset, oil painting",
    "a close-up portrait of a cat wearing sunglasses",
    "deep sea creatures glowing in the dark, underwater photography",
    "a robot chef cooking in a modern kitchen, digital art",
    "ancient ruins in a jungle overgrown with vegetation, dramatic lighting",
    "a wizard casting a spell in a fantasy world, epic fantasy art",
]


class StableDiffusionHandler(BaseModelHandler):
    def load(self):
        self.logger.info("  Loading Stable Diffusion v1.5 (diffusers)...")
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError("diffusers not installed. Run: pip install diffusers>=0.21 accelerate")

        torch_dtype = self.dtype if self.dtype != torch.float32 else torch.float32
        # fp16 is recommended for SD to reduce memory
        if self.dtype == torch.float32 and str(self.device) != "cpu":
            self.logger.info("  ℹ️  Consider --precision fp16 for SD to reduce VRAM usage")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD_REPO,
            torch_dtype=torch_dtype,
            cache_dir=str(SD_CACHE_DIR),
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(self.device)

        # Speed optimizations
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        self.logger.info("  ✅ Stable Diffusion loaded")

    def prepare_data(self):
        import itertools
        # Cycle through prompts for the batch
        cycle = itertools.cycle(SAMPLE_PROMPTS)
        self._prompts = [next(cycle) for _ in range(self.batch_size)]
        self.logger.info(f"  ✅ {len(SAMPLE_PROMPTS)} prompts ready, using batch of {self.batch_size}")

    def run_inference(self):
        with torch.no_grad():
            _ = self.pipe(
                self._prompts,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512,
                output_type="latent",  # skip VAE decode for speed
            )
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
        super().cleanup()