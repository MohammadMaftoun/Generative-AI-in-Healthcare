# ============================================================================
# FILE: image_generator.py
# Stable Diffusion pipeline for medical image generation
# ============================================================================

import torch
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
from config import Config

class MedicalImageGenerator:
    """Generates synthetic medical images using Stable Diffusion."""
    
    def __init__(
        self,
        model_id: str = None,
        precision: str = "fp16",
        scheduler: str = "dpm"
    ):
        """
        Initialize image generator.
        
        Args:
            model_id: HuggingFace model ID
            precision: fp16 or fp32
            scheduler: Scheduler type (dpm, euler)
        """
        self.model_id = model_id or Config.STABLE_DIFFUSION_MODEL
        self.precision = precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing Stable Diffusion on {self.device}...")
        print(f"Model: {self.model_id}")
        print(f"Precision: {self.precision}")
        
        # Load pipeline
        self._load_pipeline(scheduler)
        
        print("✓ Model loaded successfully")
    
    def _load_pipeline(self, scheduler: str):
        """Load Stable Diffusion pipeline."""
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=Config.MODELS_DIR
        )
        
        # Set scheduler
        if scheduler == "dpm":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
        elif scheduler == "euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Enable optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        resolution: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate synthetic medical images.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_images: Number of images to generate
            resolution: Image resolution (width and height)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL Images
        """
        if resolution not in Config.SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {Config.SUPPORTED_RESOLUTIONS}")
        
        # Set default negative prompt
        if negative_prompt is None:
            negative_prompt = (
                "text, watermark, low quality, blurry, distorted, "
                "deformed anatomy, unrealistic, cartoon, drawing"
            )
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"\\nGenerating {num_images} image(s)...")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Steps: {num_inference_steps}")
        
        # Generate images
        images = []
        for i in range(num_images):
            print(f"  Generating image {i+1}/{num_images}...", end=" ", flush=True)
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=resolution,
                height=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            
            images.append(result.images[0])
            print("✓")
        
        return images
    
    def save_images(
        self,
        images: List[Image.Image],
        modality: str,
        region: str,
        metadata: dict
    ) -> Path:
        """
        Save generated images with metadata.
        
        Args:
            images: List of PIL Images
            modality: Imaging modality
            region: Body region
            metadata: Generation metadata
            
        Returns:
            Path to output directory
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Config.OUTPUT_DIR / modality / f"{region}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        for i, img in enumerate(images, 1):
            img_path = output_dir / f"image_{i:03d}.png"
            img.save(img_path)
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\\n✓ Saved {len(images)} image(s) to: {output_dir}")
        
        return output_dir
