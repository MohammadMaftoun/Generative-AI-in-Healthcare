# ============================================================================
# FILE: config.py
# Global configuration for synthetic medical imaging system
# ============================================================================

import os
from pathlib import Path
from typing import Dict, List

class Config:
    """Global configuration for the synthetic medical imaging system."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Model settings
    STABLE_DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-1-base"
    STABLE_DIFFUSION_PRECISION = "fp16"  # fp16 or fp32
    
    # Image generation settings
    DEFAULT_RESOLUTION = 512
    SUPPORTED_RESOLUTIONS = [512, 768, 1024]
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 7.5
    
    # Modality configurations
    MODALITIES: Dict[str, Dict] = {
        "xray": {
            "name": "X-ray",
            "technical_name": "radiograph",
            "description": "2D radiographic image",
            "typical_artifacts": ["scatter", "noise", "grid lines"],
        },
        "mri": {
            "name": "MRI",
            "technical_name": "magnetic resonance imaging",
            "description": "MRI scan with soft tissue detail",
            "typical_artifacts": ["motion blur", "ringing", "aliasing"],
        },
        "ct": {
            "name": "CT",
            "technical_name": "computed tomography",
            "description": "CT scan cross-section",
            "typical_artifacts": ["beam hardening", "ring artifacts", "noise"],
        }
    }
    
    # Body regions
    BODY_REGIONS: List[str] = [
        "chest", "brain", "abdomen", "spine", "pelvis",
        "knee", "shoulder", "hand", "foot", "skull"
    ]
    
    # Detail levels
    DETAIL_LEVELS: Dict[str, Dict] = {
        "low": {
            "description": "basic anatomical structures visible",
            "complexity": "simple",
            "num_inference_steps": 30,
        },
        "medium": {
            "description": "clear anatomical detail with some texture",
            "complexity": "moderate",
            "num_inference_steps": 50,
        },
        "high": {
            "description": "highly detailed with realistic tissue texture and artifacts",
            "complexity": "complex",
            "num_inference_steps": 75,
        }
    }
    
    # Imaging angles/views
    IMAGING_ANGLES: Dict[str, List[str]] = {
        "xray": ["AP", "lateral", "oblique", "PA"],
        "mri": ["axial", "sagittal", "coronal", "oblique"],
        "ct": ["axial", "coronal", "sagittal", "3D reconstruction"]
    }
    
    # LLM settings
    LLM_PROVIDERS = ["claude", "gpt", "huggingface"]
    DEFAULT_LLM_PROVIDER = "claude"
    LLM_MAX_TOKENS = 500
    LLM_TEMPERATURE = 0.7
    
    # Safety settings
    SAFETY_MODE = "strict"  # strict, moderate, permissive
    WATERMARK_TEXT = "SYNTHETIC - NOT FOR DIAGNOSTIC USE"
    
    # Blocked terms for safety (no real patient identifiers)
    BLOCKED_TERMS = [
        "patient", "hospital", "medical record", "mrn", "dob",
        "social security", "name", "address", "phone"
    ]
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        
        for modality in cls.MODALITIES.keys():
            (cls.OUTPUT_DIR / modality).mkdir(exist_ok=True)
