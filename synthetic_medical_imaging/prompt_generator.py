# ============================================================================
# FILE: prompt_generator.py
# Medical prompt generation using LLM
# ============================================================================

import random
from typing import Dict, Optional
from config import Config
from llm_wrapper import LLMWrapper
from safety_filter import SafetyFilter

class MedicalPromptGenerator:
    """Generates detailed synthetic medical imaging prompts using LLMs."""
    
    def __init__(self, llm_provider: str = "claude", safety_mode: str = "strict"):
        """
        Initialize prompt generator.
        
        Args:
            llm_provider: LLM provider to use
            safety_mode: Safety filter mode
        """
        self.llm = LLMWrapper(provider=llm_provider)
        self.safety = SafetyFilter(mode=safety_mode)
    
    def generate_prompt(
        self,
        modality: str,
        body_region: str,
        detail_level: str = "medium",
        angle: Optional[str] = None
    ) -> str:
        """
        Generate a detailed synthetic medical imaging prompt.
        
        Args:
            modality: Imaging modality (xray, mri, ct)
            body_region: Body region to image
            detail_level: Level of detail (low, medium, high)
            angle: Imaging angle/view (optional)
            
        Returns:
            Generated prompt string
        """
        # Validate inputs
        if modality not in Config.MODALITIES:
            raise ValueError(f"Invalid modality: {modality}")
        if body_region not in Config.BODY_REGIONS:
            raise ValueError(f"Invalid body region: {body_region}")
        if detail_level not in Config.DETAIL_LEVELS:
            raise ValueError(f"Invalid detail level: {detail_level}")
        
        # Select angle if not provided
        if angle is None:
            angle = random.choice(Config.IMAGING_ANGLES[modality])
        
        # Get configuration details
        mod_config = Config.MODALITIES[modality]
        detail_config = Config.DETAIL_LEVELS[detail_level]
        
        # Create LLM instruction
        llm_instruction = self._create_llm_instruction(
            modality, mod_config, body_region, detail_level, 
            detail_config, angle
        )
        
        # Generate enhanced prompt using LLM
        enhanced_prompt = self.llm.generate(
            llm_instruction,
            max_tokens=Config.LLM_MAX_TOKENS,
            temperature=Config.LLM_TEMPERATURE
        )
        
        # Build final prompt
        final_prompt = self._build_final_prompt(
            modality, body_region, angle, detail_level,
            enhanced_prompt, mod_config
        )
        
        # Validate safety
        is_valid, error = self.safety.validate_prompt(final_prompt)
        if not is_valid:
            raise ValueError(f"Safety validation failed: {error}")
        
        # Add watermark
        final_prompt = self.safety.add_watermark_note(final_prompt)
        
        return final_prompt
    
    def _create_llm_instruction(
        self,
        modality: str,
        mod_config: Dict,
        body_region: str,
        detail_level: str,
        detail_config: Dict,
        angle: str
    ) -> str:
        """Create instruction for LLM to generate prompt details."""
        return f\"\"\"Generate a detailed, clinically-styled description for a FULLY SYNTHETIC medical image with these specifications:

Modality: {mod_config['name']} ({mod_config['technical_name']})
Body Region: {body_region}
View/Angle: {angle}
Detail Level: {detail_level} - {detail_config['description']}
Typical Artifacts: {', '.join(mod_config['typical_artifacts'])}

Requirements:
1. Describe synthetic anatomical features appropriate for {body_region} in {modality} imaging
2. Include realistic but ARTIFICIAL imaging characteristics
3. Mention appropriate {detail_level} level anatomical details
4. Suggest 2-3 subtle simulated imaging artifacts
5. Use clinical terminology but emphasize this is SYNTHETIC
6. Keep description under 150 words
7. DO NOT reference any real patients, hospitals, or identifiable information

Output ONLY the imaging description, no preamble.\"\"\"
    
    def _build_final_prompt(
        self,
        modality: str,
        body_region: str,
        angle: str,
        detail_level: str,
        llm_output: str,
        mod_config: Dict
    ) -> str:
        \"\"\"Build final Stable Diffusion prompt from components.\"\"\"
        # Base prompt structure
        prompt_parts = [
            f\"Professional medical {mod_config['technical_name']} image,\",
            f\"{angle} view of {body_region},\",
            f\"synthetic diagnostic quality,\",
            llm_output.strip(),
            f\"high resolution medical imaging,\",
            f\"realistic {modality} appearance,\",
            f\"clinical photography style,\",
            \"artificial medical illustration,\",
            \"photorealistic rendering\"
        ]
        
        return \" \".join(prompt_parts)
