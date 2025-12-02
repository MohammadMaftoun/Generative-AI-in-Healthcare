# ============================================================================
# FILE: safety_filter.py
# Safety validation and filtering system
# ============================================================================

import re
from typing import Tuple
from config import Config

class SafetyFilter:
    """Validates and filters prompts and outputs for safety compliance."""
    
    def __init__(self, mode: str = "strict"):
        """
        Initialize safety filter.
        
        Args:
            mode: Safety mode (strict, moderate, permissive)
        """
        self.mode = mode
        self.blocked_terms = Config.BLOCKED_TERMS
        
    def validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        """
        Validate that a prompt is safe and synthetic.
        
        Args:
            prompt: The prompt to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        prompt_lower = prompt.lower()
        
        # Check for blocked terms
        for term in self.blocked_terms:
            if term.lower() in prompt_lower:
                return False, f"Prompt contains blocked term: '{term}'"
        
        # Check for potential patient identifiers (patterns)
        patterns = [
            r'\\b\\d{3}-\\d{2}-\\d{4}\\b',  # SSN
            r'\\b\\d{10}\\b',  # Phone number
            r'\\b[A-Z]{2}\\d{6}\\b',  # MRN-like pattern
        ]
        
        for pattern in patterns:
            if re.search(pattern, prompt):
                return False, "Prompt contains potential identifier pattern"
        
        # Ensure "synthetic" or "simulated" is mentioned
        if self.mode == "strict":
            if "synthetic" not in prompt_lower and "simulated" not in prompt_lower:
                return False, "Prompt must explicitly indicate synthetic nature"
        
        return True, ""
    
    def sanitize_output(self, text: str) -> str:
        """
        Sanitize output text by removing any problematic content.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove any potential identifiers
        text = re.sub(r'\\b\\d{3}-\\d{2}-\\d{4}\\b', '[REDACTED]', text)
        text = re.sub(r'\\b\\d{10}\\b', '[REDACTED]', text)
        
        return text
    
    def add_watermark_note(self, prompt: str) -> str:
        """
        Add safety watermark to prompt.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Prompt with watermark
        """
        watermark = "\\n\\nIMPORTANT: This is a fully synthetic, artificial medical image for research purposes only. Not for diagnostic use."
        return prompt + watermark
