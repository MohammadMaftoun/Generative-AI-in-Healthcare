# ============================================================================
# FILE: llm_wrapper.py
# Universal LLM interface supporting multiple providers
# ============================================================================

import os
from typing import Optional, Dict
from config import Config

class LLMWrapper:
    """
    Universal interface for multiple LLM providers.
    Supports Claude (Anthropic), GPT (OpenAI), and HuggingFace models.
    """
    
    def __init__(self, provider: str = "claude"):
        """
        Initialize LLM wrapper.
        
        Args:
            provider: LLM provider name (claude, gpt, huggingface)
        """
        self.provider = provider.lower()
        
        if self.provider not in Config.LLM_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self._setup_client()
    
    def _setup_client(self):
        """Setup the appropriate client based on provider."""
        if self.provider == "claude":
            try:
                from anthropic import Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")
                self.client = Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        elif self.provider == "gpt":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        
        elif self.provider == "huggingface":
            try:
                from transformers import pipeline
                # Use a small, fast model for demo
                self.client = pipeline(
                    "text-generation",
                    model="gpt2-medium",
                    device=-1  # CPU
                )
            except ImportError:
                raise ImportError("transformers package not installed. Run: pip install transformers")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if self.provider == "claude":
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif self.provider == "gpt":
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        elif self.provider == "huggingface":
            result = self.client(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                truncation=True
            )
            return result[0]['generated_text'][len(prompt):].strip()
