"""
LLM Configuration Module
Loads provider, model, base URL, and API key configuration from environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration class for LLM client connection settings."""
    provider: str
    model: str
    base_url: Optional[str]
    api_key: Optional[str]

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        Load configuration from environment variables:
            LLM_PROVIDER
            LLM_MODEL
            LLM_BASE_URL
            LLM_API_KEY / OPENAI_API_KEY
        """
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "").strip() or None
        api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip() or None

        if provider == "lmstudio":
            model = model or "google/gemma-4-e4b"
            base_url = base_url or "http://localhost:1234/v1"
            api_key = api_key or "lm-studio"
        elif provider == "openai":
            model = model or "gpt-4o-mini"
            base_url = base_url or "https://api.openai.com/v1"
        else:
            # Detect based on available keys if LLM_PROVIDER is not explicitly specified
            if api_key and not api_key.startswith("your_"):
                provider = "openai"
                model = model or "gpt-4o-mini"
            else:
                provider = "fallback"
                model = model or "deterministic-fallback"

        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key
        )
