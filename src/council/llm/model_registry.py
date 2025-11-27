from __future__ import annotations

from typing import Dict

from council.config.settings import ModelConfig, get_settings
from council.llm.base_client import LLMClient
from council.llm.ollama_client import OllamaLLMClient


# Simple global cache so we don't recreate clients everywhere
_LLM_CLIENT: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """
    Return the default LLMClient for the application.

    Currently this is an OllamaLLMClient, but higher-level code should depend on
    the LLMClient interface, not on the concrete type.
    """
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        settings = get_settings()
        _LLM_CLIENT = OllamaLLMClient.from_settings(settings)
    return _LLM_CLIENT


def get_model_config(alias: str) -> ModelConfig:
    """
    Convenience access to model configuration for a given alias.
    """
    settings = get_settings()
    try:
        return settings.models[alias]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model alias '{alias}'. "
            f"Known aliases: {', '.join(sorted(settings.models.keys()))}"
        ) from exc


def list_models() -> Dict[str, ModelConfig]:
    """
    Return a copy of the mapping of alias -> ModelConfig.
    """
    settings = get_settings()
    return dict(settings.models)
