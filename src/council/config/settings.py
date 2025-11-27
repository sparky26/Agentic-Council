
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---- Model configuration ----------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for a single LLM model.

    This is intentionally generic so it can be used for Ollama-hosted models or others.
    """
    name: str
    max_completion_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    reasoning_effort: Optional[str] = None  # e.g. "medium"
    stream: bool = True


# ---- Application-wide settings ---------------------------------------------


@dataclass
class Settings:
    """
    Application-wide configuration for the Council.

    This should not depend on any Ollama SDK imports. It only stores data.
    """

    # --- LLM host configuration ---
    ollama_host: str

    # --- Model choices ---
    default_model_alias: str
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # --- Council configuration ---
    council_roles: List[str] = field(default_factory=list)

    # Optional: logging / debugging flags
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Build settings from environment variables.

        - OLLAMA_HOST         : optional (defaults to http://localhost:11434)
        - COUNCIL_DEBUG       : optional ("1"/"true" to enable)
        - COUNCIL_DEFAULT_MODEL_ALIAS : optional override for default model alias
        """
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        debug_env = os.getenv("COUNCIL_DEBUG", "").lower()
        debug = debug_env in {"1", "true", "yes", "on"}

        # You can add more models here later if you want.
        models: Dict[str, ModelConfig] = {
            # High-capacity, long outputs model:
            "gpt_oss_latest": ModelConfig(
                name="gpt-oss:latest",
                max_completion_tokens=1024,
                temperature=1.0,
                top_p=1.0,
                stream=True,
            ),
        }

        default_model_alias = os.getenv(
            "COUNCIL_DEFAULT_MODEL_ALIAS",
            "gpt_oss_latest",  # sensible default for local debates
        )

        # Roles are kept as plain strings – we’ll use them to map to prompts/agents.
        council_roles: List[str] = [
            "indian_historian",
            "civilizational_historian",
            "religion_expert",
            "anthropology_expert",
            "policymaker_expert",
        ]

        return cls(
            ollama_host=ollama_host,
            default_model_alias=default_model_alias,
            models=models,
            council_roles=council_roles,
            debug=debug,
        )


# ---- Lazy global accessor (DI-friendly) ------------------------------------

# Internal singleton cache. We avoid constructing Settings at import time so
# tests or tools can control the environment first.
_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Access the global Settings instance.

    Using a function instead of a module-level variable:
    - plays nicely with tests
    - avoids import-time failures if env is not ready
    """
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


def override_settings(new_settings: Settings) -> None:
    """
    Allow tests or special environments to override settings at runtime.
    """
    global _SETTINGS
    _SETTINGS = new_settings
