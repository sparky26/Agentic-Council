# src/council/config/__init__.py

from .settings import get_settings
from .prompts import get_role_system_prompt, get_base_debate_prompt

__all__ = ["get_settings", "get_role_system_prompt", "get_base_debate_prompt"]