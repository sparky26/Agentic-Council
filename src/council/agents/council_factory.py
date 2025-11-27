from __future__ import annotations

from typing import Dict, List, Type

from council.agents.base_agent import BaseAgent
from council.agents.indian_historian import IndianHistorianAgent
from council.agents.civilizational_historian import CivilizationalHistorianAgent
from council.agents.religion_expert import ReligionExpertAgent
from council.agents.anthropology_expert import AnthropologyExpertAgent
from council.agents.policymaker_expert import PolicymakerExpertAgent

from council.config.settings import get_settings
from council.llm.base_client import LLMClient
from council.llm.model_registry import get_llm_client


# Map from role_id to concrete agent class
_ROLE_TO_AGENT_CLS: Dict[str, Type[BaseAgent]] = {
    "indian_historian": IndianHistorianAgent,
    "civilizational_historian": CivilizationalHistorianAgent,
    "religion_expert": ReligionExpertAgent,
    "anthropology_expert": AnthropologyExpertAgent,
    "policymaker_expert": PolicymakerExpertAgent,
}

# Optional: per-role default model aliases for diversity.
# These aliases must exist in Settings.models; otherwise we'll fall back
# to the global default model.
_ROLE_MODEL_DEFAULTS: Dict[str, str] = {
    "indian_historian": "gpt_oss_latest",
    "civilizational_historian": "gpt_oss_latest",
    "religion_expert": "gpt_oss_latest",
    "anthropology_expert": "gpt_oss_latest",
    "policymaker_expert": "gpt_oss_latest",
}


def _resolve_model_alias(role_id: str) -> str:
    """
    Resolve the preferred model alias for a role, falling back to the
    application's default model if necessary.
    """
    settings = get_settings()

    alias = _ROLE_MODEL_DEFAULTS.get(role_id, settings.default_model_alias)
    if alias not in settings.models:
        # Safety net: if config is stale, just use default
        return settings.default_model_alias
    return alias


def create_council(
    *,
    llm_client: LLMClient | None = None,
    roles: List[str] | None = None,
) -> List[BaseAgent]:
    """
    Build a list of agents representing the council.

    - If `roles` is None, use Settings.council_roles.
    - If `llm_client` is None, use the shared client from model_registry.

    Returns a list of BaseAgent instances in the order of `roles`.
    """
    settings = get_settings()
    llm = llm_client or get_llm_client()
    roles = roles or settings.council_roles

    council: List[BaseAgent] = []

    for role_id in roles:
        agent_cls = _ROLE_TO_AGENT_CLS.get(role_id)
        if agent_cls is None:
            raise ValueError(
                f"No agent class registered for role_id '{role_id}'. "
                f"Known roles: {', '.join(sorted(_ROLE_TO_AGENT_CLS.keys()))}"
            )

        model_alias = _resolve_model_alias(role_id)
        agent = agent_cls(llm, model_alias=model_alias)
        council.append(agent)

    return council