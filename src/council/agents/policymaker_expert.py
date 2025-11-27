from __future__ import annotations

from council.agents.base_agent import AgentConfig, BaseAgent
from council.config.prompts import get_role_system_prompt
from council.llm.base_client import LLMClient


class PolicymakerExpertAgent(BaseAgent):
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model_alias: str | None = None,
    ) -> None:
        config = AgentConfig(
            name="Policy Analyst / Policymaker",
            role_id="policymaker_expert",
            model_alias=model_alias,
        )
        system_prompt = get_role_system_prompt("policymaker_expert")
        super().__init__(config, llm_client, system_prompt)