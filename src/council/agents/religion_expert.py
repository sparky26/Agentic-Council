from __future__ import annotations

from council.agents.base_agent import AgentConfig, BaseAgent
from council.config.prompts import get_role_system_prompt
from council.llm.base_client import LLMClient


class ReligionExpertAgent(BaseAgent):
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model_alias: str | None = None,
    ) -> None:
        config = AgentConfig(
            name="Religion Expert",
            role_id="religion_expert",
            model_alias=model_alias,
        )
        system_prompt = get_role_system_prompt("religion_expert")
        super().__init__(config, llm_client, system_prompt)