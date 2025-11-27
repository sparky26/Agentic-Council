from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Any, List

from council.llm.base_client import ChatMessage, LLMClient


@dataclass(frozen=True)
class AgentConfig:
    """
    Static configuration for an agent.

    - name: human-readable label ("Indian Historian")
    - role_id: stable internal id ("indian_historian")
    - model_alias: which logical model to use by default (may be None to let
      the client pick its default).
    """
    name: str
    role_id: str
    model_alias: Optional[str] = None


class BaseAgent(ABC):
    """
    Base class for all council agents.

    Responsibilities:
    - hold identity/config information
    - hold the role-specific system prompt
    - provide simple helpers to call the underlying LLM client

    It does NOT:
    - know the debate protocol
    - orchestrate turn-taking
    - store global conversation state

    The orchestrator will manage conversation and simply ask each agent to
    respond given the current list of ChatMessage objects.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        system_prompt: str,
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._system_prompt = system_prompt

    # ---- Public identity properties ----------------------------------------

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def role_id(self) -> str:
        return self._config.role_id

    @property
    def model_alias(self) -> Optional[str]:
        return self._config.model_alias

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    # ---- Internal helper ----------------------------------------------------

    def _with_system_message(
        self,
        conversation: Iterable[ChatMessage],
    ) -> List[ChatMessage]:
        """
        Ensure a system message with the agent's role prompt is the first
        message in the sequence.

        If there's already a system message with the same content at the front,
        we avoid duplicating it.
        """
        conv_list = list(conversation)
        if (
            conv_list
            and conv_list[0].role == "system"
            and conv_list[0].content == self._system_prompt
        ):
            return conv_list

        return [
            ChatMessage(role="system", content=self._system_prompt),
            *conv_list,
        ]

    # ---- Core interface: respond -------------------------------------------

    def respond(
        self,
        conversation: Iterable[ChatMessage],
        *,
        model_alias: Optional[str] = None,
        **overrides: Any,
    ) -> str:
        """
        Synchronous (non-streaming) response.

        - `conversation` is the current list of ChatMessage objects (all roles).
        - `model_alias` optionally overrides this agent's default model choice.
        - `overrides` can pass per-call parameters (temperature, etc.).

        Returns the full assistant message content.
        """
        messages = self._with_system_message(conversation)
        alias = model_alias or self._config.model_alias

        return self._llm.complete(
            messages,
            model_alias=alias,
            **overrides,
        )

    def respond_stream(
        self,
        conversation: Iterable[ChatMessage],
        *,
        model_alias: Optional[str] = None,
        **overrides: Any,
    ) -> Iterator[str]:
        """
        Streaming response.

        Yields text chunks from the underlying LLM client.
        """
        messages = self._with_system_message(conversation)
        alias = model_alias or self._config.model_alias

        return self._llm.stream(
            messages,
            model_alias=alias,
            **overrides,
        )