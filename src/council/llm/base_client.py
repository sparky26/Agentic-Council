from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Optional


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """
    Simple chat message model, independent of any specific LLM provider.
    """
    role: Role
    content: str


class LLMClient(ABC):
    """
    Abstract interface for any chat-based LLM client.

    Agents and orchestrators should depend on this interface, not on Groq
    or any other concrete SDK.
    """

    @abstractmethod
    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        model_alias: Optional[str] = None,
        **overrides,
    ) -> str:
        """
        Non-streaming completion.

        Returns the full assistant message content as a single string.

        `model_alias` is a logical key (e.g. "gpt_oss_120b", "llama_4_scout_17b")
        that the concrete client maps to provider-specific model names.
        `overrides` can be used for per-call settings like temperature, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        model_alias: Optional[str] = None,
        **overrides,
    ) -> Iterator[str]:
        """
        Streaming completion.

        Yields chunks of assistant content (strings). The caller is responsible
        for joining them if needed.
        """
        raise NotImplementedError