from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional, Any, List

from groq import Groq  # type: ignore

from council.config.settings import Settings
from council.llm.base_client import ChatMessage, LLMClient
from council.config.settings import ModelConfig


class GroqLLMClient(LLMClient):
    """
    Groq-based implementation of LLMClient.

    It:
    - takes logical model aliases (e.g. "gpt_oss_120b")
    - looks up provider-specific model names in Settings.models
    - calls Groq's chat.completions endpoint
    """

    def __init__(
        self,
        *,
        api_key: str,
        models: Dict[str, ModelConfig],
        default_model_alias: str,
    ) -> None:
        self._client = Groq(api_key=api_key)
        self._models = models
        self._default_model_alias = default_model_alias

    # ---- Public factory helpers ---------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> "GroqLLMClient":
        return cls(
            api_key=settings.groq_api_key,
            models=settings.models,
            default_model_alias=settings.default_model_alias,
        )

    # ---- Internal helpers ----------------------------------------------------

    def _resolve_model_config(
        self,
        model_alias: Optional[str],
    ) -> ModelConfig:
        alias = model_alias or self._default_model_alias
        try:
            return self._models[alias]
        except KeyError as exc:
            raise KeyError(
                f"Unknown model alias '{alias}'. "
                f"Known aliases: {', '.join(sorted(self._models.keys()))}"
            ) from exc

    @staticmethod
    def _to_groq_messages(messages: Iterable[ChatMessage]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _build_payload(
        self,
        messages: Iterable[ChatMessage],
        model_alias: Optional[str],
        overrides: Dict[str, Any],
        *,
        force_stream: Optional[bool] = None,
    ) -> Dict[str, Any]:
        cfg = self._resolve_model_config(model_alias)
        groq_messages = self._to_groq_messages(messages)

        # Start with config defaults
        payload: Dict[str, Any] = {
            "model": cfg.name,
            "messages": groq_messages,
            "temperature": cfg.temperature,
            "max_completion_tokens": cfg.max_completion_tokens,
            "top_p": cfg.top_p,
            "stream": cfg.stream if force_stream is None else force_stream,
        }

        # Optional reasoning_effort (only if configured or overridden)
        if cfg.reasoning_effort is not None:
            payload["reasoning_effort"] = cfg.reasoning_effort

        # Apply per-call overrides (e.g. temperature=0.2, stream=False, etc.)
        payload.update(overrides)

        return payload

    # ---- LLMClient implementation -------------------------------------------

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        model_alias: Optional[str] = None,
        **overrides: Any,
    ) -> str:
        """
        Non-streaming completion.

        Returns a full assistant message string. If the provider returns
        multiple choices, we just take the first one.
        """
        # Ensure stream is disabled for this method
        payload = self._build_payload(
            messages,
            model_alias,
            overrides,
            force_stream=False,
        )

        completion = self._client.chat.completions.create(**payload)
        # Groq response is similar to OpenAI: choices[0].message.content
        if not completion.choices:
            return ""
        content = completion.choices[0].message.content or ""
        return content

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        model_alias: Optional[str] = None,
        **overrides: Any,
    ) -> Iterator[str]:
        """
        Streaming completion.

        Yields text chunks from choices[0].delta.content for each update.
        """
        payload = self._build_payload(
            messages,
            model_alias,
            overrides,
            force_stream=True,
        )

        completion_stream = self._client.chat.completions.create(**payload)
        for chunk in completion_stream:
            # Each chunk has choices[0].delta.content in streaming mode
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None) or ""
            if text:
                yield text
            