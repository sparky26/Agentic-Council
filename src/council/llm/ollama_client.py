from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional

from ollama import Client  # type: ignore

from council.config.settings import ModelConfig, Settings
from council.llm.base_client import ChatMessage, LLMClient


class OllamaLLMClient(LLMClient):
    """
    Ollama-based implementation of LLMClient.

    It:
    - takes logical model aliases (e.g. "gpt_oss_latest")
    - looks up provider-specific model names in Settings.models
    - calls the Ollama chat endpoint on the configured host
    """

    def __init__(
        self,
        *,
        host: str,
        models: Dict[str, ModelConfig],
        default_model_alias: str,
    ) -> None:
        self._client = Client(host=host)
        self._models = models
        self._default_model_alias = default_model_alias

    # ---- Public factory helpers ---------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaLLMClient":
        return cls(
            host=settings.ollama_host,
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
    def _to_ollama_messages(messages: Iterable[ChatMessage]) -> List[Dict[str, str]]:
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
        ollama_messages = self._to_ollama_messages(messages)

        options: Dict[str, Any] = {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_predict": cfg.max_completion_tokens,
        }

        payload: Dict[str, Any] = {
            "model": cfg.name,
            "messages": ollama_messages,
            "stream": cfg.stream if force_stream is None else force_stream,
            "options": options,
        }

        # Map common overrides to Ollama options/payload.
        override_copy = dict(overrides)
        if "temperature" in override_copy:
            options["temperature"] = override_copy.pop("temperature")
        if "top_p" in override_copy:
            options["top_p"] = override_copy.pop("top_p")
        if "max_completion_tokens" in override_copy:
            options["num_predict"] = override_copy.pop("max_completion_tokens")
        if "num_predict" in override_copy:
            options["num_predict"] = override_copy.pop("num_predict")
        if force_stream is None and "stream" in override_copy:
            payload["stream"] = override_copy.pop("stream")
        if "options" in override_copy:
            user_options = override_copy.pop("options")
            if isinstance(user_options, dict):
                options.update(user_options)

        # Pass through known chat parameters if provided.
        for passthrough_key in ("format", "keep_alive", "context", "tools"):
            if passthrough_key in override_copy:
                payload[passthrough_key] = override_copy.pop(passthrough_key)

        # Ignore any other overrides to avoid unexpected errors from the client.

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
        multiple choices, we just take the content field.
        """
        payload = self._build_payload(
            messages,
            model_alias,
            overrides,
            force_stream=False,
        )

        completion = self._client.chat(**payload)
        message = completion.get("message") or {}
        content = message.get("content") or ""
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

        Yields text chunks from the incremental message content for each update.
        """
        payload = self._build_payload(
            messages,
            model_alias,
            overrides,
            force_stream=True,
        )

        completion_stream = self._client.chat(**payload)
        for chunk in completion_stream:
            message = chunk.get("message") or {}
            text = message.get("content") or ""
            if text:
                yield text
