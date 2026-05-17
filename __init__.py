"""LiteLLM proxy provider profile.

LiteLLM (https://github.com/BerriAI/litellm) is a unified API proxy that
aggregates many LLM providers behind an OpenAI-compatible interface.

This profile keeps Hermes aligned with OpenCode's LiteLLM model discovery:
  - fetch live chat models from LiteLLM's richer ``/v1/model/info`` endpoint
  - skip non-chat entries (embeddings, image generation, etc.) by default
  - use LiteLLM model metadata to decide whether ``reasoning_effort`` is safe
  - preserve LiteLLM's supported effort names, including ``minimal``
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


_CACHE_TTL_SECONDS = 300


class LiteLLMProfile(ProviderProfile):
    """LiteLLM proxy provider profile."""

    _EFFORT_MAP: dict[str, str] = {
        "none": "none",
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }

    _model_info_cache: dict[str, Any] | None = None
    _model_info_cache_at: float = 0.0

    @staticmethod
    def _has_usable_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and value > 0

    @staticmethod
    def _looks_non_chat_model(model_name: str) -> bool:
        name = model_name.lower()
        return any(token in name for token in ("embedding", "embed", "dall-e", "flux", "sdxl"))

    def _base_url(self) -> str:
        return (
            os.getenv("LITELLM_BASE_URL")
            or self.base_url
            or os.getenv("OPENAI_BASE_URL")
            or ""
        ).rstrip("/")

    def _request_json(self, url: str, api_key: str | None, timeout: float) -> Any | None:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        for key, value in self.default_headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return None

    def _fetch_model_info_entries(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[dict[str, Any]]:
        api_key = api_key or os.getenv("LITELLM_API_KEY") or None
        base_url = self._base_url()
        if not base_url:
            return []
        cache_key = f"{base_url}|{bool(api_key)}"
        now = time.time()
        if (
            self._model_info_cache
            and self._model_info_cache.get("key") == cache_key
            and now - self._model_info_cache_at < _CACHE_TTL_SECONDS
        ):
            return list(self._model_info_cache.get("entries") or [])

        server_url = base_url[:-3].rstrip("/") if base_url.endswith("/v1") else base_url
        entries: list[dict[str, Any]] = []
        for path in ("/v1/model/info", "/model/info"):
            payload = self._request_json(f"{server_url}{path}", api_key, timeout)
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                entries = [entry for entry in data if isinstance(entry, dict)]
                break

        self._model_info_cache = {"key": cache_key, "entries": entries}
        self._model_info_cache_at = now
        return entries

    def _model_info_score(self, model_id: str, entry: dict[str, Any]) -> int:
        info = entry.get("model_info") if isinstance(entry.get("model_info"), dict) else {}
        params = entry.get("litellm_params") if isinstance(entry.get("litellm_params"), dict) else {}
        model_name = entry.get("model_name")
        info_key = info.get("key")
        routed_model = params.get("model")
        model_id_lower = model_id.lower()
        score = 0
        if model_name == model_id:
            score += 8
        if info_key == model_id:
            score += 6
        if routed_model == model_id:
            score += 4
        if isinstance(routed_model, str) and routed_model.endswith(f"/{model_id}"):
            score += 2
        if isinstance(model_name, str) and model_name.lower() == model_id_lower:
            score += 3
        if isinstance(info_key, str) and info_key.lower() == model_id_lower:
            score += 2
        if isinstance(routed_model, str) and routed_model.lower() == model_id_lower:
            score += 2
        if isinstance(routed_model, str) and routed_model.lower().endswith(f"/{model_id_lower}"):
            score += 1
        if info.get("mode") == "chat":
            score += 5
        if self._has_usable_number(info.get("max_input_tokens")) or self._has_usable_number(info.get("max_tokens")):
            score += 10
        if info.get("supports_reasoning") is True:
            score += 4
        return score

    def _build_model_info_map(self, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for entry in entries:
            info = entry.get("model_info") if isinstance(entry.get("model_info"), dict) else {}
            params = entry.get("litellm_params") if isinstance(entry.get("litellm_params"), dict) else {}
            keys: set[str] = set()
            for value in (entry.get("model_name"), info.get("key"), params.get("model")):
                if isinstance(value, str) and value:
                    keys.add(value)
            routed_model = params.get("model")
            if isinstance(routed_model, str) and routed_model:
                parts = routed_model.split("/")
                if len(parts) > 1:
                    keys.add("/".join(parts[1:]))
                keys.add(parts[-1])

            for key in keys:
                for lookup_key in {key, key.lower()}:
                    existing = result.get(lookup_key)
                    if not existing or self._model_info_score(lookup_key, entry) > self._model_info_score(lookup_key, existing):
                        result[lookup_key] = entry
        return result

    def _model_info_for(
        self,
        model: str | None,
        *,
        api_key: str | None = None,
    ) -> dict[str, Any] | None:
        if not model:
            return None
        entries = self._fetch_model_info_entries(api_key=api_key, timeout=5.0)
        if not entries:
            return None
        info_map = self._build_model_info_map(entries)
        return info_map.get(model) or info_map.get(model.lower())

    def _supports_reasoning_effort(self, model: str | None, api_key: str | None) -> bool:
        entry = self._model_info_for(model, api_key=api_key)
        info = entry.get("model_info") if isinstance(entry, dict) and isinstance(entry.get("model_info"), dict) else None
        if not info:
            return False
        params = info.get("supported_openai_params")
        return info.get("supports_reasoning") is True and isinstance(params, list) and "reasoning_effort" in params

    def _effort_supported(self, effort: str, model: str | None, api_key: str | None) -> bool:
        entry = self._model_info_for(model, api_key=api_key)
        info = entry.get("model_info") if isinstance(entry, dict) and isinstance(entry.get("model_info"), dict) else None
        if not info:
            return False
        if effort == "medium" or effort == "high":
            return True
        if effort == "low":
            return info.get("supports_low_reasoning_effort") is not False
        return info.get(f"supports_{effort}_reasoning_effort") is True

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (extra_body_additions, top_level_kwargs)."""
        if not reasoning_config or not isinstance(reasoning_config, dict):
            return {}, {}

        api_key = os.getenv("LITELLM_API_KEY") or None
        if not self._supports_reasoning_effort(model, api_key):
            return {}, {}

        enabled = reasoning_config.get("enabled", True)
        raw_effort = reasoning_config.get("effort") or "medium"
        effort_raw = str(raw_effort).strip().lower()
        if enabled is False:
            effort_raw = "none"

        effort = self._EFFORT_MAP.get(effort_raw, "medium")
        if not self._effort_supported(effort, model, api_key):
            if effort == "none":
                return {}, {}
            effort = "medium"

        return {}, {"reasoning_effort": effort}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch chat-capable model IDs from LiteLLM.

        Prefer ``/v1/model/info`` so non-chat models can be skipped. Fall back to
        OpenAI-compatible ``/v1/models`` when LiteLLM metadata is unavailable.
        """
        api_key = api_key or os.getenv("LITELLM_API_KEY") or None
        entries = self._fetch_model_info_entries(api_key=api_key, timeout=timeout)
        if entries:
            models: list[str] = []
            seen: set[str] = set()
            for entry in entries:
                info = entry.get("model_info") if isinstance(entry.get("model_info"), dict) else {}
                mode = info.get("mode")
                if isinstance(mode, str) and mode and mode != "chat":
                    continue
                model_name = entry.get("model_name")
                if (
                    isinstance(model_name, str)
                    and model_name
                    and model_name not in seen
                    and not self._looks_non_chat_model(model_name)
                ):
                    seen.add(model_name)
                    models.append(model_name)
            return models or None

        base_url = self._base_url()
        if not base_url:
            return None
        url = (self.models_url or "").strip() or f"{base_url.rstrip('/')}/models"
        payload = self._request_json(url, api_key, timeout)
        data = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(data, list):
            return None
        models = [
            item.get("id")
            for item in data
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and not self._looks_non_chat_model(item.get("id"))
        ]
        return models or None


litellm = LiteLLMProfile(
    name="litellm",
    aliases=("litellm-proxy", "llm-proxy"),
    display_name="LiteLLM Proxy",
    description="LiteLLM — unified API proxy for 100+ LLM providers",
    signup_url="https://docs.litellm.ai/docs/",
    base_url=os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1"),
    env_vars=("LITELLM_API_KEY",),
    supports_health_check=False,
    default_aux_model="",
)

register_provider(litellm)
