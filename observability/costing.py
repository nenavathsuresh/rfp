"""Reusable token usage and cost tracking utilities."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.embeddings import Embeddings


@dataclass(frozen=True)
class PricingRate:
    """Per-million token pricing."""

    input_per_million: float
    output_per_million: float = 0.0


class _TrackingEmbeddings(Embeddings):
    """Embedding wrapper that records usage for each call."""

    def __init__(
        self,
        inner: Embeddings,
        tracker: "CostTracker",
        *,
        provider: str,
        model: str,
    ) -> None:
        self._inner = inner
        self._tracker = tracker
        self._provider = provider
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._inner.embed_documents(texts)
        self._tracker.record_embedding_call(
            provider=self._provider,
            model=self._model,
            operation="embed_documents",
            texts=texts,
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._inner.embed_query(text)
        self._tracker.record_embedding_call(
            provider=self._provider,
            model=self._model,
            operation="embed_query",
            texts=[text],
        )
        return vector


def wrap_embeddings_with_tracking(
    embedding_generator: Embeddings,
    tracker: "CostTracker" | None,
    *,
    provider: str,
    model: str,
) -> Embeddings:
    """Wrap embeddings with usage tracking when a tracker is provided."""

    if tracker is None:
        return embedding_generator
    return _TrackingEmbeddings(
        embedding_generator,
        tracker,
        provider=provider,
        model=model,
    )


class CostTracker:
    """Reusable token and cost tracker for LLM + embedding API calls."""

    _DEFAULT_RATES: dict[str, PricingRate] = {
        # OpenAI chat model (USD per 1M tokens)
        "openai:gpt-5.4-mini-2026-03-17": PricingRate(input_per_million=0.75, output_per_million=4.50),
        # OpenAI embedding model
        "openai:text-embedding-3-small": PricingRate(input_per_million=0.02, output_per_million=0.0),
        # Anthropic Claude Sonnet 5 introductory pricing through 2026-08-31
        "anthropic:claude-sonnet-5": PricingRate(input_per_million=2.00, output_per_million=10.00),
    }

    def __init__(
        self,
        *,
        report_name: str,
        run_type: str,
        pricing_overrides: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.report_name = report_name
        self.run_type = run_type
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._api_calls: list[dict[str, Any]] = []
        self._lock = Lock()
        self._call_counter = 0
        self._llm_call_index_by_request_key: dict[str, int] = {}
        self._seen_llm_fingerprints: set[str] = set()
        self._pricing = self._build_pricing(pricing_overrides or {})

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate tokens when provider usage metadata is unavailable."""

        cleaned = (text or "").strip()
        if not cleaned:
            return 0

        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(cleaned))
        except Exception:
            return max(1, math.ceil(len(cleaned) / 4))

    @staticmethod
    def _extract_usage_from_any(value: Any) -> dict[str, int] | None:
        usage = CostTracker._read_usage_fields(value)
        if usage:
            return usage

        for candidate in CostTracker._iter_usage_candidates(value, depth=0, seen=set()):
            candidate_usage = CostTracker._read_usage_fields(candidate)
            if candidate_usage:
                return candidate_usage
        return None

    @staticmethod
    def _iter_usage_candidates(value: Any, *, depth: int, seen: set[int]) -> list[Any]:
        if value is None or depth > 6:
            return []
        identity = id(value)
        if identity in seen:
            return []
        seen.add(identity)

        items: list[Any] = []
        if isinstance(value, dict):
            items.append(value)
            for k in (
                "usage",
                "usage_metadata",
                "token_usage",
                "response_metadata",
                "metadata",
                "raw_response",
                "raw",
            ):
                if k in value:
                    items.extend(
                        CostTracker._iter_usage_candidates(value[k], depth=depth + 1, seen=seen)
                    )
            # Some frameworks nest usage under less predictable keys.
            for nested in value.values():
                items.extend(
                    CostTracker._iter_usage_candidates(nested, depth=depth + 1, seen=seen)
                )
            return items

        if isinstance(value, (list, tuple)):
            for entry in value:
                items.extend(CostTracker._iter_usage_candidates(entry, depth=depth + 1, seen=seen))
            return items

        attrs = (
            "usage",
            "usage_metadata",
            "token_usage",
            "response_metadata",
            "metadata",
            "raw_response",
            "raw",
            "model_extra",
            "__dict__",
        )
        for attr in attrs:
            if hasattr(value, attr):
                try:
                    nested = getattr(value, attr)
                except Exception:
                    continue
                items.extend(CostTracker._iter_usage_candidates(nested, depth=depth + 1, seen=seen))
        return items

    @staticmethod
    def _read_usage_fields(value: Any) -> dict[str, int] | None:
        if value is None:
            return None

        if not isinstance(value, dict):
            mapped: dict[str, Any] = {}
            for name in (
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_token_count",
                "output_token_count",
                "prompt_token_count",
                "completion_token_count",
                "total_token_count",
                "promptTokens",
                "completionTokens",
                "totalTokens",
            ):
                if hasattr(value, name):
                    mapped[name] = getattr(value, name)
            if not mapped:
                return None
            value = mapped

        prompt = (
            value.get("prompt_tokens")
            if value.get("prompt_tokens") is not None
            else value.get("input_tokens")
        )
        completion = (
            value.get("completion_tokens")
            if value.get("completion_tokens") is not None
            else value.get("output_tokens")
        )
        total = value.get("total_tokens")

        if prompt is None:
            prompt = value.get("prompt_token_count")
        if prompt is None:
            prompt = value.get("input_token_count")
        if prompt is None:
            prompt = value.get("promptTokens")

        if completion is None:
            completion = value.get("completion_token_count")
        if completion is None:
            completion = value.get("output_token_count")
        if completion is None:
            completion = value.get("completionTokens")

        if total is None:
            total = value.get("total_token_count")
        if total is None:
            total = value.get("totalTokens")
        if total is None and (prompt is not None or completion is not None):
            total = (int(prompt or 0) + int(completion or 0))

        if prompt is None and completion is None and total is None:
            return None

        return {
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(completion or 0),
            "total_tokens": int(total or 0),
        }

    @staticmethod
    def _extract_model_name(value: Any) -> str | None:
        if value is None:
            return None

        if isinstance(value, dict):
            for key in ("model", "model_name", "model_id"):
                model = value.get(key)
                if isinstance(model, str) and model.strip():
                    return model.strip()
            for key in ("response_metadata", "metadata", "raw_response", "raw"):
                if key in value:
                    nested = CostTracker._extract_model_name(value[key])
                    if nested:
                        return nested
            return None

        for key in ("model", "model_name", "model_id"):
            if hasattr(value, key):
                model = getattr(value, key)
                if isinstance(model, str) and model.strip():
                    return model.strip()

        for key in ("response_metadata", "metadata", "raw_response", "raw"):
            if hasattr(value, key):
                nested = CostTracker._extract_model_name(getattr(value, key))
                if nested:
                    return nested
        return None

    @staticmethod
    def _guess_provider(model: str | None, provider_hint: str | None) -> str:
        if provider_hint:
            return provider_hint.strip().lower()

        normalized_model = (model or "").lower()
        if "claude" in normalized_model:
            return "anthropic"
        if normalized_model.startswith("gpt") or "embedding" in normalized_model:
            return "openai"
        return "unknown"

    @staticmethod
    def _build_pricing(overrides: dict[str, dict[str, float]]) -> dict[str, PricingRate]:
        pricing = dict(CostTracker._DEFAULT_RATES)
        for key, value in overrides.items():
            input_rate = float(value.get("input_per_million", 0.0))
            output_rate = float(value.get("output_per_million", 0.0))
            pricing[key.lower()] = PricingRate(
                input_per_million=input_rate,
                output_per_million=output_rate,
            )
        return pricing

    def _lookup_rate(self, provider: str, model: str) -> PricingRate:
        key = f"{provider}:{model}".lower()
        if key in self._pricing:
            return self._pricing[key]

        # Prefix fallback for versioned model names (ex: claude-3-5-sonnet-20241022)
        for pricing_key, rate in self._pricing.items():
            provider_prefix, model_prefix = pricing_key.split(":", 1)
            if provider_prefix == provider and model.lower().startswith(model_prefix):
                return rate

        return PricingRate(input_per_million=0.0, output_per_million=0.0)

    def _next_call_id(self) -> str:
        with self._lock:
            self._call_counter += 1
            return f"call_{self._call_counter:05d}"

    @staticmethod
    def _to_operation_slug(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
        return cleaned.strip("_") or "unknown_operation"

    @staticmethod
    def _service_name(call_type: str, provider: str) -> str:
        normalized_provider = (provider or "unknown").strip().lower()
        if call_type == "embedding":
            if normalized_provider == "openai":
                return "openai_embeddings"
            if normalized_provider in {"huggingface", "hf", "sentence-transformers"}:
                return "huggingface_embeddings"
            return f"{normalized_provider}_embeddings"

        if normalized_provider == "anthropic":
            return "anthropic_claude"
        if normalized_provider == "openai":
            return "openai_chat"
        return f"{normalized_provider}_llm"

    def _append_call(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._api_calls.append(payload)

    @staticmethod
    def _sanitize_token_count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def record_embedding_call(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        texts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a single embedding API call."""

        prompt_tokens = sum(self.estimate_tokens(text) for text in texts)
        rate = self._lookup_rate(provider, model)
        input_cost = (prompt_tokens / 1_000_000) * rate.input_per_million

        call = {
            "call_id": self._next_call_id(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_type": "embedding",
            "provider": provider,
            "model": model,
            "operation": operation,
            "input_items": len(texts),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": 0.0,
            "total_cost_usd": round(input_cost, 8),
            "metadata": metadata or {},
        }
        self._append_call(call)
        return call

    def record_llm_call(
        self,
        *,
        agent_name: str,
        provider: str,
        model: str,
        request_id: str | None,
        usage: dict[str, int],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a single LLM API call."""

        prompt_tokens = self._sanitize_token_count(usage.get("prompt_tokens", 0))
        completion_tokens = self._sanitize_token_count(usage.get("completion_tokens", 0))
        total_tokens = self._sanitize_token_count(usage.get("total_tokens", prompt_tokens + completion_tokens))
        total_tokens = max(total_tokens, prompt_tokens + completion_tokens)
        unique_request_id = str(request_id or "").strip()
        request_key = f"{agent_name}|{provider}|{model}|{unique_request_id}" if unique_request_id else ""

        rate = self._lookup_rate(provider, model)
        input_cost = (prompt_tokens / 1_000_000) * rate.input_per_million
        output_cost = (completion_tokens / 1_000_000) * rate.output_per_million
        total_cost = input_cost + output_cost

        with self._lock:
            if request_key and request_key in self._llm_call_index_by_request_key:
                # Streaming providers can emit multiple usage snapshots for one request.
                # Keep one call row per request and upsert to latest totals.
                existing_idx = self._llm_call_index_by_request_key[request_key]
                existing = self._api_calls[existing_idx]
                merged_prompt = max(int(existing.get("prompt_tokens") or 0), prompt_tokens)
                merged_completion = max(int(existing.get("completion_tokens") or 0), completion_tokens)
                merged_total = max(
                    int(existing.get("total_tokens") or 0),
                    total_tokens,
                    merged_prompt + merged_completion,
                )
                existing["prompt_tokens"] = merged_prompt
                existing["completion_tokens"] = merged_completion
                existing["total_tokens"] = merged_total
                existing["input_cost_usd"] = round(
                    (merged_prompt / 1_000_000) * rate.input_per_million,
                    8,
                )
                existing["output_cost_usd"] = round(
                    (merged_completion / 1_000_000) * rate.output_per_million,
                    8,
                )
                existing["total_cost_usd"] = round(
                    float(existing["input_cost_usd"]) + float(existing["output_cost_usd"]),
                    8,
                )
                merged_metadata = dict(existing.get("metadata") or {})
                merged_metadata.update(metadata or {})
                existing["metadata"] = merged_metadata
                existing["updated_at"] = datetime.now(timezone.utc).isoformat()
                return existing

            if not request_key:
                dedupe_fingerprint = (
                    f"{agent_name}|{provider}|{model}|"
                    f"{prompt_tokens}|{completion_tokens}|{total_tokens}"
                )
                if dedupe_fingerprint in self._seen_llm_fingerprints:
                    return {}
                self._seen_llm_fingerprints.add(dedupe_fingerprint)

            self._call_counter += 1
            call = {
                "call_id": f"call_{self._call_counter:05d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_type": "llm",
                "agent_name": agent_name,
                "provider": provider,
                "model": model,
                "request_id": unique_request_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "input_cost_usd": round(input_cost, 8),
                "output_cost_usd": round(output_cost, 8),
                "total_cost_usd": round(total_cost, 8),
                "metadata": metadata or {},
            }
            self._api_calls.append(call)
            if request_key:
                self._llm_call_index_by_request_key[request_key] = len(self._api_calls) - 1
            return call

    def try_record_llm_from_response(
        self,
        *,
        response: Any,
        agent_name: str,
        request_id: str | None = None,
        provider_hint: str | None = None,
        model_hint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Try to record an LLM call from a provider/framework response object."""

        usage = self._extract_usage_from_any(response)
        if not usage:
            return False

        model = self._extract_model_name(response) or model_hint or "unknown"
        provider = self._guess_provider(model, provider_hint)
        recorded = self.record_llm_call(
            agent_name=agent_name,
            provider=provider,
            model=model,
            request_id=request_id,
            usage=usage,
            metadata=metadata,
        )
        return bool(recorded)

    def build_report(self) -> dict[str, Any]:
        """Build a simple usage + cost report."""

        calls = list(self._api_calls)
        line_item_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

        def bucket_key_for_call(call: dict[str, Any]) -> tuple[str, str, str]:
            call_type = str(call.get("call_type") or "unknown")
            provider = str(call.get("provider") or "unknown")
            model = str(call.get("model") or "unknown")
            service = self._service_name(call_type, provider)

            if call_type == "embedding":
                operation = "embedding_total"
            else:
                agent_name = str(call.get("agent_name") or "agent")
                operation = f"{self._to_operation_slug(agent_name)}_response"

            return service, model, operation

        for call in calls:
            service, model, operation = bucket_key_for_call(call)
            key = (service, model, operation)
            if key not in line_item_buckets:
                provider = str(call.get("provider") or "unknown")
                rate = self._lookup_rate(provider, model)
                line_item_buckets[key] = {
                    "service": service,
                    "model": model,
                    "operation": operation,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usd_per_1m_input_tokens": rate.input_per_million,
                    "usd_per_1m_output_tokens": rate.output_per_million,
                    "estimated_cost_usd": 0.0,
                }

            item = line_item_buckets[key]
            item["input_tokens"] += int(call.get("prompt_tokens") or 0)
            item["output_tokens"] += int(call.get("completion_tokens") or 0)
            item["estimated_cost_usd"] = round(
                float(item["estimated_cost_usd"]) + float(call.get("total_cost_usd") or 0.0),
                8,
            )

        line_items = sorted(
            line_item_buckets.values(),
            key=lambda item: (item["service"], item["operation"], item["model"]),
        )
        total_cost = round(sum(float(item["estimated_cost_usd"]) for item in line_items), 8)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_cost_usd": total_cost,
            "line_items": line_items,
            "api_calls": calls,
        }

    def save_report(self, *, log_dir: str | Path = "logs") -> Path:
        """Persist report as JSON file and return path."""

        report = self.build_report()
        target_dir = Path(log_dir)
        target_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = target_dir / f"{self.report_name}_{timestamp}.json"
        path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        return path
