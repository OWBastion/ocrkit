"""Candidate adjudicators for the #4 experiment.

Two arms are compared by the experiment runner:

- deterministic normalization only (#3);
- deterministic normalization plus a candidate adjudicator over the residual
  unresolved cases.

Adjudicators are always constrained by the record's allowlist: a provider must
never select text outside the supplied candidates. The provider path is
fail-closed (timeout, invalid output, off-allowlist candidate, or any error
falls back to ``unresolved`` for human review) and is never a hard dependency of
OCRKit startup, production recognition, tests, or model release.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol
from urllib import error as url_error
from urllib import request as url_request

from pydantic import BaseModel

from .records import ReviewedAnnotationRecord, canonicalize, compute_input_digest

HEURISTIC_MARGIN = 0.1
DEFAULT_TIMEOUT_SECONDS = 30.0


class AdjudicationOutput(BaseModel):
    decision: Literal["constrained_match", "ambiguous", "unresolved"]
    candidate: str | None = None
    confidence: float | None = None
    reason_code: str
    provider: str
    model: str | None = None
    prompt_version: str | None = None
    input_digest: str
    usage: dict[str, int] | None = None
    raw_output: dict[str, Any] | None = None


class Adjudicator(Protocol):
    name: str

    def adjudicate(self, record: ReviewedAnnotationRecord) -> AdjudicationOutput:
        ...


class HeuristicAdjudicator:
    """Offline constrained picker: weighted agreement of engine texts on the allowlist.

    Uses no provider, so the experiment machinery can be measured end to end
    without network access or SDK dependencies.
    """

    name = "heuristic"

    def adjudicate(self, record: ReviewedAnnotationRecord) -> AdjudicationOutput:
        digest = compute_input_digest(record)
        weights: dict[str, float] = {}
        for engine in record.engines:
            if not engine.text or not engine.text.strip():
                continue
            normalized = canonicalize(engine.text)
            for candidate in record.candidates:
                if canonicalize(candidate) == normalized:
                    weights[candidate] = weights.get(candidate, 0.0) + (engine.confidence if engine.confidence is not None else 0.5)

        if weights:
            ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
            best, best_weight = ordered[0]
            second = ordered[1][1] if len(ordered) > 1 else 0.0
            total = sum(weights.values())
            if best_weight - second >= HEURISTIC_MARGIN:
                return AdjudicationOutput(
                    decision="constrained_match",
                    candidate=best,
                    confidence=round(best_weight / total, 4),
                    reason_code="heuristic_consensus",
                    provider="none",
                    input_digest=digest,
                )
            return AdjudicationOutput(
                decision="ambiguous",
                reason_code="heuristic_tie",
                provider="none",
                input_digest=digest,
            )

        if record.normalization and record.normalization.normalized_text:
            normalized = canonicalize(record.normalization.normalized_text)
            for candidate in record.candidates:
                if canonicalize(candidate) == normalized:
                    return AdjudicationOutput(
                        decision="constrained_match",
                        candidate=candidate,
                        confidence=1.0,
                        reason_code="normalization_allowlist",
                        provider="none",
                        input_digest=digest,
                    )
        return AdjudicationOutput(
            decision="unresolved",
            reason_code="no_evidence",
            provider="none",
            input_digest=digest,
        )


def _call_with_timeout(fn, record: ReviewedAnnotationRecord, timeout_seconds: float) -> AdjudicationOutput:
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, record)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError as exc:
        raise TimeoutError(f"adjudicator call exceeded {timeout_seconds}s") from exc
    finally:
        future.add_done_callback(lambda _done: pool.shutdown(wait=False))


class ProviderAdjudicator:
    """Fail-closed wrapper around any callable provider adapter."""

    def __init__(
        self,
        provider_name: str,
        call_fn,
        *,
        model: str | None = None,
        prompt_version: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.name = provider_name
        self._call_fn = call_fn
        self.model = model
        self.prompt_version = prompt_version
        self.timeout_seconds = timeout_seconds

    def _failure(self, record: ReviewedAnnotationRecord, reason_code: str) -> AdjudicationOutput:
        return AdjudicationOutput(
            decision="unresolved",
            reason_code=reason_code,
            provider=self.name,
            model=self.model,
            prompt_version=self.prompt_version,
            input_digest=compute_input_digest(record),
        )

    @property
    def cost_per_million_input(self) -> float:
        return float(getattr(self._call_fn, "cost_per_million_input", 0.0))

    @property
    def cost_per_million_output(self) -> float:
        return float(getattr(self._call_fn, "cost_per_million_output", 0.0))

    def adjudicate(self, record: ReviewedAnnotationRecord) -> AdjudicationOutput:
        digest = compute_input_digest(record)
        try:
            output = _call_with_timeout(self._call_fn, record, self.timeout_seconds)
        except TimeoutError:
            return self._failure(record, "timeout")
        except Exception:  # noqa: BLE001 - fail closed on any provider error
            return self._failure(record, "provider_failure")

        if not isinstance(output, AdjudicationOutput):
            return self._failure(record, "invalid_output")
        if output.decision == "constrained_match" and output.candidate not in record.candidates:
            return self._failure(record, "candidate_not_in_allowlist")
        return AdjudicationOutput(
            decision=output.decision,
            candidate=output.candidate,
            confidence=output.confidence,
            reason_code=output.reason_code,
            provider=self.name,
            model=self.model,
            prompt_version=self.prompt_version,
            input_digest=digest,
            usage=output.usage,
            raw_output=output.raw_output,
        )


class OpenAICompatibleProvider:
    """Optional, provider-agnostic adapter for any OpenAI-compatible chat endpoint.

    Constructed only when explicitly configured through environment variables
    (``OCRKIT_ADJUDICATION_ENDPOINT``, ``OCRKIT_ADJUDICATION_MODEL``,
    ``OCRKIT_ADJUDICATION_API_KEY``). The prompt contains only text metadata from
    the record and is constrained to the allowlist; the response must be JSON
    matching the candidate output contract.
    """

    name = "openai-compatible"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        prompt_version: str = "v1",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cost_per_million_input: float = 1.0,
        cost_per_million_output: float = 3.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.prompt_version = prompt_version
        self.timeout_seconds = timeout_seconds
        self.cost_per_million_input = cost_per_million_input
        self.cost_per_million_output = cost_per_million_output

    def build_prompt(self, record: ReviewedAnnotationRecord) -> list[dict[str, str]]:
        system = (
            "You resolve ambiguous OCR text for a game HUD into exactly one of the "
            "supplied allowed canonical candidates, or you abstain. Never invent text "
            "outside the allowed candidates and never propose new labels. Reply with a "
            "single JSON object: "
            '{"decision": "constrained_match" | "ambiguous" | "unresolved", '
            '"candidate": <one allowed candidate or null>, '
            '"confidence": <0..1 or null>, '
            '"reason_code": "<short stable code>"}. '
            "Use ambiguous when several candidates are equally plausible, unresolved "
            "when the evidence does not support any candidate."
        )
        user = {
            "layout_version": record.layout_version,
            "roi": record.roi,
            "field_family": record.field_family,
            "engines": [
                {"engine": item.engine, "text": item.text, "confidence": item.confidence}
                for item in record.engines
            ],
            "normalization": record.normalization.model_dump() if record.normalization else None,
            "candidates": list(record.candidates),
            "candidates_version": record.candidates_version,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
        ]

    def _adjudicate_from_response(self, record: ReviewedAnnotationRecord, body: dict[str, Any]) -> AdjudicationOutput:
        digest = compute_input_digest(record)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("provider response is missing choices[0].message.content") from exc
        parsed = json.loads(content)
        candidate = parsed.get("candidate")
        return AdjudicationOutput(
            decision=parsed.get("decision", "unresolved"),
            candidate=candidate if isinstance(candidate, str) else None,
            confidence=parsed.get("confidence"),
            reason_code=str(parsed.get("reason_code", "provider_selected")),
            provider=self.name,
            model=self.model,
            prompt_version=self.prompt_version,
            input_digest=digest,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            raw_output={"response": body},
        )

    def call(self, record: ReviewedAnnotationRecord) -> AdjudicationOutput:
        payload = {
            "model": self.model,
            "messages": self.build_prompt(record),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = url_request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with url_request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (url_error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"provider request failed: {exc}") from exc
        return self._adjudicate_from_response(record, body)


def build_adjudicator(
    kind: str,
    *,
    endpoint: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    prompt_version: str = "v1",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Adjudicator:
    """Build the requested adjudicator; provider paths fail loudly when unconfigured."""
    if kind == "heuristic":
        return HeuristicAdjudicator()
    if kind == "openai":
        if not endpoint or not model:
            raise ValueError(
                "openai adjudicator requires OCRKIT_ADJUDICATION_ENDPOINT and OCRKIT_ADJUDICATION_MODEL"
            )
        provider = OpenAICompatibleProvider(
            endpoint=endpoint,
            model=model,
            api_key=api_key or "",
            prompt_version=prompt_version,
            timeout_seconds=timeout_seconds,
        )
        return ProviderAdjudicator(
            provider.name,
            provider.call,
            model=provider.model,
            prompt_version=provider.prompt_version,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"unknown adjudicator kind: {kind!r}")
