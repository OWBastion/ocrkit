"""ROI-scoped deterministic terminology normalization.

Repeatable OCR character errors in known Bastion HUD regions are better fixed by
constrained deterministic rules than by model retraining or an LLM.  This layer
turns raw OCR text into an explainable normalized candidate keyed by
``layout_version + roi`` (and, implicitly, the field/terminology family declared
for that scope).

Guarantees:

- raw OCR evidence is never overwritten (callers keep the raw text separately);
- a normalization is adopted only when the scope, allowed terminology set,
  ambiguity checks, and rule type make the result deterministic;
- when several candidates remain plausible the token stays raw and the result is
  marked ``ambiguous`` instead of inventing text;
- rule sets are versioned and each adopted change is traceable through a rule id;
- nothing here depends on the Studio UI, submission approval, title eligibility,
  or any platform business rule, and holdout ground truth is never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from typing import Any

import yaml

ALIAS_CONFIDENCE = 1.0
CONFUSION_CONFIDENCE = 0.98
FUZZY_UNRESOLVED_BAND = 0.15

MATCH_ALIAS = "exact_alias"
MATCH_CONFUSION = "confusion"
MATCH_FUZZY = "fuzzy"

DECISION_NORMALIZED = "normalized"
DECISION_UNCHANGED = "unchanged"
DECISION_AMBIGUOUS = "ambiguous"
DECISION_UNRESOLVED = "unresolved"

# Token / separator runs. Separators: / ／ whitespace | ｜ · , ，
_TOKEN = re.compile(r"[^/\／|｜·,，\s]+|[/／|｜·,，\s]+")

DEFAULT_RULES_PATH = Path("configs/terminology.yaml")


@dataclass(frozen=True)
class Scope:
    """Rules that apply to one terminology family inside one or more layouts."""

    id: str
    roi: str
    layout_versions: frozenset[str]
    allowed_terms: tuple[str, ...]
    aliases: dict[str, str]
    confusions: dict[str, str]
    fuzzy_match: bool = False
    fuzzy_threshold: float = 0.62
    fuzzy_min_margin: float = 0.2


@dataclass(frozen=True)
class TerminologyCatalog:
    rules_version: str
    scopes: tuple[Scope, ...]


@dataclass(frozen=True)
class TokenNormalization:
    raw: str
    normalized: str
    status: str  # "none" | "adopted" | "ambiguous" | "unresolved"
    match_type: str | None  # set when status == "adopted"
    rule_id: str | None
    confidence: float


@dataclass(frozen=True)
class TerminologyResult:
    scope_id: str | None
    rules_version: str
    decision: str
    raw_text: str
    normalized_text: str
    tokens: tuple[TokenNormalization, ...]


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def scope_for(catalog: TerminologyCatalog, layout_version: str, roi: str) -> Scope | None:
    for scope in catalog.scopes:
        if scope.roi == roi and layout_version in scope.layout_versions:
            return scope
    return None


def _fuzzy_token(token: str, scope: Scope) -> TokenNormalization:
    scored = sorted(
        ((term, SequenceMatcher(None, token, term).ratio()) for term in scope.allowed_terms),
        key=lambda item: item[1],
        reverse=True,
    )
    best_term, best = scored[0]
    second = scored[1][1] if len(scored) > 1 else 0.0
    if best >= scope.fuzzy_threshold and best - second >= scope.fuzzy_min_margin:
        return TokenNormalization(
            raw=token,
            normalized=best_term,
            status="adopted",
            match_type=MATCH_FUZZY,
            rule_id=f"{scope.id}:fuzzy:{best_term}",
            confidence=best,
        )
    if best >= scope.fuzzy_threshold:
        return TokenNormalization(raw=token, normalized=token, status="ambiguous", match_type=None, rule_id=None, confidence=0.0)
    if best >= scope.fuzzy_threshold - FUZZY_UNRESOLVED_BAND:
        return TokenNormalization(raw=token, normalized=token, status="unresolved", match_type=None, rule_id=None, confidence=0.0)
    return TokenNormalization(raw=token, normalized=token, status="none", match_type=None, rule_id=None, confidence=0.0)


def _normalize_token(token: str, scope: Scope) -> TokenNormalization:
    if token in scope.allowed_terms:
        return TokenNormalization(raw=token, normalized=token, status="none", match_type=None, rule_id=None, confidence=0.0)
    if token in scope.aliases:
        return TokenNormalization(
            raw=token,
            normalized=scope.aliases[token],
            status="adopted",
            match_type=MATCH_ALIAS,
            rule_id=f"{scope.id}:alias:{token}",
            confidence=ALIAS_CONFIDENCE,
        )

    fired = [key for key in scope.confusions if key in token]
    if fired:
        replaced = token.translate(str.maketrans(scope.confusions))
        if replaced != token and replaced in scope.allowed_terms:
            return TokenNormalization(
                raw=token,
                normalized=replaced,
                status="adopted",
                match_type=MATCH_CONFUSION,
                rule_id=f"{scope.id}:confusion:{'+'.join(fired)}",
                confidence=CONFUSION_CONFIDENCE,
            )
        if replaced != token:
            return TokenNormalization(raw=token, normalized=token, status="unresolved", match_type=None, rule_id=None, confidence=0.0)

    if scope.fuzzy_match:
        return _fuzzy_token(token, scope)
    return TokenNormalization(raw=token, normalized=token, status="none", match_type=None, rule_id=None, confidence=0.0)


def normalize_scope_text(text: str, scope: Scope, rules_version: str = "") -> TerminologyResult:
    """Normalize raw text against a single scope, preserving separators."""
    tokens: list[TokenNormalization] = []
    parts: list[str] = []
    for piece in _TOKEN.findall(text):
        if not piece:
            continue
        if piece[0] in "/／|｜·,，" or piece.isspace():
            parts.append(piece)
            continue
        token = _nfkc(piece)
        result = _normalize_token(token, scope)
        tokens.append(result)
        parts.append(result.normalized if result.status == "adopted" else piece)

    if any(token.status == "ambiguous" for token in tokens):
        decision = DECISION_AMBIGUOUS
    elif any(token.status == "adopted" for token in tokens):
        decision = DECISION_NORMALIZED
    elif any(token.status == "unresolved" for token in tokens):
        decision = DECISION_UNRESOLVED
    else:
        decision = DECISION_UNCHANGED

    return TerminologyResult(
        scope_id=scope.id,
        rules_version=rules_version,
        decision=decision,
        raw_text=text,
        normalized_text="".join(parts),
        tokens=tuple(tokens),
    )


def normalize_roi_text(
    text: str,
    layout_version: str,
    roi: str,
    catalog: TerminologyCatalog,
) -> TerminologyResult:
    scope = scope_for(catalog, layout_version, roi)
    if scope is None:
        return TerminologyResult(
            scope_id=None,
            rules_version=catalog.rules_version,
            decision=DECISION_UNCHANGED,
            raw_text=text,
            normalized_text=text,
            tokens=(),
        )
    return normalize_scope_text(text, scope, catalog.rules_version)


def _parse_scope(raw: dict[str, Any]) -> Scope:
    scope_id = str(raw["id"])
    roi = str(raw["roi"])
    layout_versions = frozenset(str(item) for item in raw.get("layout_versions", []))
    if not layout_versions:
        raise ValueError(f"scope {scope_id!r} must declare at least one layout_version")
    allowed_terms = tuple(_nfkc(str(item)) for item in raw.get("allowed_terms", []))
    if not allowed_terms:
        raise ValueError(f"scope {scope_id!r} must declare a non-empty allowed_terms set")

    aliases: dict[str, str] = {}
    for key, value in (raw.get("aliases") or {}).items():
        nfkc_key, nfkc_value = _nfkc(str(key)), _nfkc(str(value))
        if not nfkc_key or not nfkc_value:
            raise ValueError(f"scope {scope_id!r} has an empty alias")
        aliases[nfkc_key] = nfkc_value

    confusions: dict[str, str] = {}
    for key, value in (raw.get("confusions") or {}).items():
        nfkc_key, nfkc_value = _nfkc(str(key)), _nfkc(str(value))
        if len(nfkc_key) != 1 or len(nfkc_value) != 1:
            raise ValueError(f"scope {scope_id!r} confusions must map single characters")
        confusions[nfkc_key] = nfkc_value

    return Scope(
        id=scope_id,
        roi=roi,
        layout_versions=layout_versions,
        allowed_terms=allowed_terms,
        aliases=aliases,
        confusions=confusions,
        fuzzy_match=bool(raw.get("fuzzy_match", False)),
        fuzzy_threshold=float(raw.get("fuzzy_threshold", 0.62)),
        fuzzy_min_margin=float(raw.get("fuzzy_min_margin", 0.2)),
    )


def load_terminology_rules(path: Path) -> TerminologyCatalog:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("rules_version"), str):
        raise ValueError(f"terminology rules are missing a rules_version: {path}")
    raw_scopes = data.get("scopes")
    if not isinstance(raw_scopes, list):
        raise ValueError(f"terminology rules are missing a scopes list: {path}")
    scopes = [_parse_scope(item) for item in raw_scopes]
    seen: set[str] = set()
    for scope in scopes:
        if scope.id in seen:
            raise ValueError(f"duplicate terminology scope id: {scope.id}")
        seen.add(scope.id)
    return TerminologyCatalog(rules_version=data["rules_version"], scopes=tuple(scopes))


@lru_cache(maxsize=1)
def default_terminology_catalog() -> TerminologyCatalog:
    """Load the checked-in rule set for offline preparation and evaluation."""
    return load_terminology_rules(DEFAULT_RULES_PATH)
