"""Pydantic schemas for validating LLM outputs in AI Scan handlers.

Each handler that calls Claude/OpenAI/Gemini and persists structured data should
validate the response through one of these models via `validate_items` (for lists)
or `validate_object` (for single dicts).

Mode is **warn-then-skip-invalid**: each item is validated individually; invalid
items are dropped with a logged warning, valid items proceed. If 0 items remain
valid in a list response, the helper raises RuntimeError so the worker marks
the scan failed and triggers C2 refund.

Models are intentionally lenient (most fields default to "" or []) because LLMs
omit fields randomly. Tighten validators only when downstream code requires it.

Ported and extended from seo-llm/src/models.py with sen-ai SaaS shapes.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ─── Validation helpers ─────────────────────────────────────────────────────

def validate_items(
    items: list,
    model: type[BaseModel],
    context: str = "",
) -> list[BaseModel]:
    """Validate a list of dicts against a Pydantic model item-by-item.

    Returns the list of valid model instances (invalid items dropped).
    Logs a warning per invalid item plus a summary when >0 dropped.
    Raises RuntimeError if items is non-empty but ALL fail (likely prompt drift
    or model regression — caller should fail the job so C2 refunds credits).
    """
    valid: list[BaseModel] = []
    invalid_count = 0
    for i, item in enumerate(items or []):
        try:
            valid.append(model.model_validate(item))
        except Exception as e:
            invalid_count += 1
            preview = str(item)[:200].replace("\n", " ")
            logger.warning(
                f"{context} item {i} failed validation, dropping: {e} "
                f"(raw: {preview})"
            )
    if invalid_count > 0:
        logger.warning(
            f"{context}: dropped {invalid_count}/{len(items)} invalid items "
            f"({len(valid)} valid)"
        )
    if items and not valid:
        raise RuntimeError(
            f"{context}: all {len(items)} items failed Pydantic validation "
            f"(prompt drift or model regression)"
        )
    return valid


def validate_object(
    obj: dict | None,
    model: type[BaseModel],
    context: str = "",
) -> BaseModel:
    """Validate a single dict against a Pydantic model.

    Raises RuntimeError on validation failure (single objects can't drop-and-continue).
    """
    if obj is None:
        raise RuntimeError(f"{context}: response is None")
    try:
        return model.model_validate(obj)
    except Exception as e:
        preview = str(obj)[:500].replace("\n", " ")
        raise RuntimeError(
            f"{context}: validation failed: {e} (raw: {preview})"
        ) from e


# ─── classify_topics LLM output ─────────────────────────────────────────────

class TopicGenerated(BaseModel):
    """One topic in the classify_topics LLM response."""

    model_config = ConfigDict(extra="ignore")

    nom: str = Field(min_length=2, description="Topic name as displayed in UI")
    description: str = ""
    urls: list[str] = Field(default_factory=list)

    @field_validator("nom", "description")
    @classmethod
    def _strip(cls, v: str) -> str:
        return (v or "").strip()


class BrandDetected(BaseModel):
    """One brand entry in classify_topics' marques_detectees array.

    The category set here is the broader Claude vocabulary (site_brand / site_gamme /
    competitor / unclassified). cleanup_brands uses a different category set —
    see BrandClassification below.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    category: Literal[
        "site_brand", "site_gamme", "competitor", "unclassified",
    ] = "unclassified"
    topics: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("brand name cannot be empty")
        return v

    @field_validator("category", mode="before")
    @classmethod
    def _default_category(cls, v):
        if not v or not isinstance(v, str):
            return "unclassified"
        v = v.strip().lower()
        # Tolerate a few synonyms Claude sometimes returns
        synonyms = {
            "target_brand": "site_brand",
            "target_gamme": "site_gamme",
            "concurrent": "competitor",
        }
        return synonyms.get(v, v)


# ─── cleanup_brands LLM output ──────────────────────────────────────────────

class BrandClassification(BaseModel):
    """One brand classification from cleanup_brands LLM response.

    Distinct from BrandDetected: cleanup_brands uses target_brand/competitor/ignore
    vocabulary (legacy adapter convention).
    """

    model_config = ConfigDict(extra="ignore")

    original: str = Field(min_length=1, description="Original detected name (lowercase)")
    name: str | None = None
    category: Literal[
        "target_brand", "target_gamme", "target_product",
        "competitor", "competitor_gamme",
        "ignore",
    ] = "ignore"
    parent: str | None = None

    @field_validator("original")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("original name cannot be empty")
        return v


# ─── persona + questions LLM output ─────────────────────────────────────────

QuestionType = Literal["basique", "validation", "comparative", "technique", "urgente"]


class QuestionGenerated(BaseModel):
    """One generated question (embedded in PersonaGenerated or standalone in
    generate_persona_questions response)."""

    model_config = ConfigDict(extra="ignore")

    type_question: QuestionType = "basique"
    question: str = Field(min_length=10)
    intention_cachee: str = ""
    signal_positif: str = ""
    signal_negatif: str = ""

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("question cannot be empty")
        return v

    @field_validator("type_question", mode="before")
    @classmethod
    def _normalize_type(cls, v):
        # Tolerate missing/null type_question — defaults to "basique"
        if not v or not isinstance(v, str):
            return "basique"
        v = v.strip().lower()
        # Accept English variants
        synonyms = {
            "basic": "basique",
            "technical": "technique",
            "urgent": "urgente",
            "compare": "comparative",
            "validate": "validation",
        }
        return synonyms.get(v, v)


class PersonaGenerated(BaseModel):
    """One persona from generate_personas LLM response."""

    model_config = ConfigDict(extra="ignore")

    nom: str = Field(min_length=2)
    segment_principal: str = ""
    profil_demographique: dict = Field(default_factory=dict)
    intentions_recherche: list[str] = Field(default_factory=list)
    parcours_type: str = ""
    points_douleur: list[str] = Field(default_factory=list)
    mots_cles_associes: list[str] = Field(default_factory=list)
    opportunites: list[str] = Field(default_factory=list)
    questions: list[QuestionGenerated] = Field(default_factory=list)

    @field_validator("nom")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("persona nom cannot be empty")
        return v


# ─── LLM-as-judge per-response judgment (Sprint J) ──────────────────────────


CitationQuality = Literal["lead", "alternative", "footnote", "absent"]


class QuestionJudgmentEntry(BaseModel):
    """One judgment entry inside the judge's batched JSON output.

    The judge processes N (question, response) pairs per Haiku call and
    emits one entry per pair, keyed by the integer index passed in the prompt.
    Stored in scan_question_judgments (migration 037).

    Foot-gun #3 enforcement (intent_addressed requires evidence span) lives
    in the handler — Pydantic accepts the raw LLM output, the handler resets
    the bool to false if evidence is empty.
    """

    model_config = ConfigDict(extra="ignore")

    idx: int = Field(ge=0, description="0-based index matching the prompt batch")
    positive_signal_hit: bool = False
    positive_signal_evidence: str = ""
    negative_signal_hit: bool = False
    negative_signal_evidence: str = ""
    intent_addressed: bool = False
    intent_evidence: str = ""
    citation_quality: CitationQuality | None = None
    enveloppement_score: int | None = Field(default=None, ge=0, le=5)

    @field_validator(
        "positive_signal_evidence",
        "negative_signal_evidence",
        "intent_evidence",
    )
    @classmethod
    def _strip(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("citation_quality", mode="before")
    @classmethod
    def _normalize_quality(cls, v):
        if not v or not isinstance(v, str):
            return None
        v = v.strip().lower()
        return v if v in ("lead", "alternative", "footnote", "absent") else None


# ─── domain brief (OpenAI Responses API + web_search) ───────────────────────

class CompetitorInBrief(BaseModel):
    """One competitor in the DomainBrief.competitors list."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    products: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        return (v or "").strip()


class DomainBrief(BaseModel):
    """Structured business brief produced by generate_domain_brief.

    Stored in scan.config.domain_brief and injected into 5 downstream prompts
    via brief_injector.format_brief_context.
    """

    model_config = ConfigDict(extra="ignore")

    company: str = ""
    description: str = ""
    industry: str = ""
    country: str = ""
    brands: list[str] = Field(default_factory=list)
    product_lines: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    competitors: list[CompetitorInBrief] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    target_audience: str = ""

    @field_validator("competitors", mode="before")
    @classmethod
    def _normalize_competitors(cls, v):
        # OpenAI sometimes returns list of strings, sometimes list of dicts
        if isinstance(v, list):
            normalized = []
            for c in v:
                if isinstance(c, str):
                    normalized.append({"name": c, "products": []})
                elif isinstance(c, dict):
                    normalized.append(c)
            return normalized
        return v


# ─── editorial summary (Claude) ─────────────────────────────────────────────

class EditorialSummary(BaseModel):
    """Marketing-friendly editorial summary stored in scan.summary.editorial.

    Consumed by /results.astro for the hero narrative section.
    """

    model_config = ConfigDict(extra="ignore")

    hook: str = ""
    summary: list[str] = Field(default_factory=list)
    interpretation: str = ""
    opportunities: list[str] = Field(default_factory=list)
    competitor_insight: str = ""
