"""Phase MR.3 — Source 5 : LLM web_search discovery of buyable media.

The opposite of `services.trust_sources.discover_trust_sources` : where trust
sources finds AUTHORITATIVE NEUTRAL references (and excludes media/blogs/
commercial), this finds MEDIA THAT PUBLISH SPONSORED / PARTNER articles — the
lifestyle magazines, specialized press and niche blogs a PR team would pitch.

Used as the credit-debited fallback in `media_replacement.suggest()` when the
free DB cascade (sources 1-4) is thin and the user opts into a web search.

Pure function : 1 OpenAI Responses API web_search call, no DB I/O. The caller
re-validates the returned domains via LinkFinder (price) + the standard hard
filters before surfacing them.

Cost ~$0.02-0.04 per call (gpt-4.1-mini + web_search). Gated by credit debit
at the API + assert_within_budget in the handler.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


_SKIP_PATTERNS: tuple[str, ...] = (
    "wikipedia.", "wikimedia.", "facebook.", "instagram.", "twitter.",
    "x.com", "tiktok.", "youtube.", "linkedin.", "pinterest.", "reddit.",
    "amazon.", "google.", "doctolib.",
)


def _normalize_domain(d: str) -> str:
    if not d:
        return ""
    s = str(d).strip().lower()
    s = re.sub(r"^https?://", "", s)
    if s.startswith("www."):
        s = s[4:]
    s = s.split("/", 1)[0].rstrip(".")
    return s if "." in s else ""


def _extract_json(text: str) -> dict | None:
    """Tolerant JSON extraction — mirrors trust_sources._extract_json."""
    if not text:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", text.strip())
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


_DISCOVERY_PROMPT = """You are a PR / media-relations expert building a shortlist of MEDIA OUTLETS \
where a sponsored / partner article could be published, on the following topic and market.

Topic / question : {topic}
Audience / reader : {persona}
Market / country : {country}
Industry / sector : {vertical}

Return between 8 and 12 real, currently-active MEDIA WEBSITES that:
- publish sponsored content, partner articles, native advertising, or accept guest/expert contributions
- are editorially relevant to the topic and read by the target audience
- operate in the given market / language

STRICT INCLUSION — keep only sites that match AT LEAST ONE of:
- Consumer / lifestyle magazines (health, beauty, parenting, wellness, etc. as relevant to the topic)
- Specialized or trade media for this sector
- Established niche blogs / editorial sites with real audience and a "work with us" / advertising offer
- General-interest news media with a branded-content / partner desk

STRICT EXCLUSION — REJECT:
- Brand, manufacturer or retailer websites (even market leaders)
- E-commerce, price-comparison, shopping aggregators, affiliate-only sites
- Government agencies, regulators, scientific journals, academic publishers (those are reference sites, not buyable media)
- Social media platforms, video platforms, forums
- Wikipedia / Wikimedia
{exclude_block}

For each outlet provide:
- "domain" : bare domain (e.g. "santemagazine.fr", not a full URL)
- "name" : the outlet's name (max 60 chars)
- "reason" : one short sentence on why it fits this topic + audience (max 120 chars)

Return ONLY valid JSON, no markdown, no commentary:

{{
  "media": [
    {{"domain": "...", "name": "...", "reason": "..."}},
    ...
  ]
}}
"""


def discover_media_via_web(
    *,
    topic: str,
    persona: str,
    country: str,
    language: str,
    vertical: str,
    exclude_domains: set[str] | None = None,
    openai_api_key: str = "",
    model: str = "gpt-4.1-mini",
) -> list[dict]:
    """One OpenAI web_search call → list of buyable-media dicts.

    Returns ``[{"domain","name","reason"}, ...]`` (possibly empty on failure).
    Caller re-validates via LinkFinder + hard filters. Pure function.
    """
    if not topic or not openai_api_key:
        logger.warning("discover_media_via_web: missing topic=%r or api_key", topic)
        return []

    exclude_block = ""
    excl = sorted({_normalize_domain(d) for d in (exclude_domains or set())} - {""})
    if excl:
        # Cap the list so the prompt stays bounded ; these are the user's own
        # brands + competitors we must never suggest.
        preview = ", ".join(excl[:40])
        exclude_block = (
            f"- NEVER suggest any of these domains (own brand / competitors): {preview}"
        )

    prompt = _DISCOVERY_PROMPT.format(
        topic=(topic or "").strip()[:300],
        persona=(persona or "general audience").strip()[:200],
        country=(country or "").strip() or "France",
        vertical=(vertical or "").strip()[:200] or "(unspecified)",
        exclude_block=exclude_block,
    )
    # NW.2 — inject anti-AI-detection humanizer (compact), same as trust_sources.
    try:
        from services.natural_writing_helpers import inject_humanizer
        prompt = inject_humanizer(prompt, mode="compact")
    except Exception:
        pass

    try:
        import openai
        client = openai.OpenAI(api_key=openai_api_key, timeout=120)
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=prompt,
            temperature=0.3,
        )
        text = response.output_text or ""
    except Exception as e:
        logger.warning(f"discover_media_via_web OpenAI call failed: {e}")
        return []

    data = _extract_json(text)
    if not isinstance(data, dict):
        logger.warning("discover_media_via_web: unparseable JSON (text_len=%d)", len(text))
        return []

    raw = data.get("media") or []
    if not isinstance(raw, list):
        return []

    excl_set = {_normalize_domain(d) for d in (exclude_domains or set())}
    out: list[dict] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        domain = _normalize_domain(entry.get("domain") or "")
        if not domain or domain in seen:
            continue
        if domain in excl_set:
            continue
        if any(p in domain for p in _SKIP_PATTERNS):
            continue
        seen.add(domain)
        out.append({
            "domain": domain,
            "name": (entry.get("name") or "").strip()[:60] or domain,
            "reason": (entry.get("reason") or "").strip()[:120],
        })

    logger.info(
        "discover_media_via_web: topic=%r → %d media (%s%s)",
        (topic or "")[:60], len(out),
        ", ".join(m["domain"] for m in out[:5]),
        "..." if len(out) > 5 else "",
    )
    return out
