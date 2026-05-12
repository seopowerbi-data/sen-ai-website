"""Trust sources service — per-client list of authoritative reference domains.

Built on top of OpenAI Responses API web_search, this service is the SaaS-side
replacement for seo_llm's `_discover_reference_sources` (which is brand-category
keyed and Pierre-Fabre-hardcoded). Multi-vertical by design : the only input is
the free-form industry text from `client.apps['client_brief']['industry']`, so
a dermo-cosmetic client discovers HAS/ANSM/Vidal while an automotive client
discovers EuroNCAP/NHTSA/SAE without code changes.

Lifecycle :
  1. `generate_client_brief` finishes successfully + emits an `industry` field
  2. → enqueues a `discover_trust_sources` job
  3. → handler calls `discover_trust_sources()` here (1 OpenAI web_search call,
     ~$0.02), stores the result at `client.apps['trust_sources']`
  4. content generation pipelines (generate_faq today, generate_article later)
     call `get_trust_sources_for_client()` to build the allowlist used to filter
     out competitor / commercial / blog URLs from web_search context

Refresh : 90 days OR when `industry_hash` changes (user edits brief). Trust
source landscape (regulators, academic publishers) moves slowly; per-client
cost stays ~$0.08/year.

Universal layer : `UNIVERSAL_REFERENCES` (Wikipedia/Wikimedia) and
`UNIVERSAL_AUTHORITY_TLDS` (.gov / .gouv.fr / .europa.eu / .int / …) apply to
every client regardless of vertical and are always included in the allowlist.

Public surface :
  - `discover_trust_sources(industry, language, api_key, model)` — 1-shot LLM
  - `get_trust_sources_for_client(client_id, db)` — DB read, merged allowlist
  - `is_trusted_domain(domain, trust_list)` — single decision helper
  - `is_discovery_stale(client_apps, industry)` — handler short-circuit gate
  - `build_trust_sources_payload(industry, sources)` — shape for client.apps write
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta

import openai
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ─── Cross-vertical universal layer ─────────────────────────────────────
# Always trusted regardless of client industry. Wikipedia is explicitly listed
# (not pattern-matchable); government / EU TLDs are pattern-matched via the
# tuple below so we don't have to enumerate every .gov country.
UNIVERSAL_REFERENCES: tuple[str, ...] = (
    "wikipedia.org",
    "wikimedia.org",
)

UNIVERSAL_AUTHORITY_TLDS: tuple[str, ...] = (
    ".gov",          # US federal / state
    ".gov.uk",       # UK
    ".gov.ca",       # Canada
    ".gov.au",       # Australia
    ".gouv.fr",      # France
    ".gc.ca",        # Canada gov alt
    ".europa.eu",    # EU institutions
    ".int",          # International orgs (who.int, etc.)
    ".admin.ch",     # Swiss federal
    ".bund.de",      # German federal
    ".gob.es",       # Spain
    ".gov.it",       # Italy
)

# Refresh policy. 90 days keeps cost negligible (~$0.08/year/client at
# $0.02/call) while still picking up changes when regulators reorganize.
TRUST_SOURCES_TTL_DAYS = 90

# Skip patterns applied to discovered URLs — drops commercial / social / blog
# / e-commerce hostnames before they reach the trust list. Wikipedia is also
# stripped here because it's already in the universal baseline (no need to
# double-list).
_DISCOVERY_SKIP_PATTERNS: tuple[str, ...] = (
    "blog.", "forum.", "avis.", "shop.", "boutique.", "store.",
    "amazon.", "cdiscount.", "fnac.", "ebay.", "alibaba.",
    "wikipedia.", "wikimedia.",
    "facebook.", "twitter.", "instagram.", "youtube.", "linkedin.",
    "tiktok.", "pinterest.", "reddit.",
)


# ─── Helpers (pure) ────────────────────────────────────────────────────

def _normalize_domain(d: str) -> str:
    """Strip protocol, www., trailing path. Return lowercase bare domain or ''."""
    if not d or not isinstance(d, str):
        return ""
    nd = re.sub(r"^https?://", "", d.strip().lower())
    if nd.startswith("www."):
        nd = nd[4:]
    nd = nd.split("/", 1)[0].strip().rstrip(".")
    if "." not in nd:
        return ""
    return nd


def is_universal_authority_tld(domain: str) -> bool:
    """True if `domain` ends with a public-sector TLD pattern (TLD suffix match)."""
    d = _normalize_domain(domain)
    if not d:
        return False
    return any(d.endswith(t) for t in UNIVERSAL_AUTHORITY_TLDS)


def is_trusted_domain(domain: str, trust_list: list[str] | tuple[str, ...]) -> bool:
    """Single allowlist decision — returns True iff `domain` is on `trust_list`
    (or a subdomain of) OR matches a universal authority TLD pattern.

    Used by content generation filters (FAQ, article) to decide whether a
    web_search URL is on a trusted source. `trust_list` should typically be
    the output of `get_trust_sources_for_client()`."""
    d = _normalize_domain(domain)
    if not d:
        return False
    for entry in trust_list or ():
        e = _normalize_domain(entry) if isinstance(entry, str) else ""
        if not e:
            continue
        if d == e or d.endswith("." + e):
            return True
    return is_universal_authority_tld(d)


def _hash_industry(industry: str) -> str:
    """Stable short hash — used to detect when industry text changed and we
    need to re-discover even though TTL hasn't expired."""
    norm = (industry or "").strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _extract_json(text: str) -> dict | None:
    """Tolerant JSON extraction — strips code fences, falls back to outermost {...}."""
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


# ─── Discovery (1 LLM call, pure function) ─────────────────────────────

_DISCOVERY_PROMPT = """You are building a list of authoritative, neutral reference websites for content generation in the following industry / sector :

Industry / sector : {industry}
{language_hint}

Return between 8 and 15 websites that are AUTHORITATIVE NEUTRAL THIRD-PARTY references for this industry. Use web search to verify each.

STRICT INCLUSION CRITERIA — keep only sites that match AT LEAST ONE of :
- Government agencies, regulators, or public bodies for this sector (any country relevant to the industry)
- Recognized professional / learned societies, industry standards bodies, certifying organizations
- Peer-reviewed scientific journals, academic publishers, open-access research repositories
- National or international institutions, public-funded research labs, public databases
- Major encyclopedic / educational references (broadly trusted)

STRICT EXCLUSION CRITERIA — REJECT sites that match ANY of :
- Brand websites, manufacturer websites, retailer websites (even market leaders)
- E-commerce, price comparison, shopping aggregators, affiliate marketing
- Blogs, forums, opinion sites, user-review platforms
- General-interest news media, magazines, lifestyle press
- Social media platforms
- Wikipedia / Wikimedia (already in our universal baseline — do NOT list)

For each source provide :
- "domain" : bare domain (e.g., "has-sante.fr", not "https://www.has-sante.fr/path")
- "org" : official organization name (max 80 chars)
- "type" : one of "authority" (government/regulator), "society" (professional/learned), "journal" (academic/scientific), "reference" (encyclopedic/educational), "research" (public databases)

Return ONLY valid JSON with this exact structure (no markdown, no commentary) :

{{
  "sources": [
    {{"domain": "...", "org": "...", "type": "..."}},
    ...
  ]
}}
"""


def discover_trust_sources(
    industry: str,
    language: str = "en",
    openai_api_key: str = "",
    model: str = "gpt-4.1-mini",
) -> list[dict]:
    """One OpenAI Responses API web_search call → de-duplicated list of trust
    source dicts. Pure function : no DB I/O, caller persists the result.

    Args:
        industry: free-form sector text from client.apps['client_brief']['industry']
        language: ISO code biasing which countries' authorities to prefer
        openai_api_key: required (returns [] if missing)
        model: gpt-4.1-mini matches the rest of the worker's web_search calls

    Returns:
        list[{"domain": str, "org": str, "type": str}] — possibly empty on failure.
        Caller falls back to UNIVERSAL_REFERENCES alone in that case.
    """
    if not industry or not openai_api_key:
        logger.warning(
            "discover_trust_sources: missing industry=%r or api_key — returning []",
            industry,
        )
        return []

    language_hint = ""
    if language and language.lower() not in ("en", "english", ""):
        language_hint = f"Language / primary market : {language}"

    prompt = _DISCOVERY_PROMPT.format(
        industry=industry.strip(), language_hint=language_hint,
    )

    try:
        client = openai.OpenAI(api_key=openai_api_key, timeout=120)
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=prompt,
            temperature=0.2,
        )
        text = response.output_text or ""
    except Exception as e:
        logger.warning(f"discover_trust_sources OpenAI call failed: {e}")
        return []

    data = _extract_json(text)
    if not isinstance(data, dict):
        logger.warning(
            "discover_trust_sources: could not parse JSON (industry=%r, text_len=%d)",
            industry, len(text),
        )
        return []

    raw_sources = data.get("sources") or []
    if not isinstance(raw_sources, list):
        return []

    sources: list[dict] = []
    seen: set[str] = set()
    for entry in raw_sources:
        if not isinstance(entry, dict):
            continue
        domain = _normalize_domain(entry.get("domain") or "")
        if not domain:
            continue
        if any(p in domain for p in _DISCOVERY_SKIP_PATTERNS):
            continue
        if domain in seen:
            continue
        seen.add(domain)
        sources.append({
            "domain": domain,
            "org": (entry.get("org") or "").strip()[:80] or domain,
            "type": (entry.get("type") or "reference").strip().lower(),
        })

    logger.info(
        "discover_trust_sources: industry=%r → %d sources (%s%s)",
        industry, len(sources),
        ", ".join(s["domain"] for s in sources[:5]),
        "..." if len(sources) > 5 else "",
    )
    return sources


# ─── Persistence helpers (DB-aware) ────────────────────────────────────

def is_discovery_stale(client_apps: dict | None, industry: str) -> bool:
    """True if trust_sources is missing, older than TTL, or industry changed.

    Handler short-circuit gate : if False, skip the OpenAI call entirely.
    """
    trust = (client_apps or {}).get("trust_sources") or {}
    if not (trust.get("domains") or []):
        return True
    if trust.get("industry_hash") != _hash_industry(industry):
        return True
    discovered_at = trust.get("discovered_at")
    if not discovered_at:
        return True
    try:
        when = datetime.fromisoformat(discovered_at.replace("Z", ""))
    except Exception:
        return True
    return (datetime.utcnow() - when) > timedelta(days=TRUST_SOURCES_TTL_DAYS)


def build_trust_sources_payload(industry: str, sources: list[dict],
                                 prior_extras: list | None = None) -> dict:
    """Build the dict to store at client.apps['trust_sources'].

    `prior_extras` carries forward any user-added domains (future Settings UI
    will write to `extra_domains`). We never overwrite user input on refresh.
    """
    domains = [s["domain"] for s in sources if s.get("domain")]
    return {
        "domains": domains,
        "details": sources,
        "industry_text": industry,
        "industry_hash": _hash_industry(industry),
        "discovered_at": datetime.utcnow().isoformat(),
        "sources_count": len(domains),
        "extra_domains": list(prior_extras or []),
    }


def get_trust_sources_for_client(client_id, db: Session) -> list[str]:
    """Return the merged, de-duplicated list of trusted bare domains for one
    client. Always includes UNIVERSAL_REFERENCES; appends discovered domains
    and user extras if present.

    Note : universal authority TLDs (.gov / .gouv.fr / .europa.eu / .int / …)
    are NOT enumerable so they aren't in this list — callers should also
    invoke `is_universal_authority_tld(domain)` (or use `is_trusted_domain`
    which checks both).

    Returns UNIVERSAL_REFERENCES alone (with a warning log) when the client
    hasn't been discovered yet — the FAQ / article filter still works, just
    with a narrower allowlist until the discover handler runs.
    """
    from models import Client  # local import to avoid worker bootstrap cycle

    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        logger.warning(f"get_trust_sources_for_client: client {client_id} not found")
        return list(UNIVERSAL_REFERENCES)

    apps = client.apps or {}
    trust = apps.get("trust_sources") or {}
    discovered = trust.get("domains") or []
    extras = trust.get("extra_domains") or []

    if not discovered:
        logger.info(
            "trust_sources not yet discovered for client %s — using UNIVERSAL only "
            "(enqueue discover_trust_sources to populate)",
            client_id,
        )

    merged: list[str] = []
    seen: set[str] = set()
    for raw in list(UNIVERSAL_REFERENCES) + list(discovered) + list(extras):
        nd = _normalize_domain(raw)
        if not nd or nd in seen:
            continue
        seen.add(nd)
        merged.append(nd)

    return merged
