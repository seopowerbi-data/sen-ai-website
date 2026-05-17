"""Media-partner picker for netlinking article opportunities.

Symmetric in spirit to `services.sitemap_matcher` (FAQ) but on the opposite
side of the publication flow :

  - sitemap_matcher  → discovers pages on the user's OWN brand site
                        where a FAQ should live (embeddings on a corpus we
                        crawled and indexed).
  - media_picker     → discovers third-party MEDIA PARTNERS where a sponsored
                        article should publish (aggregation of LLM citations
                        from the scan + LinkFinder pricing enrichment).

## Why the discovery signal is the scan itself

The same scan that justified the netlinking opportunity already contains the
answer to "which medias matter for this question" : every LLM response was
parsed, citations recorded in `scan_llm_results.citations`. The domains the
LLMs cited when asked about eczema, dermatite atopique, etc. are by
construction the medias that already cover the topic in the AI conversation.
Publishing a sponsored article on those medias is the most direct way to
recapture citation share.

This is more powerful than a generic Google search would be : it's the
LLMs' own observed behavior on this exact question, captured at scan time,
already in our DB. Zero extra API call for discovery.

## LinkFinder serves DOUBLE DUTY

1. **Explicit enrichment** : DA / TF / CF / RD / price HT per domain.
2. **Implicit buyability classifier** : a domain present in link-finder.net's
   catalog SELLS paid placements (that's literally what the platform is for).
   A domain absent from the catalog likely doesn't (or hasn't been listed yet
   → user contacts directly).

So LinkFinder is *both* the pricing layer AND a free site-type filter, which
is why we can ship without porting seo_llm's full `site_classifier.py` for
MVP. The ranking formula naturally surfaces LinkFinder-known domains above
the rest (price/DA components contribute, unknowns score 0 on enrichment).

## Filter stack (in order)

  1. user-rejected (item.rejected_target_urls passed as exclude_domains)
  2. own brand domains      (BrandResolver.resolve_promotion)
  3. competitor brand domains (services.competitor_domains)
  4. universal authority TLDs (.gov / .gouv.fr / .europa.eu / .int / …)
  5. trust_sources discovered domains (client's institutional list)
  6. universal patterns (services.url_filter : ecommerce / social / blog)

Items 4-6 cross-vertical : work for any client industry without configuration.
Items 2-3 vertical-aware via per-client BrandResolver / per-scan SBC.
Item 1 user-aware (cumulative rejection across rematch attempts).

## Ranking

  relevance_score = citation_count × log(da + 1) - price_eur / 1000

  Tiebreak : providers_cited length DESC, then domain alphabetic.

Rationale :
  - citation_count : strongest signal (LLM-validated topical relevance).
  - log(da + 1) : diminishing returns on authority (DA 95 vs DA 80 should
                  not swamp; both are "great", citation_count picks winner).
  - price / 1000 : modest penalty so premium medias aren't excluded but
                   secondary to relevance / authority.
  - Domains without LinkFinder data : da=0, price=0 → score = 0 → coule
                   below domains with any enrichment data.
  - Provider diversity tiebreak : cross-validated by multiple LLMs > single LLM.

## Public surface

  pick_media_candidates(scan_id, db, *, question_id | target_question, ...)
      → list[MediaCandidate]

  refresh_enrichment(candidates, force=False)
      → list[MediaCandidate]  (re-fetch LinkFinder data for existing candidates)

  MediaCandidate (TypedDict) : full shape of one candidate row.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import TypedDict

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Cap on candidates sent to LinkFinder batch enrichment. After universal
# filter we typically have 10-30 candidates per question ; capping at 20
# keeps the LinkFinder batch small (1 API call, ~1-3 s on cache miss) while
# giving the ranker enough diversity to surface a meaningful top-k.
_MAX_ENRICH_BATCH = 20


class MediaCandidate(TypedDict, total=False):
    """One media partner candidate for a netlinking opportunity.

    Persisted as a JSONB element of `scan_content_items.target_url_candidates`
    so the validation page top-3 picker can render the chips without a DB
    round-trip. Stored in this exact shape so UI code can branch on
    content_type and read the fields it needs.
    """

    # Identity
    domain: str                       # 'doctissimo.fr' (normalized, no www, no scheme)
    url: str                          # 'https://www.doctissimo.fr/' (canonical home)
    name: str | None                  # 'Doctissimo' (from LinkFinder, may be None)

    # Discovery signal (from scan_llm_results.citations)
    citation_count: int               # total across all LLM responses for this question
    providers_cited: list[str]        # ['openai', 'gemini', ...] — diversity proxy
    excerpt: str | None               # snippet of context from 1st citation (≤200 chars)
    source: str                       # 'scan_citation' (only source today)

    # LinkFinder enrichment (None if no creds OR domain absent from catalog)
    da: int | None
    tf: int | None
    cf: int | None
    rd: int | None
    price_eur: float | None
    platform_url: str | None          # where to buy on link-finder.net

    # Computed ranking metric (not displayed, used for sort)
    relevance_score: float


# ─── Public API ────────────────────────────────────────────────────────


def pick_media_candidates(
    scan_id: str,
    db: Session,
    *,
    question_id: str | None = None,
    target_question: str | None = None,
    top_k: int = 3,
    exclude_domains: set[str] | None = None,
    enrich: bool = True,
) -> list[MediaCandidate]:
    """Return top-K media partner candidates for one opportunity, ranked.

    Provide EITHER `question_id` (preferred, exact PK lookup) OR
    `target_question` (text, will be resolved to question_id via
    case-insensitive match against `scan_questions.question`). If both,
    `question_id` wins.

    Returns an empty list when :
      - scan_id is falsy
      - question_id can't be resolved
      - the scan doesn't exist
      - the question has 0 citations
      - all citations are filtered out (own brand only, all institutional, ...)

    Side effects : 1 DB query (citations) + 1 optional LinkFinder batch call.
    No DB writes. Caller persists the result to
    `scan_content_items.target_url_candidates` if desired.
    """
    if not scan_id:
        return []

    # ── Resolve question_id if only text was provided ─────────────────
    if not question_id and target_question:
        question_id = _resolve_question_id(scan_id, target_question, db)
    if not question_id:
        logger.info(
            f"media_picker: no question_id resolvable for scan {scan_id} "
            f"(target_question={target_question!r}) — returning []"
        )
        return []

    exclude_domains_norm: set[str] = {
        _normalize_domain(d) for d in (exclude_domains or [])
    }
    exclude_domains_norm.discard("")

    # ── 1. Pull raw citation rows ─────────────────────────────────────
    rows = _query_citation_rows(str(question_id), db)
    if not rows:
        logger.info(
            f"media_picker: 0 citations for question {question_id} "
            f"(scan {scan_id}) — returning []"
        )
        return []

    # ── 2. Build filter context (own brands + competitors + trust) ────
    from models import Scan

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        logger.warning(f"media_picker: scan {scan_id} not found")
        return []

    own_brand_domains = _resolve_own_brand_domains(scan, db)
    competitor_domains_set = _resolve_competitor_domains(scan_id, db)
    trust_domains_set = _resolve_trust_domains(scan.client_id, db)

    # ── 3. Aggregate citations by domain + filter inline ──────────────
    by_domain: dict[str, dict] = {}
    drop_reasons: dict[str, int] = defaultdict(int)

    for domain, provider, url, excerpt in rows:
        # If this domain has been seen before, just increment counters.
        # The filter decision was cached on first encounter.
        if domain in by_domain:
            entry = by_domain[domain]
            if entry.get("_filtered"):
                continue  # already excluded — don't even bump counters
            entry["citation_count"] += 1
            if provider and provider not in entry["providers_cited"]:
                entry["providers_cited"].append(provider)
            continue

        kept, drop_reason = _filter_candidate_domain(
            domain, own_brand_domains, competitor_domains_set,
            trust_domains_set, exclude_domains_norm,
        )
        if not kept:
            drop_reasons[drop_reason] += 1
            by_domain[domain] = {"_filtered": True}
            continue

        by_domain[domain] = _new_candidate(domain, provider, excerpt)

    candidates = [v for v in by_domain.values() if not v.get("_filtered")]

    logger.info(
        f"media_picker: question={question_id} → {len(rows)} raw citations, "
        f"{len(candidates)} candidates after filter "
        f"(dropped: {dict(drop_reasons) if drop_reasons else 'none'})"
    )

    if not candidates:
        return []

    # ── 4. Cap to top _MAX_ENRICH_BATCH by raw citation_count BEFORE enrichment.
    # This keeps the LinkFinder batch small. The final ranking happens post-
    # enrichment with the full score formula.
    candidates.sort(key=lambda c: (-c["citation_count"], c["domain"]))
    to_rank = candidates[:_MAX_ENRICH_BATCH]

    # ── 5. Enrich via LinkFinder (graceful if creds absent or API fails) ──
    if enrich:
        _apply_linkfinder_enrichment(to_rank)

    # ── 6. Compute relevance_score per candidate ──────────────────────
    for c in to_rank:
        c["relevance_score"] = _score(c)

    # ── 7. Final sort : score DESC, provider diversity DESC, alphabetic ──
    to_rank.sort(
        key=lambda c: (
            -c["relevance_score"],
            -len(c.get("providers_cited") or []),
            c["domain"],
        )
    )

    return to_rank[:top_k]


def refresh_enrichment(
    candidates: list[dict],
    force: bool = False,
) -> list[dict]:
    """Re-fetch LinkFinder enrichment for an existing candidates list.

    Mutates candidates in place + re-computes relevance_score + re-sorts.
    Returns the same list (for chaining).

    `force` is reserved for future use — LinkFinder client today has a 7-day
    internal cache we can't bypass without clearing its on-disk cache file.
    v1 always returns whatever the cache hands back (which is fresh enough
    for most use cases — prices on link-finder.net move slowly).
    """
    if not candidates:
        return candidates
    _apply_linkfinder_enrichment(candidates)
    for c in candidates:
        c["relevance_score"] = _score(c)
    candidates.sort(
        key=lambda c: (
            -c["relevance_score"],
            -len(c.get("providers_cited") or []),
            c["domain"],
        )
    )
    return candidates


# ─── Internal helpers ──────────────────────────────────────────────────


def _normalize_domain(raw: str | None) -> str:
    """Strip protocol, www., trailing path. Return lowercase bare domain or ''."""
    if not raw:
        return ""
    s = str(raw).strip().lower()
    s = re.sub(r"^https?://", "", s)
    if s.startswith("www."):
        s = s[4:]
    s = s.split("/", 1)[0].rstrip(".")
    return s if "." in s else ""


def _resolve_question_id(
    scan_id: str, target_question: str, db: Session,
) -> str | None:
    """Lookup ScanQuestion.id from (scan_id, question text), case-insensitive.

    Same pattern as content_items.py:_build_competitor_snapshot. Returns None
    when no match — caller's responsibility to decide (skip vs warn)."""
    from sqlalchemy import func

    from models import ScanQuestion

    q_text = (target_question or "").strip().lower()
    if not q_text:
        return None
    row = (
        db.query(ScanQuestion.id)
        .filter(
            ScanQuestion.scan_id == scan_id,
            func.lower(ScanQuestion.question) == q_text,
        )
        .first()
    )
    return str(row[0]) if row else None


def _query_citation_rows(
    question_id: str, db: Session,
) -> list[tuple[str, str, str, str]]:
    """Flatten scan_llm_results.citations for one question.

    Returns list of (domain, provider, url, excerpt) tuples. Citations
    flagged `est_site_cible=true` (= the scanned site itself, either own or
    competitor) are skipped here — they're already excluded by the regular
    brand-domain filter, but skipping early saves work."""
    rows = db.execute(
        text("""
            SELECT
                LOWER(c->>'domaine')                                AS domain,
                LOWER(slr.provider)                                 AS provider,
                COALESCE(c->>'url', '')                             AS url,
                COALESCE(c->>'contexte', '')                        AS excerpt,
                COALESCE((c->>'est_site_cible')::bool, false)       AS is_scanned_site
            FROM scan_llm_results slr,
                 jsonb_array_elements(slr.citations) c
            WHERE slr.question_id = :qid
              AND slr.citations IS NOT NULL
              AND jsonb_array_length(slr.citations) > 0
              AND c->>'domaine' IS NOT NULL
              AND c->>'domaine' <> ''
        """),
        {"qid": question_id},
    ).fetchall()

    out: list[tuple[str, str, str, str]] = []
    for r in rows:
        if r.is_scanned_site:
            continue
        d = _normalize_domain(r.domain)
        if not d:
            continue
        out.append((d, r.provider or "", r.url or "", r.excerpt or ""))
    return out


def _resolve_own_brand_domains(scan, db: Session) -> set[str]:
    """Get the user's own brand domains (normalized) from BrandResolver."""
    try:
        from services.brand_resolver import PromotionUnsetError, resolve_promotion

        try:
            promotion = resolve_promotion(scan, db)
            domains = {
                _normalize_domain(b.domain) for b in promotion.promote_brands if b.domain
            }
            domains.discard("")
            return domains
        except PromotionUnsetError:
            # No primary brand set for this client — filter still works
            # but won't exclude own brands. That's OK ; they'll be in
            # citations only if the user owns the scanned site, in which
            # case competitor_domains is empty too (own-brand scan).
            return set()
    except Exception:
        logger.exception(
            "media_picker: resolve_promotion crashed — proceeding without own-brand filter"
        )
        return set()


def _resolve_competitor_domains(scan_id: str, db: Session) -> set[str]:
    try:
        from services.competitor_domains import get_competitor_domains_for_scan
        return get_competitor_domains_for_scan(scan_id, db)
    except Exception:
        logger.exception(
            "media_picker: get_competitor_domains_for_scan crashed — proceeding without competitor filter"
        )
        return set()


def _resolve_trust_domains(client_id, db: Session) -> set[str]:
    try:
        from services.trust_sources import get_trust_sources_for_client
        return {
            _normalize_domain(d) for d in get_trust_sources_for_client(client_id, db)
        }
    except Exception:
        logger.exception(
            "media_picker: get_trust_sources_for_client crashed — proceeding without trust filter"
        )
        return set()


def _filter_candidate_domain(
    domain: str,
    own_brand_domains: set[str],
    competitor_domains_set: set[str],
    trust_domains_set: set[str],
    exclude_domains: set[str],
) -> tuple[bool, str]:
    """Apply all hard filters in order. Returns (kept, drop_reason_if_dropped).

    Order matters for the diagnostic logging ; the first match wins so we
    know WHY a domain was rejected. Reasons mirror url_filter.py for
    consistency in log aggregation.
    """
    from services.trust_sources import is_universal_authority_tld
    from services.url_filter import is_excluded_url

    if domain in exclude_domains:
        return False, "user_rejected"

    for own in own_brand_domains:
        if not own:
            continue
        if domain == own or domain.endswith("." + own):
            return False, "own_brand"

    for comp in competitor_domains_set:
        if not comp:
            continue
        if domain == comp or domain.endswith("." + comp):
            return False, "competitor"

    if is_universal_authority_tld(domain):
        return False, "gov_authority"

    # Trust domains : subdomain-aware match (fr.wikipedia.org matches
    # wikipedia.org in the trust list). Exact-match here would let language
    # subdomains slip through — observed on PF backfill 2026-05-17 where
    # fr.wikipedia.org was picked as a candidate.
    for trusted in trust_domains_set:
        if not trusted:
            continue
        if domain == trusted or domain.endswith("." + trusted):
            return False, "trust_institutional"

    # Universal cross-vertical patterns (ecommerce / social / blog / forum).
    # We pass competitor_domains=None to is_excluded_url since we've already
    # handled the competitor case above with the per-scan SBC denylist.
    excluded, reason = is_excluded_url(f"https://{domain}/", competitor_domains=None)
    if excluded:
        return False, reason

    return True, ""


def _new_candidate(domain: str, provider: str, excerpt: str) -> dict:
    """Initialize a fresh MediaCandidate dict with first citation.

    All LinkFinder fields default to None — they get filled in by
    `_apply_linkfinder_enrichment` if creds are configured. Initializing
    them here ensures the dict shape is consistent regardless of whether
    enrichment runs (UI never sees a missing key)."""
    url = f"https://www.{domain}/"
    return {
        "domain": domain,
        "url": url,
        "name": None,
        "citation_count": 1,
        "providers_cited": [provider] if provider else [],
        "excerpt": (excerpt[:200] + "…") if excerpt and len(excerpt) > 200 else (excerpt or None),
        "source": "scan_citation",
        "da": None,
        "tf": None,
        "cf": None,
        "rd": None,
        "price_eur": None,
        "platform_url": None,
    }


def _apply_linkfinder_enrichment(candidates: list[dict]) -> None:
    """Mutate candidates in place : add da / tf / cf / rd / price_eur / platform_url.

    Graceful : if LinkFinder isn't configured OR the API call fails, leaves
    all enrichment fields as None and logs a warning. Caller never sees an
    exception — the candidates list comes back enrichment-less but valid.

    The LinkFinder client has a 7-day internal cache, so repeated calls for
    the same domains are cheap on the second hit.
    """
    if not candidates:
        return

    try:
        from seo_llm.src.link_finder_client import LinkFinderClient
        client = LinkFinderClient()
    except Exception:
        logger.exception(
            "media_picker: LinkFinderClient import failed — candidates "
            "will return without enrichment"
        )
        return

    if not client.is_api_configured:
        logger.info(
            "media_picker: LinkFinder not configured (LINKFINDER_EMAIL + "
            "LINKFINDER_PASSWORD env vars missing on worker) — skipping "
            "enrichment. Candidates will surface without DA / price chips. "
            "Configure creds in worker/.env and restart the worker to enable."
        )
        return

    domains = [c["domain"] for c in candidates]
    try:
        prices = client.get_prices_batch(domains)
    except Exception:
        logger.exception(
            "media_picker: LinkFinder.get_prices_batch crashed — degraded mode"
        )
        return

    enriched_count = 0
    for c in candidates:
        entry = prices.get(c["domain"]) or {}
        # `source` field on the LinkFinder entry tells us if data is real or 'not_found'
        if entry.get("source") and entry["source"] != "not_found":
            c["da"] = _int_or_none(entry.get("da"))
            c["tf"] = _int_or_none(entry.get("tf"))
            c["cf"] = _int_or_none(entry.get("cf"))
            c["rd"] = _int_or_none(entry.get("rd"))
            c["price_eur"] = _float_or_none(entry.get("prix_ht"))
            c["platform_url"] = entry.get("platform_url") or None
            if any(c.get(k) is not None for k in ("da", "tf", "price_eur")):
                enriched_count += 1

    logger.info(
        f"media_picker: LinkFinder enriched {enriched_count}/{len(candidates)} domains "
        f"(others : not in catalog → '💬 Contact' UI chip)"
    )


def _int_or_none(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float_or_none(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        # Treat 0 (or negative) as null — paywall, contact-only, or malformed.
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _score(c: dict) -> float:
    """relevance_score = citation_count × log(da + 1) - price_eur / 1000.

    See module docstring "Ranking" section for the rationale.
    Returns 0.0 for a candidate with no citation_count, no DA, no price.
    """
    cc = c.get("citation_count") or 0
    da = c.get("da") or 0
    price = c.get("price_eur") or 0.0
    return cc * math.log(da + 1) - (price / 1000.0)
