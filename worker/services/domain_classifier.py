"""Domain classifier — Gemini-backed site_type tagging with persistent DB cache.

Ported from `seo_llm.src.site_classifier.py` (the seo-llm CLI module that
populates `dim_domain.csv` on SharePoint). Two adaptations for SaaS :

  1. **Cache backend** : local CSV file → Postgres `domain_classifications`
     table (migration 033). Cross-client global cache : a Brand site is a
     Brand site regardless of which client surfaced it, so PF's prior
     classifications benefit every future client.

  2. **Trigger model** : the seo-llm CLI batch-classifies all uncached
     domains after each scan run as a separate step ; here we classify
     lazily, on-demand, from `services.media_picker.pick_media_candidates`
     when a never-seen-before domain shows up. Keeps the call path simple
     (no extra handler), latency acceptable (~1-3 s for a 30-domain batch).

The 10 categories + the classification prompt are VERBATIM from the
seo-llm source so we preserve PF's accumulated classification quality
(3503 domains pre-imported via the one-shot script).

## BUYABLE_SITE_TYPES — the netlinking filter set

  Health & Beauty Media   — editorial wellness/beauty press
  Blog                    — lifestyle / beauty / niche bloggers (accept sponsored)
  News                    — general press / news magazines

Excludes :
  Brand (any brand's own site — can't publish there)
  Medical Reference (HAS / ANSM / Vidal — institutional, not paid placements)
  Government / Encyclopedia / Forum / E-commerce — same reason
  Other (unclassifiable — conservative drop)

The set is cross-vertical : for an automotive / finance / B2B client whose
citations don't include "Health & Beauty Media", the picker still surfaces
Blog + News candidates. Verticals needing their own category (e.g.,
"Auto Media", "Finance Media") would extend SITE_CATEGORIES + the prompt
+ re-run the classifier on existing rows — defer until needed.

## Public surface

  classify_domains(domains, db, model=...) → dict[domain → site_type]
  is_buyable(site_type) → bool
  SITE_CATEGORIES (list of valid types — write-side validation)
  BUYABLE_SITE_TYPES (set used by media_picker filter)
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ─── Categories (verbatim from seo_llm.src.site_classifier) ────────────

SITE_CATEGORIES: list[str] = [
    "Government",
    "Medical Reference",
    "News",
    "Health & Beauty Media",
    "Brand",
    "E-commerce",
    "Encyclopedia",
    "Forum",
    "Blog",
    "Other",
]

BUYABLE_SITE_TYPES: set[str] = {
    "Health & Beauty Media",
    "Blog",
    "News",
}


# ─── Gemini prompt (verbatim from seo_llm with adapted Task line) ──────

CLASSIFICATION_PROMPT = """You are an expert in website classification for content marketing / SEO purposes.

## Task
Classify each domain into ONE category.
Some domains include sample page paths as context — use them to improve accuracy.

## Categories (priority order if ambiguous)

1. **Government** : Government sites, public health organizations
   Examples: ameli.fr, has-sante.fr, service-public.fr, *.gouv.fr, *.gov, who.int

2. **Medical Reference** : Professional medical sources, drug databases
   Examples: vidal.fr, msdmanuals.com, ansm.sante.fr, ema.europa.eu

3. **News** : Newspapers, general news magazines
   Examples: lemonde.fr, lefigaro.fr, 20minutes.fr, bbc.com, nytimes.com

4. **Health & Beauty Media** : Specialized editorial health/wellness/beauty sites with a team of writers or journalists
   Examples: doctissimo.fr, passeportsante.net, topsante.com, healthline.com, beaute-test.com, cosmo.fr

5. **Brand** : Official brand websites — cosmetics, pharma, clinics, medical practices, beauty salons, aesthetic medicine centers
   Examples: eau-thermale-avene.fr, laroche-posay.fr, bioderma.fr, nivea.fr, aesthe.com (aesthetic clinic), cliniqueduparc.fr

6. **E-commerce** : Online retail, pharmacies
   Examples: amazon.fr, shop-pharmacie.fr, pharma-gdd.com, sephora.fr

7. **Encyclopedia** : Collaborative encyclopedic sources
   Examples: wikipedia.org, wikihow.com, britannica.com

8. **Forum** : Discussion spaces, consumer reviews, Q&A sites
   Examples: reddit.com, aufeminin.com/forum, quora.com

9. **Blog** : Personal blogs, individual content creators, lifestyle/beauty bloggers
   Examples: hellgygy.com, mybigbang.fr, fj-beauty.com (personal curly hair blog), monblogbio.com

10. **Other** : If no category fits

## Disambiguation rules
1. Clinics, medical practices, salons, aesthetic centers with their own domain = **Brand** (not Media)
2. Large editorial sites with a team of journalists/writers = **Health & Beauty Media** (not Blog)
3. Individual content creators, personal beauty/lifestyle blogs = **Blog** (not Media)
4. Domain names containing "-beauty"/"-beaute" are ambiguous — check sample page paths if available
5. When sample page paths are provided, use them as strong signals (e.g. /medecine-esthetique... → Brand, /se-debarrasser-silicones... → Blog)

## Rules
- ONE domain = ONE category (most relevant)
- If unsure, prioritize the site's PRIMARY function
- Subdomains inherit from main domain
- For unknown domains, use "Other"

## Domains to classify
{domains_list}

## Response format (strict JSON only, no markdown)
{{"domain1.fr": "Category", "domain2.com": "Category"}}
"""


# ─── Tunables ──────────────────────────────────────────────────────────

_BATCH_SIZE = 30                  # Gemini : 30 domains / call (avoid JSON truncation)
_MAX_PARALLEL_BATCHES = 5         # 5 workers, mirrors seo-llm site_classifier
_BATCH_TIMEOUT_SECONDS = 60       # per-batch Gemini call timeout
_OVERALL_TIMEOUT_SECONDS = 300    # all batches must complete within 5 min


# ─── Public API ────────────────────────────────────────────────────────


def is_buyable(site_type: str | None) -> bool:
    """Return True iff `site_type` is in BUYABLE_SITE_TYPES."""
    return (site_type or "") in BUYABLE_SITE_TYPES


def classify_domains(
    domains: list[str],
    db: Session,
    model: str = "gemini-2.5-flash",
    domain_urls: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Return {domain: site_type}. DB cache lookup first, Gemini for misses.

    All inputs lowercased + www-stripped (same normalization as the rest of
    the codebase). Multiple workers calling this concurrently for overlapping
    domain sets is safe — DB INSERT uses ON CONFLICT DO NOTHING.

    Args:
        domains: bare domains (with or without www, mixed case OK — we
                 normalize). Empty / garbage entries dropped.
        db: SQLAlchemy session (used for cache lookup + writes).
        model: Gemini model name. gemini-2.5-flash is the prod default (fast,
               cheap, accurate for this taxonomy task).
        domain_urls: optional {domain: [url1, url2, ...]} hints — sample page
                 paths help the LLM disambiguate (e.g. /shop/ → E-commerce).

    Returns:
        dict {normalized_domain: site_type}. Domains that couldn't be
        classified (LLM error, missing API key) are NOT in the dict —
        caller should treat missing-from-result as "unknown" (= drop or
        keep depending on policy).

    Side effect: DB inserts for newly-classified domains.
    """
    if not domains:
        return {}

    norm_domains = {_normalize_domain(d) for d in domains}
    norm_domains.discard("")
    if not norm_domains:
        return {}

    # 1. Cache lookup (single query, indexed PK)
    cached_rows = db.execute(
        text("""
            SELECT domain, site_type
            FROM domain_classifications
            WHERE domain = ANY(:d)
        """),
        {"d": list(norm_domains)},
    ).fetchall()
    result: dict[str, str] = {r.domain: r.site_type for r in cached_rows}
    missing = sorted(norm_domains - set(result.keys()))

    if not missing:
        logger.info(
            f"domain_classifier: {len(result)}/{len(norm_domains)} cache hit, "
            f"0 to classify"
        )
        return result

    # 2. Classify misses via Gemini batch
    logger.info(
        f"domain_classifier: {len(result)}/{len(norm_domains)} cache hit, "
        f"{len(missing)} to classify via Gemini ({model})"
    )

    fresh = _classify_with_gemini(missing, model, domain_urls)
    if not fresh:
        logger.warning(
            f"domain_classifier: Gemini classification returned 0 results for "
            f"{len(missing)} domains — they'll be retried next time. Returning cache hits only."
        )
        return result

    # 3. Persist (skip "Other" — seo-llm pattern : retry on next run)
    persisted = 0
    for domain, site_type in fresh.items():
        if site_type == "Other":
            continue  # let next run try again with a fresh batch context
        try:
            db.execute(
                text("""
                    INSERT INTO domain_classifications (domain, site_type, model, source)
                    VALUES (:d, :st, :m, 'gemini')
                    ON CONFLICT (domain) DO NOTHING
                """),
                {"d": domain, "st": site_type, "m": model},
            )
            persisted += 1
        except Exception:
            logger.exception(
                f"domain_classifier: failed to persist {domain}={site_type}"
            )

    if persisted > 0:
        try:
            db.commit()
        except Exception:
            logger.exception("domain_classifier: commit failed")
            db.rollback()

    result.update(fresh)
    logger.info(
        f"domain_classifier: persisted {persisted} new classifications "
        f"({len(fresh) - persisted} 'Other' skipped for retry next time)"
    )
    return result


# ─── Gemini batch internals ────────────────────────────────────────────


def _classify_with_gemini(
    domains: list[str],
    model: str,
    domain_urls: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Batch + parallelize Gemini calls. Returns {domain: site_type}.

    Splits into _BATCH_SIZE batches, dispatches up to _MAX_PARALLEL_BATCHES
    in parallel. Any batch that fails (timeout, API error, parse failure)
    contributes 0 results — caller will retry next time. Robustness over
    optimism : we don't fall back to "Other" for failed domains because
    that'd pollute the cache with bad data.
    """
    if not domains:
        return {}

    # Get a Gemini API key from the worker's existing pool
    api_key = _get_gemini_api_key()
    if not api_key:
        logger.warning(
            "domain_classifier: no GEMINI_API_KEY configured — classification "
            "skipped. Candidates will pass through media_picker unfiltered by "
            "site_type. Set GEMINI_API_KEY (or GEMINI_API_KEYS pool) in worker/.env."
        )
        return {}

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.exception(
            "domain_classifier: google-genai SDK not installed in worker. "
            "Run pip install google-genai>=1.0.0 + rebuild."
        )
        return {}

    client = genai.Client(api_key=api_key)

    batches = [
        domains[i:i + _BATCH_SIZE]
        for i in range(0, len(domains), _BATCH_SIZE)
    ]

    all_results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_BATCHES) as executor:
        future_to_batch = {
            executor.submit(
                _classify_one_batch, batch, client, model, types, domain_urls,
            ): batch
            for batch in batches
        }
        try:
            for future in as_completed(future_to_batch, timeout=_OVERALL_TIMEOUT_SECONDS):
                try:
                    batch_result = future.result(timeout=_BATCH_TIMEOUT_SECONDS)
                    all_results.update(batch_result)
                except Exception:
                    logger.exception("domain_classifier: batch future raised")
        except TimeoutError:
            logger.warning(
                f"domain_classifier: overall timeout (>{_OVERALL_TIMEOUT_SECONDS}s) "
                f"— some batches may have been dropped. Got {len(all_results)} classifications."
            )

    return all_results


def _classify_one_batch(
    domains: list[str],
    client,
    model: str,
    types_module,
    domain_urls: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Classify one batch (~30 domains) via 1 Gemini call. Parse JSON, validate."""
    domains_list = _format_domains_with_urls(domains, domain_urls)
    prompt = CLASSIFICATION_PROMPT.format(domains_list=domains_list)

    try:
        # Disable thinking for gemini-2.5+ (this is a simple classification task,
        # thinking tokens waste cost). Mirrors seo-llm site_classifier.
        thinking_config = None
        if "2.5" in model or "3" in model:
            thinking_config = types_module.ThinkingConfig(thinking_budget=0)

        config = types_module.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
            thinking_config=thinking_config,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        response_text = response.text or ""
        return _parse_gemini_response(response_text, domains)

    except Exception as e:
        logger.warning(
            f"domain_classifier: Gemini batch failed ({len(domains)} domains): {e}"
        )
        return {}


def _parse_gemini_response(response_text: str, domains: list[str]) -> dict[str, str]:
    """Strip markdown fences, parse JSON, validate categories.

    Returns {domain: site_type} only for entries with a valid category.
    Invalid / missing categories are dropped — they'll be retried next time
    (seo-llm robustness pattern : never pollute the cache with bad data).
    """
    text_clean = (response_text or "").strip()
    # Strip markdown code fences if Gemini wrapped its response
    if text_clean.startswith("```"):
        parts = text_clean.split("```")
        if len(parts) >= 2:
            text_clean = parts[1]
            if text_clean.startswith("json"):
                text_clean = text_clean[4:]
    text_clean = text_clean.strip()

    try:
        parsed = json.loads(text_clean)
    except json.JSONDecodeError as e:
        logger.warning(
            f"domain_classifier: JSON parse failed: {e} — "
            f"response head: {text_clean[:200]!r}"
        )
        return {}

    if not isinstance(parsed, dict):
        logger.warning(f"domain_classifier: response is not a dict: {type(parsed)}")
        return {}

    validated: dict[str, str] = {}
    seen_invalid: set[str] = set()
    for domain_raw, site_type in parsed.items():
        domain_norm = _normalize_domain(domain_raw)
        if not domain_norm:
            continue
        if site_type not in SITE_CATEGORIES:
            seen_invalid.add(site_type)
            continue
        validated[domain_norm] = site_type

    if seen_invalid:
        logger.warning(
            f"domain_classifier: dropped {len(seen_invalid)} unknown categor"
            f"y/ies returned by Gemini: {sorted(seen_invalid)[:5]}"
        )

    return validated


# ─── Helpers ───────────────────────────────────────────────────────────


def _normalize_domain(raw: str | None) -> str:
    """Strip scheme, www, path. Return lowercase bare domain or '' on garbage."""
    if not raw:
        return ""
    import re
    s = str(raw).strip().lower()
    s = re.sub(r"^https?://", "", s)
    if s.startswith("www."):
        s = s[4:]
    s = s.split("/", 1)[0].rstrip(".")
    return s if "." in s else ""


def _format_domains_with_urls(
    domains: list[str],
    domain_urls: dict[str, list[str]] | None = None,
) -> str:
    """Format domains list for the prompt, optionally with sample paths."""
    lines = []
    for domain in domains:
        urls = (domain_urls or {}).get(domain) or []
        sample_paths = _extract_sample_paths(urls)
        if sample_paths:
            lines.append(f"- {domain} (sample pages: {', '.join(sample_paths)})")
        else:
            lines.append(f"- {domain}")
    return "\n".join(lines)


def _extract_sample_paths(urls: list[str], max_paths: int = 3) -> list[str]:
    """Pull up to max_paths informative URL paths to help disambiguate."""
    from urllib.parse import urlparse
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        try:
            parsed = urlparse(url)
            path = (parsed.path or "").rstrip("/")
        except Exception:
            continue
        if len(path) < 5 or path in seen:
            continue
        seen.add(path)
        if len(path) > 80:
            path = path[:80] + "..."
        out.append(path)
        if len(out) >= max_paths:
            break
    return out


def _get_gemini_api_key() -> str | None:
    """Pull a Gemini API key from the worker's existing pool (rotated)
    or fall back to the bare GEMINI_API_KEY env var.

    The pool gives us multi-project rotation + cooldown on 429 ; the bare
    env var is the back-compat path used by handlers written before the
    pool existed.
    """
    try:
        from services.gemini_key_pool import get_gemini_pool
        pool = get_gemini_pool()
        if pool.has_keys():
            return pool.next_key()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or None
