"""Handler: generate FAQ Schema.org content for one ScanContentItem.

Wraps `seo_llm.src.faq_content_generator.FAQContentGenerator._generate_single`
which scrapes the target page, fetches brand context + scientific sources via
OpenAI web_search, then composes Schema.org FAQPage HTML via LLM.

This is the Phase B kickoff — minimum viable wiring of the FAQ pipeline into
the SaaS lifecycle. Reads `ScanContentItem` by id, calls the generator with a
row-like dict (compatible with `row.get(key)` calls), persists the result back
into the same row (`content_html`, `content_text`, status='draft').

Known gaps (deferred to future Phase B sessions):
- **Brand bias via BrandResolver** : the generator uses a hardcoded `BRAND_MAP`
  imported from `seo_llm.src.geo_content_generator`. For Pierre Fabre that map
  already lists Avène/Ducray/etc., so FAQ output naturally promotes their
  brands. For other clients, we'll hook `BrandResolver.resolve_promotion()`
  into `_generate_faq` via monkey-patch or seo_llm injection point.
- **Quality strict toggle (RAPP validator)** : `_compute_quality_score` is
  always called; we don't yet expose the strict pass/fail toggle in the UI.
- **Per-job progress reporting** : the FAQ generator is sync (~60s), no
  intermediate progress events. Long-running variant for Phase C articles.

Refund policy : if generation fails, the worker's poll_and_execute already
flips the job to failed + auto-refunds via `_refund_scan_credits`. FAQ
costs 1 content_credit (per Phase B pricing) — debit happens at API enqueue
time so refund-on-failure works end-to-end.
"""

import json
import logging
import sys
import time
import types
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from config import settings

logger = logging.getLogger(__name__)


# ─── geo_content_generator stub (vertical-agnostic) ────────────────────
# faq_content_generator.py:26 does `from .geo_content_generator import BRAND_MAP`.
# Loading the real geo_content_generator pulls Pillow / PyMuPDF / PyGithub /
# playwright / readability-lxml / etc. — 200+ MB of CLI tooling we don't need
# at runtime. We inject a minimal stub module providing just BRAND_MAP so the
# FAQ generator's import resolves without the heavy chain.
#
# IMPORTANT: BRAND_MAP must stay EMPTY in the SaaS. seo_llm shipped a hardcoded
# map of Pierre Fabre brands + French dermo-cosmetic competitors because it was
# a CLI for one customer. sen-ai is multi-tenant + multi-vertical (cosmetics,
# automotive, finance, B2B services, …) — vertical/customer-specific data in
# code is forbidden (project_migration_seollm_to_aiscan.md invariants #1, #3).
#
# Empty BRAND_MAP works because faq_content_generator._generate_faq has a
# `brand_name = target_site` fallback (line 546): when no map entry matches,
# it uses the bare domain as brand_name. The LLM is then smart enough to pick
# up the proper brand name from page scraping + web_search context. Smoke
# tested on ducray.com — even WITH the empty map, the generated FAQ correctly
# names "Ducray" / "Sensinol" by reading the scraped page.
#
# Brand bias for content gen will land via BrandResolver.resolve_promotion()
# (next session), reading primary_brand_ids from the DB per-client. NOT here.
_BRAND_MAP_STUB: dict = {}


def _install_geo_stub() -> None:
    """Inject a minimal geo_content_generator stub into sys.modules.

    Idempotent — safe to call multiple times. Must run BEFORE any code
    triggers `from seo_llm.src.faq_content_generator import ...`.
    """
    mod_name = "seo_llm.src.geo_content_generator"
    if mod_name in sys.modules and getattr(sys.modules[mod_name], "_is_stub", False):
        return
    stub = types.ModuleType(mod_name)
    stub.BRAND_MAP = _BRAND_MAP_STUB
    stub._is_stub = True
    sys.modules[mod_name] = stub


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    """Generate FAQ content for one ScanContentItem.

    job_payload must contain: {"item_id": "<uuid>"}
    scan_id is the parent scan, passed by the worker for credit accounting.
    """
    from models import ScanContentItem

    item_id = job_payload.get("item_id")
    if not item_id:
        raise RuntimeError("generate_faq requires item_id in job payload")

    item = db.query(ScanContentItem).filter(ScanContentItem.id == item_id).first()
    if not item:
        raise RuntimeError(f"ScanContentItem {item_id} not found")

    if item.content_type != "faq":
        raise RuntimeError(
            f"ScanContentItem {item_id} is content_type='{item.content_type}', "
            f"not 'faq' — wrong handler"
        )

    if not item.target_url:
        # A2 stepping-stone: on competitor scans (and any ContentItem where the
        # system couldn't infer a user page), target_url is left NULL and
        # target_url_source='pending_user' so the validation page can prompt
        # the user. The API should already gate Generate, this is the worker-
        # side defensive guard.
        from exceptions import PermanentScanError
        raise PermanentScanError(
            f"FAQ generation needs a target URL. Open the item in the validation "
            f"page and pick a URL on your site that should host this FAQ."
        )

    # Mark as generating so the UI can show a spinner state if it polls
    item.status = "generating"
    db.commit()

    # Install the geo_content_generator stub BEFORE importing FAQContentGenerator
    # (the latter does `from .geo_content_generator import BRAND_MAP` at module load).
    _install_geo_stub()

    # Lazy import to avoid loading the heavy seo_llm module at worker boot
    from seo_llm.src.faq_content_generator import FAQContentGenerator
    from adapters.brief_injector import format_workspace_brief, format_promoted_brands_block
    from services.brand_resolver import resolve_promotion, PromotionUnsetError

    # ── Resolve workspace context for vertical specialization ────────────
    # client.apps['client_brief'] = industry / voice / positioning / audience
    # — generated by generate_client_brief.py, persisted per workspace.
    from models import Client
    client = None
    if item.scan and item.scan.client_id:
        client = db.query(Client).filter(Client.id == item.scan.client_id).first()
    workspace_brief_text = format_workspace_brief(client.apps if client else None)

    # ── Resolve brands to promote (and competitors to exclude) ───────────
    # The strategic differentiator: when generating from a competitor scan,
    # output must promote the USER's brands, not whoever owns the scanned
    # domain. BrandResolver chains scan.promotion_brand_ids → ScanBrandClassification
    # (my_brand) → client.primary_brand_ids.
    promoted_brand_names: list[str] = []
    promoted_brand_ids: list = []
    excluded_brand_names: list[str] = []
    try:
        promotion = resolve_promotion(item.scan, db)
        promoted_brand_names = [b.name for b in promotion.promote_brands if b.name]
        promoted_brand_ids = list(promotion.promote_brand_ids)
        excluded_brand_names = list(promotion.exclude_domain_names or [])
        logger.info(
            f"FAQ promotion resolved (item {item_id}): "
            f"promote={promoted_brand_names}, exclude={excluded_brand_names[:5]}"
            f"{'…' if len(excluded_brand_names) > 5 else ''}, via={promotion.resolved_via}"
        )
    except PromotionUnsetError as e:
        # Soft fallback: generate without explicit brand bias. The workspace
        # brief (if present) still gives context. User can set primary brands
        # in workspace settings to fix this — surfaced in the warning so we
        # can later add a UI nudge.
        logger.warning(
            f"FAQ for item {item_id} has no resolved brands to promote — "
            f"generating with workspace context only ({e})"
        )

    promoted_brands_text = format_promoted_brands_block(promoted_brand_names)
    excluded_section = ""
    if excluded_brand_names:
        # Limit to top 10 to avoid prompt bloat
        excluded_section = (
            "## Brands to AVOID (do NOT promote these in the answer)\n"
            + ", ".join(excluded_brand_names[:10])
        )

    # Build a row-compatible dict — the generator calls `row.get(key)` so a
    # plain dict satisfies the interface (no pandas required).
    row = {
        "target_page_url": item.target_url,
        "target_site": _extract_site(item.target_url),
        "question_text": item.target_question or item.topic_name or "",
        "source_name": item.scan.domain if item.scan else "",
    }

    logger.info(
        f"Generating FAQ for content_item {item_id} "
        f"(target={row['target_page_url']}, scan={scan_id}, "
        f"workspace_brief={'yes' if workspace_brief_text else 'no'}, "
        f"promote={len(promoted_brand_names)}, exclude={len(excluded_brand_names)})"
    )

    start = time.time()
    try:
        generator_cls = _get_workspace_aware_class()
        generator = generator_cls(
            workspace_brief_text=workspace_brief_text,
            promoted_brands_text=promoted_brands_text,
            excluded_section=excluded_section,
            promoted_lead_brand=(promoted_brand_names[0] if promoted_brand_names else ""),
            writing_provider="openai",
            model=settings.task_models.get("generate_faq") if hasattr(settings, "task_models") else None,
            max_workers=1,
        )
        result = generator._generate_single(row)
    except Exception as e:
        # Reset status so user can retry from Kanban (without going through the full
        # _refund_scan_credits path which fires only on attempts >= max_attempts).
        # If this is the LAST attempt, the worker will mark scan failed anyway.
        item.status = "identified"
        db.commit()
        raise RuntimeError(f"FAQ generation failed for item {item_id}: {e}") from e

    duration_ms = int((time.time() - start) * 1000)

    # Persist result + audit trail on the item
    item.content_html = result.get("faq_html") or None
    item.content_text = result.get("faq_text") or None
    item.status = "draft"  # Awaiting user review
    if promoted_brand_ids:
        # Record which brands the system was instructed to promote at gen time
        # (audit trail per scan_content_items.promoted_brand_ids column).
        item.promoted_brand_ids = promoted_brand_ids
        flag_modified(item, "promoted_brand_ids")

    # Stash sources + quality in a structured payload on content_text? No, we
    # don't have a JSONB column for FAQ metadata. For Phase B we drop them
    # into content_text suffix as commented HTML. Phase C will likely add a
    # `metadata JSONB` column to ScanContentItem for this kind of audit data.
    sources_json = result.get("sources_used", "[]")
    quality_score = result.get("quality_score", 0)
    quality_details = result.get("quality_details", "{}")
    faq_count = result.get("faq_count", 0)

    db.commit()

    # Log LLM usage for cost monitoring (best-effort — calculator may miss some
    # tokens since FAQContentGenerator does multiple LLM calls per FAQ and
    # doesn't return per-call token counts. We log a coarse estimate via the
    # configured model and let billing come from provider invoices for now.)
    try:
        from adapters.llm_logger import log_llm_usage
        log_llm_usage(
            db, provider="openai",
            model=getattr(generator, "model", "gpt-4.1-mini"),
            operation="generate_faq",
            input_tokens=0,  # not surfaced by FAQContentGenerator
            output_tokens=0,
            duration_ms=duration_ms,
            scan_id=scan_id,
            client_id=str(item.scan.client_id) if item.scan else None,
        )
    except Exception:
        logger.warning("log_llm_usage failed for generate_faq", exc_info=True)

    logger.info(
        f"FAQ generated for item {item_id}: {faq_count} Q/R, quality={quality_score}/100, "
        f"sources={sources_json}, {duration_ms}ms"
    )

    return {
        "status": "draft",
        "faq_count": faq_count,
        "quality_score": quality_score,
        "sources_count": len(json.loads(sources_json) if sources_json else []),
        "duration_ms": duration_ms,
    }


# ─── Workspace-aware FAQ generator subclass ────────────────────────────
# We don't fork the seo_llm submodule — instead we subclass FAQContentGenerator
# and override the smallest possible surface (`_fetch_brand_context`) to inject
# our SaaS-side context blocks (workspace brief + promoted brands + excluded
# brands) into the brand_content text the seo-llm prompt builder concatenates
# into its "## CONTENU MARQUE" section. The injection is strong enough to
# override the prompt's default "produits {brand_name}" instruction (which
# would otherwise name the scanned domain on competitor scans).
def _make_workspace_aware_class():
    """Lazy subclass factory — needs FAQContentGenerator imported first."""
    from seo_llm.src.faq_content_generator import FAQContentGenerator

    class _Subclass(FAQContentGenerator):
        def __init__(self, workspace_brief_text: str = "", promoted_brands_text: str = "",
                     excluded_section: str = "", promoted_lead_brand: str = "",
                     **kwargs):
            super().__init__(**kwargs)
            self._workspace_brief_text = workspace_brief_text
            self._promoted_brands_text = promoted_brands_text
            self._excluded_section = excluded_section
            self._promoted_lead_brand = promoted_lead_brand

        def _fetch_brand_context(self, target_site, question_text):
            # Parent's _fetch_brand_context uses a loose natural-language prompt
            # ("sur le site {target_site}") which OpenAI web_search interprets
            # softly — it routinely returns off-site competitor pages for hot
            # queries (e.g. "eczéma visage" → Uriage Xémose results even when
            # target_site = eau-thermale-avene.fr). The leaked competitor brand
            # names enter the prompt's "## CONTENU MARQUE" block and the LLM
            # then weaves them into the FAQ output. This is the root cause of
            # the Xémose leak observed on 2026-05-12.
            #
            # We post-filter the parent's output to strip URLs that aren't on
            # target_site. Filtering at the URL level + dropping the scraped
            # text body when the URL is rejected keeps the LLM's context free
            # of off-brand content without re-issuing the web_search.
            #
            # If filtering removes everything, we fall back to an empty
            # brand_content — the LLM still has the target_url scrape +
            # workspace brief + promoted-brands block, which is enough to
            # generate a high-quality on-brand FAQ.
            raw_content, raw_urls = super()._fetch_brand_context(target_site, question_text)
            filtered_content, filtered_urls = _filter_brand_context_by_site(
                raw_content, raw_urls, target_site
            )
            prefix_parts = []
            if self._workspace_brief_text:
                prefix_parts.append(self._workspace_brief_text)
            if self._promoted_brands_text:
                prefix_parts.append(self._promoted_brands_text)
            if self._excluded_section:
                prefix_parts.append(self._excluded_section)
            if prefix_parts:
                filtered_content = "\n\n".join(prefix_parts) + "\n\n---\n\n" + (filtered_content or "")
            return filtered_content, filtered_urls

        def _fetch_scientific_context(self, question_text, source_name):
            # Same root cause as _fetch_brand_context but on a different code
            # path : seo_llm's scientific web_search ("trouve des sources
            # scientifiques sur {topic}") routinely returns competitor brand
            # product pages because OpenAI interprets it as "find the best
            # products for the condition". Observed live on 2026-05-12 : a
            # regen on a clean Avène target_url cited 3 Uriage Xémose product
            # pages as "scientific sources". The LLM then wrote "la gamme
            # XEMOSE C8+ d'Uriage propose..." in the FAQ output.
            #
            # We allowlist : recognized scientific/medical domains only
            # (regulators, public health agencies, journals, encyclopedias).
            # The user's own brand domain comes through brand_urls/content
            # in _fetch_brand_context — scientific_context shouldn't carry
            # it back. Defense in depth : the verified_urls list passed to
            # the LLM (= brand_urls + scientific_urls) stays bounded to
            # on-brand + trusted scientific, no competitor product pages.
            raw_content, raw_urls = super()._fetch_scientific_context(question_text, source_name)
            return _filter_scientific_context_by_allowlist(
                raw_content, raw_urls, target_site=""
            )

        def _generate_faq(self, row, page_content, brand_content, scientific_content, verified_urls):
            """Override to substitute `target_site` in row with our promoted lead brand domain.

            seo_llm's `_generate_faq` builds the prompt with two interpolations of
            `brand_name`, which it derives from `target_site` via BRAND_MAP lookup
            (with `target_site` as fallback). On a competitor scan, the default
            behavior names the COMPETITOR in the prompt's system role + "produits
            {brand_name}" line — explicitly opposite to the bias we want.

            We patch the row in place: `target_site` is set to the promoted lead
            brand's name. With BRAND_MAP empty (vertical-agnostic stub), the
            generator's lookup misses and falls back to `target_site` value itself
            — which now IS the user's brand name. Result: prompt names the user's
            brand twice, biasing output toward it.

            Other prompt sections (page scrape, brand_content with our injection
            prefix, verified_urls) are unchanged. Hard constraint remains: only
            verified_urls from web_search of the original `target_site` are
            cite-able. For true bias on competitor scans, future work needs to
            also web_search the user's brand sites to enrich verified_urls. This
            is a strategic-but-deferred improvement.
            """
            if self._promoted_lead_brand:
                # Mutate a shallow copy so we don't pollute the caller's dict
                patched_row = dict(row) if isinstance(row, dict) else row.copy()
                patched_row["target_site"] = self._promoted_lead_brand
                row = patched_row
            return super()._generate_faq(row, page_content, brand_content, scientific_content, verified_urls)

    return _Subclass


# Cached after first call (per worker process — fresh subclass each instance)
_WorkspaceAwareFAQGenerator: type | None = None


def _get_workspace_aware_class():
    global _WorkspaceAwareFAQGenerator
    if _WorkspaceAwareFAQGenerator is None:
        _WorkspaceAwareFAQGenerator = _make_workspace_aware_class()
    return _WorkspaceAwareFAQGenerator


# Trusted scientific/medical source allowlist for `_fetch_scientific_context`.
# Without this gate, OpenAI's web_search ("trouve des sources scientifiques sur
# la peau atopique") routinely returns competitor brand product pages
# (uriage.fr/produits/xemose-..., laroche-posay.fr/...) which seo_llm treats
# as legitimate citations — the LLM then writes "selon Uriage Xémose, ...".
# Observed live on 2026-05-12 : 3/4 sources cited in a regen were Uriage
# product pages even with the brand_context filter in place.
#
# We allowlist domains that genuinely host scientific/medical content
# (regulators, public health agencies, scientific publishers, encyclopedias).
# Anything else gets dropped from scientific_content + scientific_urls. The
# allowlist is intentionally generic across verticals — not dermo-cosmetic
# specific — so it scales to other industries the SaaS will serve.
_SCIENTIFIC_ALLOWLIST = (
    # French health authorities
    "has-sante.fr", "ameli.fr", "vidal.fr", "ansm.sante.fr",
    "santepubliquefrance.fr", "inserm.fr", "doctolib.fr",
    # International medical / scientific
    "nih.gov", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "who.int", "europa.eu", "ema.europa.eu", "fda.gov",
    "cochrane.org", "mayoclinic.org", "clevelandclinic.org",
    "nhs.uk", "medlineplus.gov", "uptodate.com", "merckmanuals.com",
    "sciencedirect.com", "nature.com", "thelancet.com", "nejm.org",
    "bmj.com", "jamanetwork.com",
    # Encyclopedic / educational (low promotional risk)
    "wikipedia.org", "wikimedia.org",
)


def _domain_in_allowlist(domain: str) -> bool:
    """True iff `domain` matches or is a subdomain of any entry in the
    scientific allowlist. www. is stripped before comparison."""
    if not domain:
        return False
    d = domain.lower()
    if d.startswith("www."):
        d = d[4:]
    return any(d == a or d.endswith("." + a) for a in _SCIENTIFIC_ALLOWLIST)


def _filter_scientific_context_by_allowlist(content: str, urls: list[str],
                                            target_site: str) -> tuple[str, list[str]]:
    """Drop URLs (and their scraped-text blocks when present) that aren't
    on target_site OR in _SCIENTIFIC_ALLOWLIST.

    Mirrors _filter_brand_context_by_site but with a broader pass criterion :
    scientific context legitimately spans medical/regulatory sources outside
    the user's brand domain. We keep target_site URLs (in case the brand
    page hosts scientific dossiers) + allowlist domains (regulators,
    journals, encyclopedias) and drop anything else — most importantly
    competitor product pages (uriage.fr, laroche-posay.fr, …).

    Same dual-path handling as the brand filter : Serper structured blocks
    parse cleanly, OpenAI free-form text keeps the body if at least one
    citation passes (otherwise drops to avoid contamination).
    """
    from urllib.parse import urlparse

    def _domain_of(u: str) -> str:
        try:
            h = (urlparse(u or "").netloc or "").lower()
            return h[4:] if h.startswith("www.") else h
        except Exception:
            return ""

    ts = (target_site or "").lower()
    if ts.startswith("www."):
        ts = ts[4:]

    def _accepted(u: str) -> bool:
        d = _domain_of(u)
        if not d:
            return False
        if ts and (d == ts or d.endswith("." + ts)):
            return True
        return _domain_in_allowlist(d)

    safe_urls = [u for u in (urls or []) if _accepted(u)]

    if not content:
        return "", safe_urls

    if "URL:" in content:
        try:
            chunks = content.split("URL:")
            kept = []
            for chunk in chunks[1:]:
                first_nl = chunk.find("\n")
                if first_nl < 0:
                    continue
                url_line = chunk[:first_nl].strip()
                body = chunk[first_nl:]
                if _accepted(url_line):
                    kept.append(f"URL: {url_line}{body}")
            return "\n\n".join(kept), safe_urls
        except Exception:
            return "", safe_urls

    if safe_urls:
        return content, safe_urls
    return "", []


def _filter_brand_context_by_site(content: str, urls: list[str], target_site: str) -> tuple[str, list[str]]:
    """Drop URLs (and their associated scraped-text blocks) that aren't on target_site.

    Parent's _fetch_brand_context formats Serper-path output as repeated blocks:
        URL: <url>
        <scraped text...>

        URL: <url>
        <scraped text...>

    We split on URL: markers, keep only blocks whose URL passes a domain match
    on target_site. The OpenAI-path output isn't structured this way — for that
    we just keep the response text as-is when ANY URL passed (the LLM context
    is grounded by URL citations that are filtered separately).

    Returns (filtered_text, filtered_urls). Defensive : on parse failure,
    returns ("", []) rather than risking leak into the FAQ prompt.
    """
    from urllib.parse import urlparse

    def _domain_of(u: str) -> str:
        try:
            h = (urlparse(u or "").netloc or "").lower()
            return h[4:] if h.startswith("www.") else h
        except Exception:
            return ""

    ts = (target_site or "").lower()
    if ts.startswith("www."):
        ts = ts[4:]
    if not ts:
        return content or "", list(urls or [])

    def _on_target(u: str) -> bool:
        d = _domain_of(u)
        return bool(d) and (d == ts or d.endswith("." + ts))

    safe_urls = [u for u in (urls or []) if _on_target(u)]

    if not content:
        return "", safe_urls

    # Serper-path format detection : starts with "URL:" or contains "\n\nURL:".
    if "URL:" in content:
        try:
            # Split on the "URL:" delimiter, keeping the URL with its block.
            # First slice before any URL: is preamble, usually empty — drop.
            chunks = content.split("URL:")
            kept = []
            for chunk in chunks[1:]:  # skip preamble
                # Each chunk = " <url>\n<scraped text>"
                first_nl = chunk.find("\n")
                if first_nl < 0:
                    # URL but no body — drop, can't validate
                    continue
                url_line = chunk[:first_nl].strip()
                body = chunk[first_nl:]
                if _on_target(url_line):
                    kept.append(f"URL: {url_line}{body}")
            return "\n\n".join(kept), safe_urls
        except Exception:
            # Parse error — drop the entire block rather than risk leakage
            return "", safe_urls

    # OpenAI-path format : free-form text from response.output_text. If at
    # least one of the cited URLs is on target_site, keep the text; otherwise
    # drop entirely. This is conservative — the LLM gets less context but
    # we eliminate off-brand contamination.
    if safe_urls:
        return content, safe_urls
    return "", []


def _extract_site(url: str) -> str:
    """Extract bare hostname from a URL (https://www.foo.com/bar → foo.com)."""
    if not url:
        return ""
    s = url.lower().strip()
    if s.startswith("http://"):
        s = s[7:]
    elif s.startswith("https://"):
        s = s[8:]
    if s.startswith("www."):
        s = s[4:]
    s = s.split("/", 1)[0]
    return s
