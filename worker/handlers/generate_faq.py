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
            brand_content, brand_urls = super()._fetch_brand_context(target_site, question_text)
            prefix_parts = []
            if self._workspace_brief_text:
                prefix_parts.append(self._workspace_brief_text)
            if self._promoted_brands_text:
                prefix_parts.append(self._promoted_brands_text)
            if self._excluded_section:
                prefix_parts.append(self._excluded_section)
            if prefix_parts:
                brand_content = "\n\n".join(prefix_parts) + "\n\n---\n\n" + (brand_content or "")
            return brand_content, brand_urls

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
