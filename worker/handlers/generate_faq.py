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

    # Cap-then-call : a FAQ runs ~$0.005 ; pass a defensive $0.05 projection
    # so a near-cap client doesn't trip mid-call. Raises BudgetExceeded —
    # propagated to the worker retry chain (will retry up to max_attempts,
    # then refund the content_credit via CONTENT_ITEM_JOB_TYPES branch).
    from services.llm_budget import assert_within_budget
    client_id_for_budget = item.scan.client_id if item.scan else None
    assert_within_budget(client_id_for_budget, db, projected_cost_usd=0.05)

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
    from services.trust_sources import get_trust_sources_for_client
    from services.competitor_domains import get_competitor_domains_for_scan

    # ── Resolve workspace context for vertical specialization ────────────
    # client.apps['client_brief'] = industry / voice / positioning / audience
    # — generated by generate_client_brief.py, persisted per workspace.
    from models import Client, ClientBrand
    client = None
    if item.scan and item.scan.client_id:
        client = db.query(Client).filter(Client.id == item.scan.client_id).first()
    # Phase BB : pick the focus brand to surcharge the workspace brief with.
    # Priority : item.scan.focus_brand_id (explicit Gate-2 selection) →
    # first promoted brand id (resolved below) — but we don't have promotion
    # resolved yet at this point, so fall back here on focus_brand_id only.
    # Promotion-driven override is applied a few lines below.
    focus_brand_brief: dict | None = None
    focus_brand_for_logging = None
    if item.scan and getattr(item.scan, "focus_brand_id", None):
        fb = (
            db.query(ClientBrand)
            .filter(ClientBrand.id == item.scan.focus_brand_id)
            .first()
        )
        if fb and fb.brief:
            focus_brand_brief = fb.brief
            focus_brand_for_logging = fb.name
    workspace_brief_text = format_workspace_brief(
        client.apps if client else None,
        focus_brand_brief,
    )

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

    # ── Per-item override (validation-page star toggle + auto LEAD suggest) ──
    # `item.promoted_brand_ids` carries the user's most recent intent for THIS
    # specific opportunity (set by PATCH /content-items/{id} when the user
    # clicks the ☆ star, OR by materialize_content_items when the LLM auto-
    # picks a LEAD). It must win over workspace-level resolution otherwise
    # Regenerate quietly reverts the LEAD + writes content for the wrong
    # brand. We REORDER the existing promotion list so the chosen LEAD goes
    # first ; un-overridden brands keep their workspace position behind it,
    # so the model still has co-promote signal.
    item_override_ids = [str(b) for b in (getattr(item, "promoted_brand_ids", None) or [])]
    if item_override_ids and promoted_brand_ids:
        from models import ClientBrand
        # Resolve names for any override IDs not already in the workspace
        # promotion list (e.g. user picked a workspace brand that resolve_
        # promotion didn't include because primary_brand_ids was tighter than
        # workspace catalog). Defensive — the PATCH validation already gates
        # IDs against client.primary_brand_ids.
        existing_id_strs = {str(b) for b in promoted_brand_ids}
        unknown_ids = [bid for bid in item_override_ids if bid not in existing_id_strs]
        if unknown_ids:
            extra_rows = (
                db.query(ClientBrand)
                .filter(ClientBrand.id.in_(unknown_ids))
                .all()
            )
            by_id_extra = {str(b.id): b for b in extra_rows}
            for bid in unknown_ids:
                b = by_id_extra.get(bid)
                if b and b.name:
                    promoted_brand_ids.append(b.id)
                    promoted_brand_names.append(b.name)

        # Reorder : every override ID first (in override order), then the
        # remaining workspace IDs in their original order.
        by_id_pos = {str(b): i for i, b in enumerate(promoted_brand_ids)}
        ordered_ids = []
        ordered_names = []
        seen: set[str] = set()
        for bid in item_override_ids:
            pos = by_id_pos.get(bid)
            if pos is None or bid in seen:
                continue
            ordered_ids.append(promoted_brand_ids[pos])
            ordered_names.append(promoted_brand_names[pos])
            seen.add(bid)
        for i, b in enumerate(promoted_brand_ids):
            if str(b) in seen:
                continue
            ordered_ids.append(b)
            ordered_names.append(promoted_brand_names[i])
            seen.add(str(b))
        if ordered_names and ordered_names[0] != promoted_brand_names[0]:
            logger.info(
                f"FAQ promotion: per-item override applied (item {item_id}) — "
                f"LEAD={ordered_names[0]} (workspace default LEAD was {promoted_brand_names[0]})"
            )
        promoted_brand_ids = ordered_ids
        promoted_brand_names = ordered_names

    promoted_brands_text = format_promoted_brands_block(promoted_brand_names)
    excluded_section = ""
    if excluded_brand_names:
        # Limit to top 10 to avoid prompt bloat
        excluded_section = (
            "## Brands to AVOID (do NOT promote these in the answer)\n"
            + ", ".join(excluded_brand_names[:10])
        )

    # ── Resolve trust sources (SOFT prefer-signal) ───────────────────────
    # Trust sources are no longer a hard filter — they're a prefer-hint
    # injected into the scientific_context search prompt so OpenAI biases
    # toward authoritative domains. URLs NOT on the trust list still pass
    # the filter (the hard filter is the competitor denylist + universal
    # e-commerce/social patterns, see services.url_filter).
    trust_domains: list[str] = []
    if client:
        try:
            trust_domains = get_trust_sources_for_client(client.id, db)
        except Exception:
            logger.exception(
                f"get_trust_sources_for_client failed for client {client.id} "
                f"— FAQ will run without prefer-hint (filter still applies)"
            )

    # ── Resolve competitor domains (HARD denylist) ───────────────────────
    # Per-scan brands classified as 'competitor' in scan_brand_classifications.
    # Joined with client_brands.domain to yield the bare domain set we drop
    # from web_search outputs no matter what. This is the strategic
    # differentiator's enforcement mechanism : guaranteed clean on
    # competitor scans, deterministic and DB-backed (not LLM-dependent).
    competitor_domains: set[str] = set()
    if item.scan and item.scan.id:
        try:
            competitor_domains = get_competitor_domains_for_scan(item.scan.id, db)
        except Exception:
            logger.exception(
                f"get_competitor_domains_for_scan failed for scan {item.scan.id} "
                f"— FAQ will run with universal e-commerce/social denylist only"
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
            trust_domains=trust_domains,
            competitor_domains=competitor_domains,
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

    sources_json = result.get("sources_used", "[]")
    quality_score = result.get("quality_score", 0)
    faq_count = result.get("faq_count", 0)
    try:
        sources_raw = json.loads(sources_json) if sources_json else []
    except Exception:
        sources_raw = []

    # ── Persist generation audit on item.content_metadata (migration 024) ──
    # quality_score, sources cited (enriched with org names so the UI can
    # show "Société Française de Dermatologie" instead of "sfd.asso.fr"),
    # plus the scientific_context denylist diagnostic (raw vs kept, drop
    # reasons). The latter makes the brand-bias defense auditable from the
    # UI — "0 sources dropped as competitors" is exactly the transparency
    # the strategic differentiator promises.
    trust_details_by_domain: dict[str, dict] = {}
    if client:
        trust_payload = (client.apps or {}).get("trust_sources") or {}
        for entry in trust_payload.get("details") or []:
            d = (entry.get("domain") or "").strip().lower()
            if d:
                trust_details_by_domain[d] = entry

    target_site = (row.get("target_site") or "").lower()
    if target_site.startswith("www."):
        target_site = target_site[4:]
    primary_lead_brand = promoted_brand_names[0] if promoted_brand_names else ""

    def _enrich_source(url: str) -> dict:
        from urllib.parse import urlparse
        try:
            netloc = (urlparse(url).netloc or "").lower()
        except Exception:
            netloc = ""
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if target_site and (netloc == target_site or netloc.endswith("." + target_site)):
            return {"url": url, "domain": netloc, "org": primary_lead_brand or netloc,
                    "type": "brand_site"}
        match = trust_details_by_domain.get(netloc)
        if not match:
            # try parent-domain match (e.g. sub.domain.tld → domain.tld)
            parts = netloc.split(".")
            for i in range(len(parts) - 1):
                cand = ".".join(parts[i + 1:])
                if cand in trust_details_by_domain:
                    match = trust_details_by_domain[cand]
                    break
        if match:
            return {"url": url, "domain": netloc, "org": match.get("org") or netloc,
                    "type": match.get("type") or "reference"}
        return {"url": url, "domain": netloc, "org": netloc, "type": "other"}

    sources_enriched = [_enrich_source(u) for u in sources_raw if u]
    sci_diag = getattr(generator, "scientific_diagnostic", {}) or {}

    item.content_metadata = {
        "quality_score": int(quality_score) if quality_score is not None else 0,
        "faq_count": int(faq_count) if faq_count is not None else 0,
        "sources_used": sources_enriched,
        "sources_count": len(sources_enriched),
        "competitor_drops": int(sci_diag.get("drop_reasons", {}).get("competitor", 0)),
        "drop_reasons": sci_diag.get("drop_reasons", {}),
        "scientific_kept": int(sci_diag.get("kept_count", 0)),
        "scientific_raw": int(sci_diag.get("raw_count", 0)),
        "generated_at": datetime.utcnow().isoformat(),
        "duration_ms": duration_ms,
        "generator_version": "denylist-prefer-hint-v1",
    }
    flag_modified(item, "content_metadata")

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
                     trust_domains: list[str] | None = None,
                     competitor_domains: set[str] | None = None,
                     **kwargs):
            super().__init__(**kwargs)
            self._workspace_brief_text = workspace_brief_text
            self._promoted_brands_text = promoted_brands_text
            self._excluded_section = excluded_section
            self._promoted_lead_brand = promoted_lead_brand
            self._trust_domains = list(trust_domains or [])
            # HARD denylist : per-scan competitor brand domains. Source of
            # truth = scan_brand_classifications (deterministic, DB-backed,
            # NOT LLM-discovered). This is the mechanism that enforces the
            # strategic differentiator "no competitor citation on competitor
            # scans". Universal e-commerce / social patterns are layered on
            # top inside services.url_filter.is_excluded_url().
            self._competitor_domains: set[str] = set(competitor_domains or set())
            # Per-call diagnostic captured by _fetch_scientific_context and
            # surfaced in execute() so it lands in scan_content_items.content_metadata.
            self.scientific_diagnostic: dict = {}

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
            # Two-layer strategy :
            #
            # 1. SOFT prefer-signal : inject the per-client trust source
            #    domains into the question text so OpenAI's web_search prompt
            #    biases toward authoritative sites (HAS, ANSM, journals, …).
            #    The LLM still surfaces other relevant URLs — we don't lose
            #    coverage when the trust list is incomplete (the historical
            #    pain : allowlist dropped ameli.fr / vidal.fr / inserm.fr
            #    that weren't on the discovered list, leaving 1 citation
            #    and quality_score 72).
            #
            # 2. HARD post-filter denylist : drop URLs on per-scan competitor
            #    brand domains (from scan_brand_classifications) + universal
            #    e-commerce / cart / social / blog patterns. This is the
            #    deterministic, auditable enforcement of the brand-bias
            #    promise — guaranteed clean regardless of what the LLM
            #    decides to retrieve.
            #
            # Combined : high recall on legitimate scientific sources +
            # strict prevention of competitor citations. Replaces the prior
            # allowlist approach (2026-05-12 → 2026-05-13) which traded
            # recall for false-strict precision.
            from services.url_filter import partition_urls, format_drop_summary

            augmented_question = question_text
            if self._trust_domains:
                # Only the first ~8 domains — keep the prompt tight; the
                # rest are universal TLD patterns which OpenAI naturally
                # prefers anyway for queries about health / public-sector
                # / regulated topics.
                domains_str = ", ".join(self._trust_domains[:8])
                augmented_question = (
                    f"{question_text}\n\n"
                    f"PRIORITISE authoritative sources from these reference "
                    f"domains when available : {domains_str}. Other "
                    f"government, regulatory or peer-reviewed sources are "
                    f"also acceptable. AVOID commercial product pages, brand "
                    f"e-commerce sites, blogs and forums."
                )

            raw_content, raw_urls = super()._fetch_scientific_context(
                augmented_question, source_name
            )

            safe_urls, dropped = partition_urls(
                raw_urls, competitor_domains=self._competitor_domains,
            )
            # Stash structured diagnostic for execute() to persist on the
            # item. Counts only — full dropped URL lists stay in the log to
            # keep the JSONB payload small.
            self.scientific_diagnostic = {
                "raw_count": len(raw_urls or []),
                "kept_count": len(safe_urls),
                "drop_reasons": {k: len(v) for k, v in dropped.items()},
            }
            if dropped:
                logger.warning(
                    f"scientific_context denylist filter: kept={len(safe_urls)} / "
                    f"raw={len(raw_urls or [])} — dropped: {format_drop_summary(dropped)}"
                )
            else:
                logger.info(
                    f"scientific_context denylist filter: kept={len(safe_urls)} / "
                    f"raw={len(raw_urls or [])} — no drops"
                )

            # Rebuild content body keeping only sections whose URL passed.
            # Mirror the parent's Serper-block format on the way back so
            # downstream FAQ prompt stitching is unchanged.
            safe_set = set(safe_urls)
            filtered_content = _strip_blocks_by_url(raw_content, safe_set) \
                if raw_content else ""

            return filtered_content, safe_urls

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


# Block-level filter helper for `_fetch_scientific_context`.
#
# Parent's _fetch_scientific_context (Serper mode) formats output as repeated
# "URL: <url>\n<scraped body>" blocks. After we've decided which URLs to keep
# (via services.url_filter.partition_urls applied with the per-scan competitor
# denylist + universal e-commerce/social patterns), we need to also strip the
# scraped body of any rejected URL so its content doesn't bleed into the FAQ
# prompt's "## SOURCES SCIENTIFIQUES" block.
#
# The OpenAI-path output isn't structured the same way — it's free text with
# URL citations referenced inline. In that case the kept_urls list is what
# gets passed to the FAQ generation prompt as `verified_urls`, and the
# content text is kept as-is if any URL passed (we don't try to surgically
# remove competitor mentions from free-form text — defense in depth comes
# from the verified_urls bound).


def _strip_blocks_by_url(content: str, safe_urls: set[str]) -> str:
    """Keep only "URL: ...\\n<body>" blocks whose URL is in `safe_urls`.

    If the content doesn't look like Serper-format blocks (no "URL:" marker),
    returns the content as-is when at least one URL passed, else empty string.
    Defensive on parse failure : returns empty rather than risk leak.
    """
    if not content:
        return ""
    if "URL:" not in content:
        return content if safe_urls else ""

    try:
        chunks = content.split("URL:")
        kept: list[str] = []
        for chunk in chunks[1:]:
            first_nl = chunk.find("\n")
            if first_nl < 0:
                continue
            url_line = chunk[:first_nl].strip()
            body = chunk[first_nl:]
            if url_line in safe_urls:
                kept.append(f"URL: {url_line}{body}")
        return "\n\n".join(kept)
    except Exception:
        return ""


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
