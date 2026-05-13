"""Handler: materialize ScanContentItem rows from ScanOpportunity.

Bridge between the scan-analysis pipeline (which writes `ScanOpportunity` rows
keyed on questions) and the content lifecycle Kanban (which operates on
`ScanContentItem` rows keyed on items). Runs at the end of
`generate_opportunities.execute()` so opportunities exist by the time we
materialize.

## What gets materialized

For now, only **FAQ opportunities with priority 'critique' or 'haute'**. This
keeps the Kanban readable (Miller's Law, ~5-10 cards per scan) and aligns
with the Phase B scope (only `generate_faq` handler is wired). Article and
netlinking materialization come later when their handlers ship (Phase C).

## target_url policy — auto-suggest via FAQPageMatcher + manual fallback

We reuse `seo_llm.src.faq_page_matcher.FAQPageMatcher` (same code seo-llm
CLI shipped with) to web_search the user's lead brand domain and pick the
most relevant deep page per question. Outcomes :

  - Match found  → target_url set, target_url_source='auto_suggest'.
                   User can override on the validation page (flips to
                   'user_input').
  - Match empty  → target_url NULL, target_url_source='pending_user'.
                   The validation page surfaces the URL input with a banner
                   so the user can pick a page manually (A2 fallback).
  - No primary
    brand on
    client       → target_url NULL, target_url_source='pending_user'.
                   Same UX as match empty.

The `target_site` for matching is **always the user's lead primary brand
domain**, never the scanned domain — this is the key fix for competitor
scans. On a user-owned scan, lead brand = scan.domain naturally, so the
matcher behaves the seo-llm-canonical way. On a competitor scan (uriage.fr
for a Pierre Fabre user), lead brand = e.g. eau-thermale-avene.fr, so the
matcher finds Avène pages — not Uriage's.

We deliberately read `client.primary_brand_ids` instead of going through
BrandResolver's full resolution chain. The merged chain (scan SBC +
client primary) gets polluted on competitor scans by per-scan
classifications, sometimes including the competitor itself as 'my_brand'
(observed: 98 brands resolved on uriage.fr scan). Workspace primary brands
are the stable signal.

## Idempotency

On rescan, this handler runs again. We dedupe by `(scan_id, content_type,
target_question)` — existing ContentItems are preserved (user may have
already edited them), only NEW questions create new ContentItems. An
opportunity that drops in priority on rescan keeps its old ContentItem; an
opportunity that newly enters 'critique'/'haute' gets a fresh one.
"""

import logging

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Priority threshold for FAQ materialization. Tied to generate_opportunities.py
# scoring : 'critique' = brand absent + competitors present, 'haute' = cited
# but behind competitor. 'moyenne' opportunities are skipped because the user
# already ranks reasonably and the ROI of producing a FAQ is unclear.
_FAQ_PRIORITIES = ("critique", "haute")


def _resolve_target_site(scan, db, item=None) -> tuple[str | None, str | None]:
    """Pick the domain to point FAQPageMatcher at.

    Returns (target_site, lead_brand_name). target_site is None when no
    primary brand has a domain set — caller then skips auto-suggest and the
    user picks manually.

    **Item-level override priority** : when `item.promoted_brand_ids` is set
    (the user picked a non-workspace-default LEAD on the validation page via
    the star toggle), we prefer THAT brand's domain. This makes rematch
    respect the per-item override : if user said "for this opportunity, push
    Aderma not Avène", re-running the matcher must search aderma.fr, not
    eau-thermale-avene.fr. Falls back to workspace default if the item's
    LEAD brand has no domain set (defensive — user might pick a brand with
    a NULL domain in workspace settings).

    Workspace default uses `client.primary_brand_ids` rather than the full
    BrandResolver chain because per-scan SBC classifications can spuriously
    include the scanned competitor itself as 'my_brand' (observed: 98 brands
    resolved on a uriage.fr scan). Workspace primary brands = stable signal.

    The 'lead' iteration finds the first brand whose `domain` field is set
    — PF workspace has 190 primary brands but only ~10 carry domains, so
    strict [0] would fail when [0] is domain-less. Workspace settings UI
    is where the user keeps [0] meaningful.
    """
    from models import Client, ClientBrand

    # Per-item override : the user's explicit choice for THIS content item.
    if item is not None and getattr(item, "promoted_brand_ids", None):
        item_ids = list(item.promoted_brand_ids)
        item_brands = (
            db.query(ClientBrand)
            .filter(ClientBrand.id.in_(item_ids))
            .all()
        )
        by_id_item = {b.id: b for b in item_brands}
        for bid in item_ids:
            b = by_id_item.get(bid)
            if b and b.domain and b.domain.strip():
                return b.domain.strip(), b.name
        # Override set but no brand has a domain → fall through to workspace
        # default rather than aborting the rematch.

    client = db.query(Client).filter(Client.id == scan.client_id).first()
    if not client or not client.primary_brand_ids:
        return None, None

    brands = (
        db.query(ClientBrand)
        .filter(ClientBrand.id.in_(client.primary_brand_ids))
        .all()
    )
    by_id = {b.id: b for b in brands}

    for bid in client.primary_brand_ids:
        b = by_id.get(bid)
        if b and b.domain and b.domain.strip():
            return b.domain.strip(), b.name

    return None, None


def _auto_suggest_leads(items: list, scan, db) -> dict:
    """Wrap services.lead_picker.pick_leads_for_items for the materialize handler.

    `items` is the list of (item_obj, question_text, topic_name) tuples used
    elsewhere in this handler. We unpack into the dict shape pick_leads_for_items
    expects. Failure mode delegates to the service — empty dict means "fall
    back to workspace default", which is exactly what the rest of the pipeline
    already handles.
    """
    if not items:
        return {}
    try:
        from services.lead_picker import pick_leads_for_items
    except Exception as e:
        logger.warning(f"materialize: lead_picker import failed ({e}) — skipping auto-LEAD")
        return {}

    payload = [
        {
            "id": str(item.id),
            "topic": topic_name or "",
            "question": question_text or "",
            "persona": (getattr(item, "persona_name", None) or "") or "",
        }
        for item, question_text, topic_name in items
    ]
    try:
        return pick_leads_for_items(scan.client_id, scan.id, payload, db)
    except Exception as e:
        logger.warning(f"materialize: lead_picker crashed ({e}) — falling back to workspace default")
        return {}


def _auto_match_target_urls(items: list, scan, db) -> dict:
    """Run FAQPageMatcher on a list of (item, question_text) pairs.

    Returns a dict {item_id_str: {"target_page_url": str, "target_page_title": str}}
    for items where a match was found. Items missing from the dict get
    pending_user fallback.

    Failures are swallowed (logged) — the manual A2 path is always available
    as the final fallback, so a transient web_search outage doesn't block
    the scan pipeline.
    """
    if not items:
        return {}

    target_site, lead_name = _resolve_target_site(scan, db)
    if not target_site:
        logger.info(
            f"materialize: skipping auto-suggest for scan {scan.id} — "
            f"no primary brand with a domain set on the client. "
            f"Fix: set client.primary_brand_ids[0..N] to brands that have "
            f"a domain field populated. User picks URLs manually meanwhile."
        )
        return {}

    logger.info(
        f"materialize: auto-suggest target_url for {len(items)} items "
        f"on target_site='{target_site}' (lead brand: {lead_name})"
    )

    # Install the geo_content_generator stub so faq_page_matcher imports cleanly
    # (matches the same pattern used in generate_faq handler).
    from handlers.generate_faq import _install_geo_stub
    _install_geo_stub()

    try:
        import pandas as pd
        from seo_llm.src.faq_page_matcher import FAQPageMatcher
    except Exception as e:
        logger.warning(f"materialize: FAQPageMatcher unavailable ({e}) — falling back to manual")
        return {}

    # Per-item target_site so Phase 1.5's auto LEAD suggestion drives URL
    # matching toward the right brand domain. _resolve_target_site reads
    # item.promoted_brand_ids first when item is passed — set by the lead
    # picker. Falls back to the workspace target_site computed above when the
    # item has no override.
    rows = []
    for item, question_text, source_name in items:
        item_target, _ = _resolve_target_site(scan, db, item=item)
        rows.append({
            "faq_opportunity_id": str(item.id),
            "target_site": item_target or target_site,
            "question_text": question_text,
            "source_name": source_name or "",
        })

    df = pd.DataFrame(rows)
    try:
        matcher = FAQPageMatcher(max_workers=3)  # conservative: rate-limited to ~1/s anyway
        df = matcher.match_pages(df)
    except Exception as e:
        logger.warning(f"materialize: FAQPageMatcher.match_pages crashed ({e}) — falling back to manual")
        return {}

    out: dict = {}
    for _, r in df.iterrows():
        url = (r.get("target_page_url") or "").strip()
        if url:
            out[r["faq_opportunity_id"]] = {
                "target_page_url": _strip_tracking_params(url),
                "target_page_title": (r.get("target_page_title") or "").strip() or None,
            }

    # SEO best-practice : one FAQ per page. When the matcher returns the same
    # URL for multiple semantically-similar questions (common — a brand's
    # XERACALM AD product page is the canonical answer for half a dozen baby-
    # atopic-skin questions), the user will eventually want to either merge
    # them or pick alternate pages. We log the dups here so it's visible in
    # worker logs, and the API surfaces `is_target_url_shared` per item so
    # the validation UI can prompt the user.
    from collections import Counter
    url_counts = Counter(v["target_page_url"] for v in out.values())
    dups = {u: c for u, c in url_counts.items() if c > 1}
    if dups:
        for url, count in dups.items():
            logger.info(
                f"materialize: target_url collision — {count} items share {url!r}; "
                f"validation UI will prompt the user to diversify"
            )

    logger.info(
        f"materialize: auto-suggest results: {len(out)}/{len(items)} matched, "
        f"{len(items) - len(out)} fall back to pending_user, "
        f"{len(dups)} duplicate URL group(s)"
    )
    return out


# Tracking parameter classifiers. We keep these generic so we drop anything
# any search/citation tool (OpenAI web_search, Gemini grounding, Bing, Serper,
# Google Ads click IDs, Facebook click IDs, mailing campaigns, etc.) injects.
# Add more as we see them in the wild — but PREFIX-based rules cover most
# `utm_*` variants automatically.
_TRACKING_PARAM_PREFIXES = ("utm_", "mc_", "ga_")
_TRACKING_PARAM_EXACT = {
    "gclid",      # Google Ads click ID
    "fbclid",     # Facebook click ID
    "msclkid",    # Microsoft Ads click ID
    "yclid",      # Yandex click ID
    "wbraid",     # Google Ads attribution
    "gbraid",     # Google Ads attribution
    "ref",        # Generic referral
    "ref_src",
    "src",        # Twitter / generic source
    "_hsenc",     # HubSpot
    "_hsmi",      # HubSpot
}


def _is_tracking_param(name: str) -> bool:
    n = (name or "").lower()
    if n in _TRACKING_PARAM_EXACT:
        return True
    return any(n.startswith(p) for p in _TRACKING_PARAM_PREFIXES)


def _strip_tracking_params(url: str) -> str:
    """Strip tracking params from a URL so target_url matches the canonical
    page address.

    Citation tools (OpenAI web_search, Gemini grounding, etc.) inject their
    own tracking params (utm_source=openai, utm_source=gemini, ...) into
    returned URLs. Google Ads / Facebook / Microsoft Ads / mailers / HubSpot
    do the same. None of these belong in the FAQ target_url — the user
    expects to publish on the clean canonical URL.

    Uses urlparse + query-key matching (prefix-based for utm_*, ga_*, mc_*;
    exact for click-id-style params). Path/fragment preserved as-is.
    Idempotent + safe on malformed URLs (returns input on parse failure).
    """
    if not url:
        return url
    from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        kept = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not _is_tracking_param(k)
        ]
        new_query = urlencode(kept)
        return urlunparse(parsed._replace(query=new_query))
    except Exception:
        return url


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Read ScanOpportunity rows + create ScanContentItem rows for FAQ targets."""
    from models import (
        Scan,
        ScanContentItem,
        ScanOpportunity,
        ScanQuestion,
    )
    from services.brand_resolver import is_competitor_scan

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError(f"Scan {scan_id} not found")

    # Read FAQ-eligible opportunities for this scan.
    opps = (
        db.query(ScanOpportunity)
        .filter(
            ScanOpportunity.scan_id == scan_id,
            ScanOpportunity.priority.in_(_FAQ_PRIORITIES),
            ScanOpportunity.recommended_action == "faq",
        )
        .all()
    )
    if not opps:
        logger.info(f"materialize_content_items: 0 FAQ opportunities for scan {scan_id}")
        return {"materialized": 0, "skipped_existing": 0, "auto_matched": 0, "is_competitor_scan": False}

    competitor = is_competitor_scan(scan, db)
    logger.info(
        f"materialize_content_items: scan={scan_id}, "
        f"is_competitor={competitor}, eligible_opps={len(opps)}"
    )

    # Pre-load existing FAQ ContentItems for this scan to dedupe by target_question.
    existing = (
        db.query(ScanContentItem)
        .filter(
            ScanContentItem.scan_id == scan_id,
            ScanContentItem.content_type == "faq",
        )
        .all()
    )
    existing_questions = {
        (item.target_question or "").strip().lower() for item in existing if item.target_question
    }

    # Phase 1: create ContentItem rows (without target_url yet) so they get UUIDs
    # we can key the matcher results on.
    new_items: list = []  # list of (item, question_text, source_name)
    skipped = 0

    for opp in opps:
        question = db.query(ScanQuestion).filter(ScanQuestion.id == opp.question_id).first()
        if not question or not (question.question or "").strip():
            logger.debug(f"materialize: skip opp {opp.id} — no question text")
            continue

        q_text = question.question.strip()
        q_key = q_text.lower()
        if q_key in existing_questions:
            skipped += 1
            continue

        item = ScanContentItem(
            scan_id=scan_id,
            content_type="faq",
            topic_name=opp.topic_name,
            persona_name=opp.persona_name,
            target_url=None,
            target_url_source="pending_user",
            target_question=q_text,
            priority=opp.priority,
            opportunity_score=opp.opportunity_score,
            brand_position=opp.brand_position,
            best_competitor=opp.best_competitor_name,
            nb_competitors_cited=opp.nb_competitors_cited,
            status="identified",
        )
        db.add(item)
        new_items.append((item, q_text, opp.topic_name))
        existing_questions.add(q_key)

    if not new_items:
        db.commit()
        logger.info(
            f"materialize_content_items done: scan={scan_id}, "
            f"materialized=0, skipped_existing={skipped}, auto_matched=0"
        )
        return {
            "materialized": 0,
            "skipped_existing": skipped,
            "auto_matched": 0,
            "is_competitor_scan": competitor,
        }

    # Flush so the new items get UUIDs assigned, which the matcher needs as keys.
    db.flush()

    # Phase 1.5: auto-suggest per-item LEAD brand via 1 batched Claude call.
    # Skipped when client has 0 or 1 primary brand with a domain (no choice to
    # make). Sets item.promoted_brand_ids so Phase 2's _resolve_target_site
    # picks the right domain for FAQPageMatcher. content_metadata records
    # provenance so the UI can show an "Auto" chip the user can override.
    lead_suggestions = _auto_suggest_leads(new_items, scan, db)
    auto_lead_count = 0
    if lead_suggestions:
        for item, _, _ in new_items:
            sug = lead_suggestions.get(str(item.id))
            if not sug:
                continue
            item.promoted_brand_ids = [sug["brand_id"]]
            meta = dict(item.content_metadata or {})
            meta["lead_suggestion"] = {
                "brand_id": sug["brand_id"],
                "reason": sug.get("reason") or "",
                "source": "auto",
                "model": sug.get("model"),
            }
            item.content_metadata = meta
            auto_lead_count += 1
        # Re-flush so Phase 2 sees the override on each item.
        db.flush()

    # Phase 2: auto-suggest target_url via FAQPageMatcher on the user's lead brand.
    matches = _auto_match_target_urls(new_items, scan, db)
    auto_matched = 0
    for item, _, _ in new_items:
        m = matches.get(str(item.id))
        if m and m.get("target_page_url"):
            item.target_url = m["target_page_url"]
            item.target_url_source = "auto_suggest"
            if m.get("target_page_title"):
                item.target_page_title = m["target_page_title"]
            auto_matched += 1

    db.commit()

    logger.info(
        f"materialize_content_items done: scan={scan_id}, "
        f"materialized={len(new_items)}, skipped_existing={skipped}, "
        f"auto_matched={auto_matched}, "
        f"pending_user={len(new_items) - auto_matched}, "
        f"auto_lead_suggested={auto_lead_count}, "
        f"is_competitor_scan={competitor}"
    )

    return {
        "materialized": len(new_items),
        "skipped_existing": skipped,
        "auto_matched": auto_matched,
        "auto_lead_suggested": auto_lead_count,
        "is_competitor_scan": competitor,
    }
