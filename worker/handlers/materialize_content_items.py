"""Handler: materialize ScanContentItem rows from ScanOpportunity.

Bridge between the scan-analysis pipeline (which writes `ScanOpportunity` rows
keyed on questions) and the content lifecycle Kanban (which operates on
`ScanContentItem` rows keyed on items). Runs at the end of
`generate_opportunities.execute()` so opportunities exist by the time we
materialize.

## What gets materialized

Opportunities with priority 'critique' or 'haute' AND a recommended_action
mapped to a known content_type (see `_CONTENT_TYPE_BY_ACTION`) :

  - `recommended_action='faq'`        → `content_type='faq'`
  - `recommended_action='netlinking'` → `content_type='netlinking_article'`
                                        (Phase C.1 — handler wired)

Both go through `_auto_suggest_leads` (per-item LEAD brand selection via
Claude). FAQ items additionally go through `_auto_match_target_urls` to
suggest a page on the brand's own site. **Netlinking items skip the matcher
entirely** : the article will be published on a third-party media domain
(not the user's brand site), so a brand-site sitemap match makes no sense.
The user fills `target_url` manually on the validation page (URL of the
media partner where the article will be published).

`content_update` opportunities (priority 'haute' fallback) are NOT
materialized — no handler exists for them yet.

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


# Priority threshold for materialization. Tied to generate_opportunities.py
# scoring : 'critique' = brand absent + competitors present, 'haute' = cited
# but behind competitor. 'moyenne' opportunities are skipped because the user
# already ranks reasonably and the ROI of producing content is unclear.
_CONTENT_PRIORITIES = ("critique", "haute")
# Alias kept for any external import; same value.
_FAQ_PRIORITIES = _CONTENT_PRIORITIES

# Map opportunity action → ScanContentItem.content_type. Extend this dict (+
# wire the handler in worker/main.py + api dispatcher) to materialize new
# content types. NB: generate_opportunities.py uses the SHORT form for the
# action label ("netlinking") while ScanContentItem uses the LONG form
# ("netlinking_article") — this map is the translation table between the two.
_CONTENT_TYPE_BY_ACTION = {
    "faq": "faq",
    "netlinking": "netlinking_article",
}


def _resolve_lead_brand(scan, db, item=None):
    """Return the lead ClientBrand object (domain + id + name etc.) using
    the same priority chain as _resolve_target_site.

    Phase D wiring needs the brand_id (not just the domain) to query the
    sitemap-index corpus. _resolve_target_site stays as the thin domain-
    string wrapper so existing call sites are unchanged.

    Returns None when no primary brand has a domain set.
    """
    from models import Client, ClientBrand

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
                return b

    client = db.query(Client).filter(Client.id == scan.client_id).first()
    if not client or not client.primary_brand_ids:
        return None

    brands = (
        db.query(ClientBrand)
        .filter(ClientBrand.id.in_(client.primary_brand_ids))
        .all()
    )
    by_id = {b.id: b for b in brands}

    for bid in client.primary_brand_ids:
        b = by_id.get(bid)
        if b and b.domain and b.domain.strip():
            return b
    return None


def _resolve_target_site(scan, db, item=None) -> tuple[str | None, str | None]:
    """Pick the domain to point FAQPageMatcher at.

    Thin wrapper over `_resolve_lead_brand` that returns (domain, name) so
    existing call sites stay unchanged.

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
    """Match each item to its best target_url.

    Two-layer cascade :
      1. **Sitemap-index matcher** (Phase D) — semantic match against the
         brand's embedded sitemap corpus. When the top-1 score clears
         SITEMAP_THRESHOLD (default 0.55), we take it and persist the
         top-3 candidates + the score. Source='sitemap_index'.
      2. **FAQPageMatcher** (legacy web_search) — fallback for items
         where sitemap returned nothing or scored below threshold. Same
         behavior as pre-Phase-D. Source='auto_suggest'.

    Returns a dict per-item :
        {
          "target_page_url": str,
          "target_page_title": str | None,
          "target_url_source": "sitemap_index" | "auto_suggest",
          "target_url_score": float | None,             # set by sitemap matcher
          "target_url_candidates": list[dict] | None,   # top-3 if sitemap
        }

    Items missing from the dict get pending_user fallback. Failures are
    swallowed (logged) — the manual A2 path is always available.
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

    # ── Layer 1 : sitemap-index matcher (Phase D) ─────────────────────
    out: dict[str, dict] = {}
    remaining: list = []          # items that fell below threshold (Layer 2)

    try:
        from services.sitemap_matcher import (
            SITEMAP_THRESHOLD, find_best_pages, slugify_brand_name,
        )
        from config import settings
        sitemap_enabled = bool(settings.openai_api_key)
    except Exception as exc:
        logger.warning(f"materialize: sitemap_matcher import failed ({exc}) — skipping Layer 1")
        sitemap_enabled = False
        SITEMAP_THRESHOLD = 0.55  # noqa: F841 — value unused if disabled

    sitemap_hits = 0
    sitemap_below_threshold = 0
    sitemap_no_corpus = 0
    if sitemap_enabled:
        for item, question_text, _source_name in items:
            lead_brand = _resolve_lead_brand(scan, db, item=item)
            if not lead_brand:
                remaining.append((item, question_text, _source_name))
                continue
            gamme_slug = (
                slugify_brand_name(lead_brand.name)
                if lead_brand.parent_id else None
            )
            try:
                matches = find_best_pages(
                    question_text=question_text or "",
                    client_brand_id=str(lead_brand.id),
                    db=db,
                    openai_api_key=settings.openai_api_key,
                    top_k=3,
                    gamme_slug=gamme_slug,
                )
            except Exception as exc:
                logger.exception(
                    f"materialize: sitemap_matcher.find_best_pages crashed for "
                    f"item={item.id} brand={lead_brand.id}: {exc} — falling back"
                )
                remaining.append((item, question_text, _source_name))
                continue

            if not matches:
                sitemap_no_corpus += 1
                remaining.append((item, question_text, _source_name))
                continue
            top1 = matches[0]
            if top1["score"] < SITEMAP_THRESHOLD:
                sitemap_below_threshold += 1
                remaining.append((item, question_text, _source_name))
                continue

            out[str(item.id)] = {
                "target_page_url": _strip_tracking_params(top1["url"]),
                "target_page_title": top1.get("title"),
                "target_url_source": "sitemap_index",
                "target_url_score": float(top1["score"]),
                "target_url_candidates": [
                    {
                        "url": _strip_tracking_params(m["url"]),
                        "title": m.get("title"),
                        "score": float(m["score"]),
                        "inlink_count": int(m.get("inlink_count") or 0),
                    }
                    for m in matches[:3]
                ],
            }
            sitemap_hits += 1
        logger.info(
            f"materialize sitemap Layer 1: hits={sitemap_hits} "
            f"below_threshold={sitemap_below_threshold} no_corpus={sitemap_no_corpus} "
            f"threshold={SITEMAP_THRESHOLD}"
        )
    else:
        remaining = list(items)

    # ── Layer 2 : FAQPageMatcher fallback (legacy web_search) ─────────
    if not remaining:
        return out

    # Install the geo_content_generator stub so faq_page_matcher imports cleanly
    from handlers.generate_faq import _install_geo_stub
    _install_geo_stub()

    try:
        import pandas as pd
        from seo_llm.src.faq_page_matcher import FAQPageMatcher
    except Exception as e:
        logger.warning(
            f"materialize: FAQPageMatcher unavailable ({e}) — Layer 2 "
            f"fallback skipped, {len(remaining)} items go pending_user"
        )
        return out

    # Per-item target_site so Phase 1.5's auto LEAD suggestion drives URL
    # matching toward the right brand domain.
    rows = []
    for item, question_text, source_name in remaining:
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
        return out

    for _, r in df.iterrows():
        url = (r.get("target_page_url") or "").strip()
        if url:
            out[r["faq_opportunity_id"]] = {
                "target_page_url": _strip_tracking_params(url),
                "target_page_title": (r.get("target_page_title") or "").strip() or None,
                "target_url_source": "auto_suggest",
                "target_url_score": None,
                "target_url_candidates": None,
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
    """Read ScanOpportunity rows + create ScanContentItem rows.

    Handles both FAQ (`recommended_action='faq'` → `content_type='faq'`) and
    netlinking article (`recommended_action='netlinking'` →
    `content_type='netlinking_article'`) opportunities. See module docstring
    for the matcher-skip rationale for netlinking items.
    """
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

    # Read all materializable opportunities for this scan (FAQ + netlinking).
    eligible_actions = list(_CONTENT_TYPE_BY_ACTION.keys())
    opps = (
        db.query(ScanOpportunity)
        .filter(
            ScanOpportunity.scan_id == scan_id,
            ScanOpportunity.priority.in_(_CONTENT_PRIORITIES),
            ScanOpportunity.recommended_action.in_(eligible_actions),
        )
        .all()
    )
    if not opps:
        logger.info(
            f"materialize_content_items: 0 eligible opportunities "
            f"(actions in {eligible_actions}) for scan {scan_id}"
        )
        return {
            "materialized": 0,
            "materialized_by_type": {},
            "skipped_existing": 0,
            "auto_matched": 0,
            "is_competitor_scan": False,
        }

    competitor = is_competitor_scan(scan, db)
    logger.info(
        f"materialize_content_items: scan={scan_id}, "
        f"is_competitor={competitor}, eligible_opps={len(opps)}"
    )

    # Pre-load existing ContentItems for this scan to dedupe by
    # (content_type, target_question). Allows the same question to have BOTH
    # a FAQ and a netlinking_article variant if generate_opportunities ever
    # emits both actions for the same question — unlikely today (action is
    # exclusive per the ternary in generate_opportunities.py:84) but the
    # tuple-key dedupe is the correct semantic and future-proof.
    existing = (
        db.query(ScanContentItem)
        .filter(
            ScanContentItem.scan_id == scan_id,
            ScanContentItem.content_type.in_(_CONTENT_TYPE_BY_ACTION.values()),
        )
        .all()
    )
    existing_keys: set[tuple[str, str]] = {
        (item.content_type, (item.target_question or "").strip().lower())
        for item in existing if item.target_question
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

        ct = _CONTENT_TYPE_BY_ACTION.get(opp.recommended_action)
        if not ct:
            # Defensive : the IN filter above already guarantees this can't
            # happen, but keep the guard so future actions added to the table
            # don't silently produce items with a stale content_type.
            logger.warning(
                f"materialize: opp {opp.id} action='{opp.recommended_action}' "
                f"not in _CONTENT_TYPE_BY_ACTION — skipping"
            )
            continue

        q_text = question.question.strip()
        q_key = q_text.lower()
        if (ct, q_key) in existing_keys:
            skipped += 1
            continue

        item = ScanContentItem(
            scan_id=scan_id,
            content_type=ct,
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
        existing_keys.add((ct, q_key))

    if not new_items:
        db.commit()
        logger.info(
            f"materialize_content_items done: scan={scan_id}, "
            f"materialized=0, skipped_existing={skipped}, auto_matched=0"
        )
        return {
            "materialized": 0,
            "materialized_by_type": {},
            "skipped_existing": skipped,
            "auto_matched": 0,
            "is_competitor_scan": competitor,
        }

    # Flush so the new items get UUIDs assigned, which the matcher needs as keys.
    db.flush()

    # Phase 1.5: auto-suggest per-item LEAD brand via 1 batched Claude call.
    # Skipped when client has 0 or 1 primary brand with a domain (no choice to
    # make). Sets item.promoted_brand_ids so Phase 2's _resolve_target_site
    # picks the right domain for FAQPageMatcher (FAQ items) and the article
    # handler's BrandResolver injection. content_metadata records provenance
    # so the UI can show an "Auto" chip the user can override.
    #
    # Runs on ALL items regardless of content_type — LEAD brand selection is
    # relevant for both FAQ (where it drives the matcher target_site) and
    # netlinking_article (where it drives the promoted brand in the article).
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

    # Phase 2: auto-suggest target_url. ONLY for FAQ items.
    #
    # Netlinking articles get published on third-party media domains (e.g.
    # doctissimo.fr), not on the user's brand site, so neither the sitemap-
    # index matcher (which scans brand-site corpora) nor the FAQPageMatcher
    # web_search (which targets brand sub-paths) yields a meaningful result.
    # The user fills `target_url` manually on the validation page with the
    # URL of the media partner where the article will be published.
    faq_items = [t for t in new_items if t[0].content_type == "faq"]
    matches = _auto_match_target_urls(faq_items, scan, db) if faq_items else {}
    auto_matched = 0
    sitemap_matched = 0
    for item, _, _ in faq_items:
        m = matches.get(str(item.id))
        if m and m.get("target_page_url"):
            item.target_url = m["target_page_url"]
            # New : honor the layer-specific source ('sitemap_index' or 'auto_suggest')
            # returned by the matcher. Default keeps legacy behavior.
            item.target_url_source = m.get("target_url_source") or "auto_suggest"
            if m.get("target_page_title"):
                item.target_page_title = m["target_page_title"]
            # Phase D : persist score + top-3 candidates when sitemap matcher fired.
            if m.get("target_url_score") is not None:
                item.target_url_score = m["target_url_score"]
            if m.get("target_url_candidates"):
                item.target_url_candidates = m["target_url_candidates"]
            if item.target_url_source == "sitemap_index":
                sitemap_matched += 1
            auto_matched += 1

    db.commit()

    # Per-type counts for the log + return payload — useful when scanning
    # logs to spot a regression where one content_type stops materializing.
    by_type: dict[str, int] = {}
    for item, _, _ in new_items:
        by_type[item.content_type] = by_type.get(item.content_type, 0) + 1

    logger.info(
        f"materialize_content_items done: scan={scan_id}, "
        f"materialized={len(new_items)} {by_type}, "
        f"skipped_existing={skipped}, "
        f"auto_matched={auto_matched}/{len(faq_items)} FAQ "
        f"(sitemap_index={sitemap_matched}, "
        f"auto_suggest={auto_matched - sitemap_matched}), "
        f"pending_user={len(new_items) - auto_matched}, "
        f"auto_lead_suggested={auto_lead_count}, "
        f"is_competitor_scan={competitor}"
    )

    return {
        "materialized": len(new_items),
        "materialized_by_type": by_type,
        "skipped_existing": skipped,
        "auto_matched": auto_matched,
        "sitemap_matched": sitemap_matched,
        "auto_lead_suggested": auto_lead_count,
        "is_competitor_scan": competitor,
    }
