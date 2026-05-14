"""Content items lifecycle endpoints — Kanban + validation page backend.

ScanContentItem rows are the unit of content (FAQ or article) generated from
opportunities discovered during AI scans. This router exposes:

- GET  /api/clients/{client_id}/content-items   → Kanban list, filterable
- GET  /api/content-items/{id}                  → single item full detail
- PATCH /api/content-items/{id}                 → status / validation / content edit

RBAC piggybacks on the existing scans.py `_check_scan_access` — items belong to
a scan, so scan-level access controls apply transitively. Editor role is auto-
required on PATCH via the request_method contextvar middleware (see scans.py).

Status workflow (mapped to Kanban columns):
    identified, generating       → "TO CREATE" column
    draft, in_review             → "IN REVIEW" column
    approved                     → "APPROVED" column
    published                    → "PUBLISHED" column
    rejected                     → hidden by default (filter to show)
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.attributes import flag_modified

from models import (
    ClientBrand, Job, Scan, ScanBrandClassification, ScanContentItem,
    ScanLLMResult, ScanQuestion, UserClient, get_db,
)
from routers.scans import _check_scan_access
from services.audit import audit_log
from services.auth_service import get_current_user

router = APIRouter()


# ── Handler dispatch by content_type ──────────────────────────────────
GENERATOR_BY_CONTENT_TYPE = {
    "faq": "generate_faq",
    # "netlinking_article": "generate_article" — Phase C
}


# ── Status → Kanban column mapping ──────────────────────────────────────
COLUMN_BY_STATUS = {
    "identified": "to_create",
    "generating": "to_create",
    "draft": "in_review",
    "in_review": "in_review",
    "approved": "approved",
    "published": "published",
    "rejected": "rejected",  # hidden by default
}

# Priority sort order (critique → haute → moyenne → null)
PRIORITY_RANK = {"critique": 0, "haute": 1, "moyenne": 2}


def _check_client_access(client_id: str, user, db: Session):
    """Lightweight client-level RBAC for the Kanban list endpoint."""
    link = db.query(UserClient).filter(
        UserClient.user_id == user.id,
        UserClient.client_id == client_id,
    ).first()
    if not link:
        raise HTTPException(403, "Access denied")


def _resolve_scan_brand_groups(scan_id: str, db: Session) -> dict[str, list[str]]:
    """Pull all brand names per SBC classification for a scan.

    Used by the validation UI to color-code brand mentions in the rendered
    content : own brands = coral chip (promotion working), competitors = red
    chip (leak warning). Also exposes 'all_known' (raw brand + aliases) so
    the matcher catches product-line variants like 'XERACALM A.D' without
    needing them in promoted_brand_ids.

    Returns {"my_brand": [...], "competitor": [...], "all_known": [...]}.
    Aliases are flattened into all_known so the matching picks them up too.
    """
    rows = (
        db.query(ScanBrandClassification, ClientBrand)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(ScanBrandClassification.scan_id == scan_id)
        .all()
    )
    out: dict[str, list[str]] = {"my_brand": [], "competitor": [], "all_known": []}
    seen_my: set[str] = set()
    seen_comp: set[str] = set()
    seen_all: set[str] = set()
    for sbc, brand in rows:
        name = (brand.name or "").strip()
        if not name:
            continue
        key = name.lower()
        if sbc.classification == "my_brand" and key not in seen_my:
            seen_my.add(key)
            out["my_brand"].append(name)
        elif sbc.classification == "competitor" and key not in seen_comp:
            seen_comp.add(key)
            out["competitor"].append(name)
        # all_known includes my_brand + competitor + ignored + aliases. The UI
        # falls back on this when promoted_brand_names is empty (older items).
        if key not in seen_all:
            seen_all.add(key)
            out["all_known"].append(name)
        for alias in (brand.aliases or []):
            a = (alias or "").strip()
            if not a:
                continue
            akey = a.lower()
            if akey in seen_all:
                continue
            seen_all.add(akey)
            out["all_known"].append(a)
            # Aliases inherit the parent classification for color-coding.
            if sbc.classification == "my_brand" and akey not in seen_my:
                seen_my.add(akey)
                out["my_brand"].append(a)
            elif sbc.classification == "competitor" and akey not in seen_comp:
                seen_comp.add(akey)
                out["competitor"].append(a)
    return out


def _serialize_item(item: ScanContentItem, brand_names: dict[str, str] | None = None,
                    primary_brand_domains: list[str] | None = None,
                    target_url_share_count: int = 0,
                    scan_brand_groups: dict[str, list[str]] | None = None,
                    competitor_snapshot: dict | None = None) -> dict:
    """Convert a ScanContentItem ORM row into the dict shape the UI consumes.

    `primary_brand_domains` is the list of domains from the client's
    `primary_brand_ids` — used client-side to validate user-edited target_url
    against the brands the user is meant to promote.

    `target_url_share_count` is how many OTHER content items in this scan
    target the same URL. >0 means the user should consider diversifying
    (one FAQ per page is the SEO best practice).
    """
    promoted_names: list[str] = []
    if item.promoted_brand_ids and brand_names:
        for bid in item.promoted_brand_ids:
            name = brand_names.get(str(bid))
            if name:
                promoted_names.append(name)

    scan = item.scan
    groups = scan_brand_groups or {"my_brand": [], "competitor": [], "all_known": []}
    return {
        "primary_brand_domains": primary_brand_domains or [],
        "target_url_share_count": target_url_share_count,
        "is_target_url_shared": target_url_share_count > 0,
        # Per-scan brand classifications for inline color-coded highlights :
        # own brands = coral (promotion landing), competitors = red (leak alarm).
        # Aliases inherited from the brand catalog so gammes/variants resolve too.
        "own_brand_names": list(groups.get("my_brand") or []),
        "competitor_brand_names": list(groups.get("competitor") or []),
        "all_known_brand_names": list(groups.get("all_known") or []),
        "id": str(item.id),
        "scan_id": str(item.scan_id),
        "scan_domain": scan.domain if scan else None,
        "scan_name": scan.name if scan else None,
        "content_type": item.content_type,
        "topic_name": item.topic_name,
        "persona_name": item.persona_name,
        "target_url": item.target_url,
        "target_url_source": item.target_url_source,
        "target_url_score": item.target_url_score,
        "target_url_candidates": list(item.target_url_candidates or []),
        "target_page_title": item.target_page_title,
        "target_question": item.target_question,
        "rejected_target_urls": list(item.rejected_target_urls or []),
        "content_metadata": dict(item.content_metadata or {}),
        "content_html": item.content_html,
        "content_text": item.content_text,
        "article_outline": item.article_outline,
        "gdrive_doc_url": item.gdrive_doc_url,
        "priority": item.priority,
        "opportunity_score": item.opportunity_score,
        "brand_position": item.brand_position,
        "best_competitor": item.best_competitor,
        "nb_competitors_cited": item.nb_competitors_cited,
        "estimated_price": item.estimated_price,
        "platform_link": item.platform_link,
        "promoted_brand_ids": [str(b) for b in (item.promoted_brand_ids or [])],
        "promoted_brand_names": promoted_names,
        "status": item.status,
        "column": COLUMN_BY_STATUS.get(item.status, "to_create"),
        "validation": item.validation,
        "validated_by": item.validated_by,
        "validated_at": item.validated_at.isoformat() if item.validated_at else None,
        "identified_at": item.identified_at.isoformat() if item.identified_at else None,
        "ordered_at": item.ordered_at.isoformat() if item.ordered_at else None,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "published_url": item.published_url,
        "latest_position": item.latest_position,
        "position_delta": item.position_delta,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        # Phase E Pilier 5 : 'what LLMs currently say' snapshot. None for items
        # whose target_question can't be matched to a ScanQuestion or whose
        # parent scan has no LLM results (legacy items, pre-tracking pipeline).
        # Surfaced only on the detail endpoint to keep the kanban list cheap.
        "competitor_snapshot": competitor_snapshot,
    }


def _resolve_brand_names(client_id: str, item_brand_ids: set[str], db: Session) -> dict[str, str]:
    """Fetch brand names for promoted_brand_ids in a single query (avoids N+1)."""
    if not item_brand_ids:
        return {}
    rows = (
        db.query(ClientBrand)
        .filter(ClientBrand.client_id == client_id, ClientBrand.id.in_(item_brand_ids))
        .all()
    )
    return {str(b.id): b.name for b in rows}


def _resolve_primary_brand_domains(client_id: str, db: Session) -> list[str]:
    """Return the domains of the client's primary brands, ordered by primary_brand_ids.

    Used by the UI to validate user-edited target_url against the brands the
    user is meant to promote. Empty list when the client has no primary brand
    configured yet (legitimate state — the UI then skips off-brand warnings).
    """
    from models import Client

    client = db.query(Client).filter(Client.id == client_id).first()
    if not client or not client.primary_brand_ids:
        return []
    rows = (
        db.query(ClientBrand)
        .filter(
            ClientBrand.client_id == client_id,
            ClientBrand.id.in_(client.primary_brand_ids),
        )
        .all()
    )
    by_id = {b.id: b for b in rows}
    seen: set[str] = set()
    domains: list[str] = []
    for bid in client.primary_brand_ids:
        b = by_id.get(bid)
        if not b or not b.domain:
            continue
        normalized = b.domain.lower().strip()
        if normalized.startswith("www."):
            normalized = normalized[4:]
        if normalized in seen:
            continue
        seen.add(normalized)
        domains.append(normalized)
    return domains


# ── Endpoints ───────────────────────────────────────────────────────────

@router.get("/clients/{client_id}/content-items")
async def list_content_items(
    client_id: str,
    status: str | None = Query(None, description="Filter by single status"),
    content_type: str | None = Query(None, description="'faq' or 'netlinking_article'"),
    scan_id: str | None = Query(None, description="Filter by scan id"),
    include_rejected: bool = Query(False, description="Include rejected items"),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List ScanContentItem rows for a client, ordered by priority then identified_at desc."""
    _check_client_access(client_id, user, db)

    q = (
        db.query(ScanContentItem)
        .options(joinedload(ScanContentItem.scan))
        .join(Scan, Scan.id == ScanContentItem.scan_id)
        .filter(Scan.client_id == client_id)
    )
    if status:
        q = q.filter(ScanContentItem.status == status)
    if content_type:
        q = q.filter(ScanContentItem.content_type == content_type)
    if scan_id:
        q = q.filter(ScanContentItem.scan_id == scan_id)
    if not include_rejected:
        q = q.filter((ScanContentItem.status != "rejected") | (ScanContentItem.status.is_(None)))

    items = q.all()

    # Sort: priority rank asc (critique=0 first), then identified_at desc
    items.sort(key=lambda i: (
        PRIORITY_RANK.get(i.priority, 99),
        -(i.identified_at.timestamp() if i.identified_at else 0),
    ))

    # Resolve brand names for promoted_brand_ids in one query
    all_brand_ids: set[str] = set()
    for it in items:
        for bid in (it.promoted_brand_ids or []):
            all_brand_ids.add(str(bid))
    brand_names = _resolve_brand_names(client_id, all_brand_ids, db)
    primary_domains = _resolve_primary_brand_domains(client_id, db)

    # Build per-scan target_url frequency map so each card knows whether its
    # URL is shared with sibling items (SEO best-practice : one FAQ per page).
    from collections import Counter
    per_scan_url_counts: dict[str, Counter] = {}
    for it in items:
        if it.target_url:
            per_scan_url_counts.setdefault(str(it.scan_id), Counter())[it.target_url] += 1

    def _share_count_for(it: ScanContentItem) -> int:
        if not it.target_url:
            return 0
        counter = per_scan_url_counts.get(str(it.scan_id))
        if not counter:
            return 0
        # "Other items sharing this URL" — exclude self
        return max(0, counter[it.target_url] - 1)

    serialized = [
        _serialize_item(it, brand_names, primary_domains, _share_count_for(it))
        for it in items
    ]

    # Pre-bucket by Kanban column for the UI (saves a JS reduce pass)
    by_column: dict[str, list[dict]] = {
        "to_create": [], "in_review": [], "approved": [], "published": [], "rejected": [],
    }
    for d in serialized:
        col = d["column"]
        if col in by_column:
            by_column[col].append(d)

    return {
        "items": serialized,
        "by_column": by_column,
        "counts": {col: len(rows) for col, rows in by_column.items()},
        "total": len(serialized),
    }


@router.get("/content-items/{item_id}")
async def get_content_item(item_id: str, user=Depends(get_current_user),
                           db: Session = Depends(get_db)):
    """Return a single content item with full detail + resolved brand names."""
    item = (
        db.query(ScanContentItem)
        .options(joinedload(ScanContentItem.scan))
        .filter(ScanContentItem.id == item_id)
        .first()
    )
    if not item:
        raise HTTPException(404, "Content item not found")

    # RBAC via parent scan
    _check_scan_access(str(item.scan_id), user, db)

    client_id_str = str(item.scan.client_id)
    brand_names = _resolve_brand_names(
        client_id_str,
        {str(b) for b in (item.promoted_brand_ids or [])},
        db,
    )
    primary_domains = _resolve_primary_brand_domains(client_id_str, db)
    share_count = _count_sibling_items_at_url(item, db)
    groups = _resolve_scan_brand_groups(str(item.scan_id), db)
    competitor_snapshot = _build_competitor_snapshot(item, db)
    return _serialize_item(item, brand_names, primary_domains, share_count, groups, competitor_snapshot)


def _build_competitor_snapshot(item: ScanContentItem, db: Session) -> dict | None:
    """Build the Pilier 5 'side-by-side' payload : what the LLMs currently say
    about this question, with the competitor highlighted.

    The user's FAQ answers `item.target_question`. The same question was tested
    in the original scan against multiple LLM providers (ChatGPT, Gemini, ...)
    and their raw responses are stored in `scan_llm_results.response_text`.
    Surfacing them next to our generated FAQ on the validation page lets the
    user SEE concretely what they're trying to displace — and whether the
    FAQ they're about to publish actually beats the citation pattern that
    favors the current best_competitor.

    Lookup chain : item.target_question -> ScanQuestion (text match in this
    scan, case-insensitive) -> ScanLLMResult rows for that question_id.

    Returns None when :
      - the question can't be resolved (legacy items, manual entry)
      - no LLM results exist (scan still running, or pre-tracking pipeline)
    Caller treats None as "panel doesn't render" — graceful degrade.

    Shape :
      {
        "question_text": str,
        "competitor_brand_name": str | None,         # from item.best_competitor
        "competitor_position": int | None,
        "responses": [
          {
            "provider": str,                          # 'openai' | 'gemini' | ...
            "model": str,
            "response_text": str,                     # full LLM answer
            "citations": [{url, domain, source_type, title}],
            "competitor_cited": bool,                 # any citation domain matches a competitor
            "our_brand_cited": bool,                  # from target_cited flag
            "our_brand_position": int | None,
          },
          ...
        ],
      }
    """
    q_text = (item.target_question or "").strip()
    if not q_text:
        return None

    question = (
        db.query(ScanQuestion)
        .filter(
            ScanQuestion.scan_id == item.scan_id,
            func.lower(ScanQuestion.question) == q_text.lower(),
        )
        .first()
    )
    if not question:
        return None

    # Latest row per provider — refreshes append new rows tagged with a
    # fresh created_at, the old rows linger in the table for the audit
    # trail (and for Pilier 7's before/after delta). We only surface the
    # MOST RECENT one per provider in the panel.
    from sqlalchemy import and_
    latest = (
        db.query(
            ScanLLMResult.provider,
            func.max(ScanLLMResult.created_at).label("max_at"),
        )
        .filter(ScanLLMResult.question_id == question.id)
        .group_by(ScanLLMResult.provider)
        .subquery()
    )
    llm_rows = (
        db.query(ScanLLMResult)
        .join(
            latest,
            and_(
                ScanLLMResult.provider == latest.c.provider,
                ScanLLMResult.created_at == latest.c.max_at,
            ),
        )
        .filter(ScanLLMResult.question_id == question.id)
        .order_by(ScanLLMResult.provider.asc())
        .all()
    )
    if not llm_rows:
        return None

    responses = []
    for r in llm_rows:
        if not (r.response_text or "").strip():
            continue
        comp_domains = r.competitor_domains or {}
        responses.append({
            "id": str(r.id),
            "provider": r.provider,
            "model": r.model,
            "response_text": r.response_text,
            "citations": list(r.citations or []),
            "competitor_cited": bool(comp_domains),
            "competitor_domains_count": int(sum(comp_domains.values())) if comp_domains else 0,
            "our_brand_cited": bool(r.target_cited),
            "our_brand_position": r.target_position,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    if not responses:
        return None

    # Latest snapshot date across all providers — drives the freshness chip
    # in the UI ('AI snapshot · 14 days ago'). Per-response timestamps stay
    # in `responses[].created_at` for finer-grained display if needed.
    latest_at = max(
        (r.created_at for r in llm_rows if r.created_at),
        default=None,
    )

    # Check if there's an in-flight refresh job for this item — UI uses
    # this to disable the Refresh button and poll for completion.
    in_flight_refresh_job = (
        db.query(Job)
        .filter(
            Job.client_id == item.scan.client_id if item.scan else None,
            Job.job_type == "refresh_ai_snapshot",
            Job.status.in_(("pending", "running")),
            Job.payload["item_id"].astext == str(item.id),
        )
        .order_by(Job.created_at.desc())
        .first()
    )

    return {
        "question_text": q_text,
        "competitor_brand_name": item.best_competitor,
        "competitor_position": None,  # not stored on item — could derive from llm rows
        "latest_snapshot_at": latest_at.isoformat() if latest_at else None,
        "scan_question_id": str(question.id),
        "in_flight_refresh_job_id": str(in_flight_refresh_job.id) if in_flight_refresh_job else None,
        "responses": responses,
    }


def _count_sibling_items_at_url(item: ScanContentItem, db: Session) -> int:
    """How many OTHER ContentItems in this scan target the same URL.

    0 = unique target_url (or item has no target_url). >0 triggers the
    "consider a different page" warning chip in the validation UI.
    """
    if not item.target_url:
        return 0
    n = (
        db.query(ScanContentItem)
        .filter(
            ScanContentItem.scan_id == item.scan_id,
            ScanContentItem.target_url == item.target_url,
            ScanContentItem.id != item.id,
        )
        .count()
    )
    return max(0, n)


class ContentItemPatch(BaseModel):
    status: str | None = None
    validation: str | None = None
    content_html: str | None = None
    content_text: str | None = None
    published_url: str | None = None
    target_url: str | None = None
    # Phase D top-3 picker : when the user picks a candidate, we want the
    # page title to match the new URL (the sitemap matcher already has the
    # title in candidate metadata). Otherwise the previous URL's title
    # lingers under the new URL and looks broken.
    target_page_title: str | None = None
    # Per-item override of the workspace-level primary_brand_ids order. First
    # entry = LEAD for this item only. Set by the validation-page Brand
    # promotion picker so the user can say "for THIS opportunity, promote
    # Aderma not Avène" without changing the workspace default that applies
    # to every other item. Validated against the client's primary_brand_ids
    # set so users can't inject brands the system isn't tracking.
    promoted_brand_ids: list[str] | None = None


VALID_STATUSES = {
    "identified", "generating", "draft", "in_review",
    "approved", "published", "rejected",
}
VALID_VALIDATIONS = {"approved", "needs_revision", "rejected"}


@router.patch("/content-items/{item_id}")
async def update_content_item(item_id: str, patch: ContentItemPatch,
                              user=Depends(get_current_user),
                              db: Session = Depends(get_db)):
    """Update workflow / validation / content. Editor role auto-required (PATCH method)."""
    item = (
        db.query(ScanContentItem)
        .options(joinedload(ScanContentItem.scan))
        .filter(ScanContentItem.id == item_id)
        .first()
    )
    if not item:
        raise HTTPException(404, "Content item not found")

    # RBAC: editor role auto-bumped on PATCH via request_method contextvar
    _check_scan_access(str(item.scan_id), user, db)

    changes: dict[str, dict] = {}

    if patch.status is not None:
        if patch.status not in VALID_STATUSES:
            raise HTTPException(400, f"Invalid status: {patch.status}")
        if patch.status != item.status:
            changes["status"] = {"old": item.status, "new": patch.status}
            item.status = patch.status
            # Auto-set lifecycle dates
            if patch.status == "published" and not item.published_at:
                item.published_at = datetime.utcnow()
            if patch.status == "in_review" and not item.ordered_at:
                item.ordered_at = datetime.utcnow()

    if patch.validation is not None:
        if patch.validation == "":
            # Empty string sentinel = clear validation. Used by the
            # "Regenerate from rejected" path so the rejection audit row
            # is reset (validation/validated_by/at all to NULL) but the
            # event remains in the audit_log timeline as a trail.
            if item.validation is not None:
                changes["validation"] = {"old": item.validation, "new": None}
                item.validation = None
                item.validated_by = None
                item.validated_at = None
        elif patch.validation not in VALID_VALIDATIONS:
            raise HTTPException(400, f"Invalid validation: {patch.validation}")
        elif patch.validation != item.validation:
            changes["validation"] = {"old": item.validation, "new": patch.validation}
            item.validation = patch.validation
            item.validated_by = user.email
            item.validated_at = datetime.utcnow()
            # Sync status to validation when approving/rejecting
            if patch.validation == "approved" and item.status in ("draft", "in_review"):
                item.status = "approved"
            elif patch.validation == "rejected" and item.status != "rejected":
                item.status = "rejected"

    if patch.content_html is not None and patch.content_html != item.content_html:
        changes["content_html"] = {"len_old": len(item.content_html or ""),
                                   "len_new": len(patch.content_html)}
        item.content_html = patch.content_html

    if patch.content_text is not None and patch.content_text != item.content_text:
        changes["content_text"] = {"len_old": len(item.content_text or ""),
                                   "len_new": len(patch.content_text)}
        item.content_text = patch.content_text

    if patch.published_url is not None and patch.published_url != item.published_url:
        changes["published_url"] = {"old": item.published_url, "new": patch.published_url}
        item.published_url = patch.published_url
        # Promote to published status if URL set and not already
        if patch.published_url and item.status != "published":
            changes["status"] = changes.get("status") or {"old": item.status, "new": "published"}
            item.status = "published"
            if not item.published_at:
                item.published_at = datetime.utcnow()

    if patch.promoted_brand_ids is not None:
        # Validate : every ID must be in the client's primary_brand_ids set
        # (no smuggling random brands per-item — those go through workspace
        # settings). Also dedupe + preserve order.
        from models import Client
        client = db.query(Client).filter(Client.id == item.scan.client_id).first()
        workspace_primary = {str(bid) for bid in (client.primary_brand_ids or []) if client}
        seen = set()
        deduped: list[str] = []
        invalid: list[str] = []
        for bid in (patch.promoted_brand_ids or []):
            bid_str = str(bid).strip()
            if not bid_str:
                continue
            if bid_str not in workspace_primary:
                invalid.append(bid_str)
                continue
            if bid_str in seen:
                continue
            seen.add(bid_str)
            deduped.append(bid_str)
        if invalid:
            raise HTTPException(400, {
                "error": "invalid_brand_ids",
                "message": "Some brand IDs aren't in this workspace's primary brands. "
                           "Add them in /app/settings/brands first.",
                "invalid": invalid[:5],
            })
        # Compare to current (item.promoted_brand_ids is list[UUID] from DB).
        current = [str(bid) for bid in (item.promoted_brand_ids or [])]
        if deduped != current:
            old_lead = current[0] if current else None
            new_lead = deduped[0] if deduped else None
            changes["promoted_brand_ids"] = {"old": current, "new": deduped}
            # Persist as list of UUID strings — SQLAlchemy ARRAY(UUID) accepts
            # either UUID objects or strings, the casting happens on flush.
            from uuid import UUID
            try:
                item.promoted_brand_ids = [UUID(b) for b in deduped]
            except ValueError as e:
                raise HTTPException(400, f"Malformed UUID in promoted_brand_ids: {e}")
            # Lead change → the rejected URLs were collected on the OLD lead's
            # domain, so they're irrelevant for matching on the new lead's site.
            # Clear them so the next "Find a different page" gets a clean budget.
            if old_lead != new_lead and (item.rejected_target_urls or []):
                changes["rejected_target_urls"] = {
                    "old_count": len(item.rejected_target_urls or []),
                    "new_count": 0,
                    "reason": "lead_brand_changed",
                }
                item.rejected_target_urls = []
            # If the materialize handler had auto-suggested a LEAD (chip "Auto"
            # in the UI), flip the source to 'user' so the chip disappears.
            # We don't drop the auto record — it stays for audit. Only fire
            # when the lead actually changed, to avoid spurious flips on
            # re-saves with identical values.
            if old_lead != new_lead:
                meta = dict(item.content_metadata or {})
                suggestion = dict(meta.get("lead_suggestion") or {})
                if suggestion and suggestion.get("source") == "auto":
                    suggestion["source"] = "user"
                    meta["lead_suggestion"] = suggestion
                    item.content_metadata = meta
                    flag_modified(item, "content_metadata")

    if patch.target_url is not None:
        # Normalize: empty string from the input becomes NULL (clearing the field)
        normalized = patch.target_url.strip() or None
        if normalized != item.target_url:
            changes["target_url"] = {"old": item.target_url, "new": normalized}
            item.target_url = normalized
            # A user-driven edit always flips the source to 'user_input' — this
            # is the audit signal that distinguishes manual picks from future
            # auto-suggestions (Phase D Pilier 3).
            if normalized:
                changes["target_url_source"] = {"old": item.target_url_source, "new": "user_input"}
                item.target_url_source = "user_input"
            else:
                changes["target_url_source"] = {"old": item.target_url_source, "new": "pending_user"}
                item.target_url_source = "pending_user"

    if patch.target_page_title is not None:
        normalized_title = patch.target_page_title.strip() or None
        if normalized_title != item.target_page_title:
            changes["target_page_title"] = {"old": item.target_page_title, "new": normalized_title}
            item.target_page_title = normalized_title

    if changes:
        audit_log(
            db, user_id=str(user.id),
            action="content_item.update",
            target_type="content_item", target_id=item_id,
            details=changes,
        )
    db.commit()
    db.refresh(item)

    client_id_str = str(item.scan.client_id)
    brand_names = _resolve_brand_names(
        client_id_str,
        {str(b) for b in (item.promoted_brand_ids or [])},
        db,
    )
    primary_domains = _resolve_primary_brand_domains(client_id_str, db)
    share_count = _count_sibling_items_at_url(item, db)
    return _serialize_item(item, brand_names, primary_domains, share_count)


@router.post("/content-items/{item_id}/generate")
async def generate_content(item_id: str, user=Depends(get_current_user),
                           db: Session = Depends(get_db)):
    """Enqueue a content generation job for this item.

    Dispatches to the right worker handler by content_type (faq → generate_faq,
    article → generate_article in Phase C). Dedupes : if a generate job is
    already in flight for this item, returns the existing job_id rather than
    enqueueing a duplicate.

    Status transitions: identified → generating (set by handler) → draft (on
    success) or back to identified (on failure, allowing retry from Kanban).

    Credit policy : 1 `content_credit` debited at enqueue time. If the user
    doesn't have enough credits, returns 402 with a friendly message. Refund
    on permanent failure is a known follow-up (the centralized
    `_refund_scan_credits` path in worker/main.py would over-refund the
    parent scan's scan_credits, so a content-scoped refund helper needs
    writing — out of scope for this debit-only commit).
    """
    item = (
        db.query(ScanContentItem)
        .options(joinedload(ScanContentItem.scan))
        .filter(ScanContentItem.id == item_id)
        .first()
    )
    if not item:
        raise HTTPException(404, "Content item not found")

    _check_scan_access(str(item.scan_id), user, db)

    handler = GENERATOR_BY_CONTENT_TYPE.get(item.content_type)
    if not handler:
        raise HTTPException(400, {
            "error": "unsupported_content_type",
            "message": f"No generator wired for content_type='{item.content_type}' yet. "
                       f"Supported: {list(GENERATOR_BY_CONTENT_TYPE.keys())}",
        })

    if item.status not in ("identified", "draft"):
        raise HTTPException(409, {
            "error": "invalid_status",
            "message": f"Item is in status '{item.status}' — can only (re)generate from "
                       f"'identified' or 'draft'. Use the validation page to manage "
                       f"approved/published items.",
        })

    if not item.target_url:
        raise HTTPException(400, {
            "error": "missing_target_url",
            "message": "Pick a URL on your site where this FAQ should live, "
                       "then click Generate. (Open the validation page to set it.)",
        })

    # Dedupe in-flight job (don't double-enqueue if user clicks twice)
    in_flight = (
        db.query(Job)
        .filter(
            Job.scan_id == item.scan_id,
            Job.job_type == handler,
            Job.status.in_(["pending", "running"]),
        )
        .all()
    )
    for j in in_flight:
        if (j.payload or {}).get("item_id") == item_id:
            return {
                "ok": True, "job_id": str(j.id), "status": j.status,
                "message": "Generation already in flight",
            }

    # Debit 1 content_credit before enqueuing. Locks the client row to
    # serialize against concurrent debits (re-entrant within this txn).
    # The debit row is tied to scan_id so audit queries can reconstruct
    # which scan triggered which content gen.
    from routers.stripe import add_credits
    try:
        add_credits(
            client_id=str(item.scan.client_id),
            credit_type="content",
            amount=-1,
            description=f"FAQ generation: {item_id}",
            db=db,
            scan_id=str(item.scan_id),
        )
    except ValueError as e:
        raise HTTPException(402, {
            "error": "insufficient_credits",
            "message": "You don't have enough content credits to generate this FAQ. "
                       "Buy more on the Settings page, then come back.",
            "detail": str(e),
        })

    job = Job(
        scan_id=item.scan_id,
        job_type=handler,
        status="pending",
        payload={"item_id": item_id},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    audit_log(
        db, user_id=str(user.id),
        action="content_item.generate",
        target_type="content_item", target_id=item_id,
        details={"handler": handler, "job_id": str(job.id)},
    )

    return {"ok": True, "job_id": str(job.id), "status": "pending"}


# Hard cap on user-triggered FAQPageMatcher reruns per item.
# Each attempt = one OpenAI web_search call (~$0.02). Spam-clicking would
# blow the budget quickly with no real signal past ~10 tries — at that point
# the brand site genuinely doesn't have a fitting page and the user should
# fall back to manual picking. UI mirrors this constant for symmetric UX.
# When this lives in more than one place, hoist into config.py.
REMATCH_MAX_ATTEMPTS_PER_ITEM = 10


@router.post("/content-items/{item_id}/rematch-target-url")
async def rematch_target_url(item_id: str, user=Depends(get_current_user),
                             db: Session = Depends(get_db)):
    """Enqueue a rematch job: re-run FAQPageMatcher excluding URLs already
    rejected by the user (the current target_url is implicitly rejected).

    Free operation (no content_credit debit) — iteration cost on a single
    web_search is small and we want zero friction so users converge on a
    page they're happy with. Capped at REMATCH_MAX_ATTEMPTS_PER_ITEM to
    bound LLM spend if a user spam-clicks.

    Returns {ok, job_id}. Frontend polls GET /content-items/{id} until
    target_url or target_url_source changes, then refreshes the UI.
    """
    item = (
        db.query(ScanContentItem)
        .options(joinedload(ScanContentItem.scan))
        .filter(ScanContentItem.id == item_id)
        .first()
    )
    if not item:
        raise HTTPException(404, "Content item not found")

    _check_scan_access(str(item.scan_id), user, db)

    if item.status not in ("identified", "draft"):
        raise HTTPException(409, {
            "error": "invalid_status",
            "message": f"Item is in status '{item.status}' — rematch only available on "
                       f"'identified' or 'draft'. Once content is generated, edit the "
                       f"URL manually instead.",
        })

    # Hard cap : the matcher genuinely runs out of useful candidates after
    # ~10 attempts on a normal brand site. Past that, route to manual.
    # 429 chosen over 402 because this isn't a credit issue, it's a per-item
    # rate cap. The UI surfaces the same condition client-side.
    rejected_count = len(item.rejected_target_urls or [])
    if rejected_count >= REMATCH_MAX_ATTEMPTS_PER_ITEM:
        raise HTTPException(429, {
            "error": "rematch_cap_reached",
            "message": f"You've tried {rejected_count} pages on this item without finding "
                       f"a better match. The matcher has explored what your brand site offers — "
                       f"please pick a URL manually below.",
            "attempts_used": rejected_count,
            "cap": REMATCH_MAX_ATTEMPTS_PER_ITEM,
        })

    # Dedupe in-flight rematch for the same item (double-click guard).
    in_flight = (
        db.query(Job)
        .filter(
            Job.scan_id == item.scan_id,
            Job.job_type == "rematch_target_url",
            Job.status.in_(["pending", "running"]),
        )
        .all()
    )
    for j in in_flight:
        if (j.payload or {}).get("item_id") == item_id:
            return {
                "ok": True, "job_id": str(j.id), "status": j.status,
                "message": "Rematch already in flight",
            }

    job = Job(
        scan_id=item.scan_id,
        job_type="rematch_target_url",
        status="pending",
        payload={"item_id": item_id},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    audit_log(
        db, user_id=str(user.id),
        action="content_item.rematch_target_url",
        target_type="content_item", target_id=item_id,
        details={
            "job_id": str(job.id),
            "previous_target_url": item.target_url,
            "rejected_count": len(item.rejected_target_urls or []),
        },
    )

    return {"ok": True, "job_id": str(job.id), "status": "pending"}


# Phase E Pilier 5 — manual refresh of the AI snapshot.
# 5 refreshes/24h/question. Each refresh = 2 LLM calls (~$0.04). Hard cap on
# the day's count keeps spend bounded even if a user spam-clicks.
REFRESH_AI_SNAPSHOT_MAX_PER_DAY = 5


@router.post("/content-items/{item_id}/refresh-ai-snapshot")
async def refresh_ai_snapshot(item_id: str, user=Depends(get_current_user),
                               db: Session = Depends(get_db)):
    """Re-run the LLM tests for THIS item's question.

    Calls every configured provider (ChatGPT + Gemini today) on the single
    question, stores fresh ScanLLMResult rows. The detail endpoint then
    surfaces the latest row per provider in `competitor_snapshot`. ~$0.04
    per refresh.

    Capped at REFRESH_AI_SNAPSHOT_MAX_PER_DAY refreshes/24h/question
    (counted via ScanLLMResult timestamps). In-flight protection blocks
    double-enqueue.
    """
    from datetime import datetime, timedelta

    item = (
        db.query(ScanContentItem)
        .options(joinedload(ScanContentItem.scan))
        .filter(ScanContentItem.id == item_id)
        .first()
    )
    if not item:
        raise HTTPException(404, "Content item not found")

    _check_scan_access(str(item.scan_id), user, db)

    q_text = (item.target_question or "").strip()
    if not q_text:
        raise HTTPException(422, {
            "error": "no_question",
            "message": "This item has no target_question — nothing to refresh.",
        })

    question = (
        db.query(ScanQuestion)
        .filter(
            ScanQuestion.scan_id == item.scan_id,
            func.lower(ScanQuestion.question) == q_text.lower(),
        )
        .first()
    )
    if not question:
        raise HTTPException(404, {
            "error": "scan_question_not_found",
            "message": "Couldn't link this item back to a ScanQuestion. The "
                       "snapshot can't be refreshed — try editing the question "
                       "text to match the original scan.",
        })

    cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_rows = (
        db.query(ScanLLMResult)
        .filter(
            ScanLLMResult.question_id == question.id,
            ScanLLMResult.created_at > cutoff,
        )
        .count()
    )
    approx_refreshes_done = max(0, (recent_rows // 2) - 1)
    if approx_refreshes_done >= REFRESH_AI_SNAPSHOT_MAX_PER_DAY:
        raise HTTPException(429, {
            "error": "refresh_cap_reached",
            "message": f"You've refreshed this AI snapshot {approx_refreshes_done} time(s) "
                       f"in the last 24 hours (cap: {REFRESH_AI_SNAPSHOT_MAX_PER_DAY}). "
                       f"Wait until tomorrow — or re-run the full scan if you need fresh "
                       f"data on many questions at once.",
            "refreshes_done": approx_refreshes_done,
            "cap": REFRESH_AI_SNAPSHOT_MAX_PER_DAY,
        })

    in_flight = (
        db.query(Job)
        .filter(
            Job.scan_id == item.scan_id,
            Job.job_type == "refresh_ai_snapshot",
            Job.status.in_(["pending", "running"]),
        )
        .all()
    )
    for j in in_flight:
        if (j.payload or {}).get("item_id") == item_id:
            return {
                "ok": True, "job_id": str(j.id), "status": j.status,
                "message": "Refresh already in flight",
            }

    job = Job(
        scan_id=item.scan_id,
        job_type="refresh_ai_snapshot",
        status="pending",
        payload={"item_id": item_id},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    audit_log(
        db, user_id=str(user.id),
        action="content_item.refresh_ai_snapshot",
        target_type="content_item", target_id=item_id,
        details={
            "job_id": str(job.id),
            "scan_question_id": str(question.id),
            "refreshes_done_last_24h": approx_refreshes_done,
        },
    )

    return {"ok": True, "job_id": str(job.id), "status": "pending", "refreshes_done_last_24h": approx_refreshes_done, "cap": REFRESH_AI_SNAPSHOT_MAX_PER_DAY}
