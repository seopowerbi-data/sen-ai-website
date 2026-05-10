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
from sqlalchemy import desc
from sqlalchemy.orm import Session, joinedload

from models import (
    ClientBrand, Job, Scan, ScanContentItem, UserClient, get_db,
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


def _serialize_item(item: ScanContentItem, brand_names: dict[str, str] | None = None) -> dict:
    """Convert a ScanContentItem ORM row into the dict shape the UI consumes."""
    promoted_names: list[str] = []
    if item.promoted_brand_ids and brand_names:
        for bid in item.promoted_brand_ids:
            name = brand_names.get(str(bid))
            if name:
                promoted_names.append(name)

    scan = item.scan
    return {
        "id": str(item.id),
        "scan_id": str(item.scan_id),
        "scan_domain": scan.domain if scan else None,
        "scan_name": scan.name if scan else None,
        "content_type": item.content_type,
        "topic_name": item.topic_name,
        "persona_name": item.persona_name,
        "target_url": item.target_url,
        "target_page_title": item.target_page_title,
        "target_question": item.target_question,
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

    serialized = [_serialize_item(it, brand_names) for it in items]

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

    brand_names = _resolve_brand_names(
        str(item.scan.client_id),
        {str(b) for b in (item.promoted_brand_ids or [])},
        db,
    )
    return _serialize_item(item, brand_names)


class ContentItemPatch(BaseModel):
    status: str | None = None
    validation: str | None = None
    content_html: str | None = None
    content_text: str | None = None
    published_url: str | None = None


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
        if patch.validation not in VALID_VALIDATIONS:
            raise HTTPException(400, f"Invalid validation: {patch.validation}")
        if patch.validation != item.validation:
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

    if changes:
        audit_log(
            db, user_id=str(user.id),
            action="content_item.update",
            target_type="content_item", target_id=item_id,
            details=changes,
        )
    db.commit()
    db.refresh(item)

    brand_names = _resolve_brand_names(
        str(item.scan.client_id),
        {str(b) for b in (item.promoted_brand_ids or [])},
        db,
    )
    return _serialize_item(item, brand_names)


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

    NOTE: credit debit not yet wired here. Phase B pricing = 1 content_credit
    per FAQ — to be added in a follow-up commit alongside the Stripe content
    credits flow.
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
            "message": "FAQ generation requires a target_url to scrape. This item has none.",
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
