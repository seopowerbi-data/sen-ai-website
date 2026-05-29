import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sqlalchemy import func
from sqlalchemy.orm import joinedload
from models import (
    Scan, ScanKeyword, ScanTopic, ScanPersona, ScanQuestion, ScanLLMResult,
    ScanQuestionJudgment,
    ScanBrandClassification, ScanBrandTopic, ScanOpportunity, ClientBrand,
    ScanPageAudit, ScanSchemaAudit, ScanCompetitorPage, ScanRedditThread, ScanPROutreach, ScanInternalLink, ScanYouTubeCreator, ScanCrisisSignal, ScanSentimentJudgement,
    Client,
    Job, UserClient, get_db,
)

logger = logging.getLogger(__name__)
from services.auth_service import get_current_user
from services.audit import audit_log
from services.rate_limit import limiter
from services.request_context import current_request_method
from services.sanitize import strip_tags
import unicodedata

router = APIRouter()


# --- Helpers: brand classification lookup ---

def _strip_accents(s: str) -> str:
    """Remove accents for fuzzy brand name matching (L'Oréal → L'Oreal)."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def _build_brand_classification_map(scan_id, db) -> dict[str, str]:
    """Build brand_name_lower → classification lookup from ScanBrandClassification + ClientBrand.

    Indexes by: name, canonical_name, aliases, + accent-stripped variants.
    Returns: {"la roche-posay": "competitor", "avène": "my_brand", ...}
    """
    rows = (
        db.query(ScanBrandClassification, ClientBrand)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(ScanBrandClassification.scan_id == scan_id)
        .all()
    )
    lookup = {}
    for sbc, brand in rows:
        cls = sbc.classification
        for variant in [brand.name, brand.canonical_name] + (brand.aliases or []):
            if variant:
                lookup[variant.strip().lower()] = cls
                stripped = _strip_accents(variant.strip().lower())
                if stripped != variant.strip().lower():
                    lookup[stripped] = cls
    return lookup


def _classify_brand_mention(name: str, classification_map: dict, focus_names_lower: set) -> str:
    """Classify a brand mention name using the map. Returns classification string."""
    if not name:
        return "discovered"
    low = name.lower()
    if low in focus_names_lower:
        return "my_brand"
    cls = classification_map.get(low) or classification_map.get(_strip_accents(low))
    return cls or "discovered"


# --- Schemas ---

class ScanCreate(BaseModel):
    client_id: str
    domain: str                          # Domain to scan (own or competitor's)
    name: str | None = None              # User-facing scan name (defaults to domain)
    target_domains: list[str] = []       # My domains to check in citations (default: [domain])
    brand_names: list[str] = []          # My brand names to detect in LLM responses
    max_position: int = 50               # Top N positions to keep (10, 30, 50)
    max_urls: int = 2000                 # Max keywords to fetch from HaloScan
    config: dict = {}
    # User-declared scan intent. 'own_brand' = measuring my own brand visibility,
    # 'competitor_audit' = auditing a competitor's visibility. None = legacy /
    # wizard didn't ask - downstream falls back to the domain-vs-primary-brands
    # heuristic in is_competitor_scan().
    scan_type: str | None = None


class ScanUpdate(BaseModel):
    """PATCH payload - all fields optional."""
    name: str | None = None
    focus_brand_id: str | None = None
    schedule: str | None = None  # manual | weekly | monthly


class ScanConfigUpdate(BaseModel):
    """PATCH /scans/{id}/config - update scan configuration."""
    providers: list[str] | None = None


class BrandClassify(BaseModel):
    brand_id: str
    classification: str  # my_brand | competitor | ignored | unclassified
    is_focus: bool = False


class BrandReparent(BaseModel):
    """Payload for drag-to-parent - moves brand under a new parent.
    parent_id=None ⇒ promote to root.
    """
    parent_id: str | None = None


class BrandBulkClassify(BaseModel):
    """Bulk classify N brands at once (one DB pass instead of N PATCHes).
    Used by the "Mark all as ignored" button in the amber inbox post-scan.
    """
    brand_ids: list[str]
    classification: str  # my_brand | competitor | ignored | unclassified


class ScanResponse(BaseModel):
    id: str
    client_id: str
    domain: str
    status: str
    progress_pct: int
    progress_message: str | None
    config: dict | None
    summary: dict | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    error_message: str | None

    model_config = {"from_attributes": True}


class TopicCreate(BaseModel):
    name: str
    description: str | None = None


class TopicUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_active: bool | None = None


class PersonaCreate(BaseModel):
    topic_id: str | None = None
    name: str
    data: dict


class PersonaUpdate(BaseModel):
    """PATCH payload - every field optional. Used for rename / toggle / reassign."""
    name: str | None = None
    data: dict | None = None
    topic_id: str | None = None
    is_active: bool | None = None


class QuestionCreate(BaseModel):
    persona_id: str
    question: str
    type_question: str | None = None


class QuestionUpdate(BaseModel):
    """PATCH payload - every field optional."""
    question: str | None = None
    type_question: str | None = None
    is_active: bool | None = None


class MoveUrlRequest(BaseModel):
    url: str


class MoveKeywordRequest(BaseModel):
    keyword: str
    source_topic_id: str


# --- Helpers ---

# H6: role hierarchy. Higher number = more permissions.
# Phase E.C.5.1 - _ROLE_RANK + _DESTRUCTIVE_METHODS dead since
# _check_scan_access was refactored to delegate to services/access.py.
# Removed.


def _check_scan_access(scan_id: str, user, db: Session) -> Scan:
    """Verify the current user has access to a scan, with method-aware RBAC.

    Delegates to `services.access.check_client_access` so the new
    `org_user_clients` table is honored alongside the legacy `user_clients`
    fallback (Phase E.C.2). Without this, a user who gained access to a
    client through the org layer (no legacy `user_clients` row) would get
    a spurious 403 - observed when switching to a freshly-created org with
    a client they own via `org_user_clients` only.
    """
    from services.access import check_client_access

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(404, "Scan not found")
    check_client_access(str(scan.client_id), user, db)
    return scan


def _recalc_topic_keyword_count(topic_id: str, db: Session) -> int:
    """Recalculate keyword_count for a topic using COUNT DISTINCT keyword.
    HaloScan returns 1 row per (keyword, url) pair, so we count distinct keyword texts,
    not rows, to reflect actual concept count shown to the user.
    """
    if not topic_id:
        return 0
    count = db.query(func.count(func.distinct(ScanKeyword.keyword))).filter(
        ScanKeyword.topic_id == topic_id,
    ).scalar() or 0
    topic = db.query(ScanTopic).filter(ScanTopic.id == topic_id).first()
    if topic:
        topic.keyword_count = count
    return count


def _count_topic_urls(topic_id: str, db: Session) -> int:
    """Count distinct URLs currently assigned to a topic (via its keyword rows)."""
    if not topic_id:
        return 0
    return db.query(func.count(func.distinct(ScanKeyword.url))).filter(
        ScanKeyword.topic_id == topic_id,
        ScanKeyword.url != "",
    ).scalar() or 0


def _create_job(db: Session, scan_id: str, job_type: str, payload: dict = {}) -> Job:
    job = Job(scan_id=scan_id, job_type=job_type, payload=payload)
    db.add(job)
    db.commit()
    return job


def _serialize_scan(scan: Scan) -> dict:
    return {
        "id": str(scan.id),
        "client_id": str(scan.client_id),
        "name": scan.name or scan.domain,
        "domain": scan.domain,
        "status": scan.status,
        "focus_brand_id": str(scan.focus_brand_id) if scan.focus_brand_id else None,
        "focus_brand_name": scan.focus_brand.name if scan.focus_brand else None,
        "parent_scan_id": str(scan.parent_scan_id) if scan.parent_scan_id else None,
        "run_index": scan.run_index or 1,
        "schedule": scan.schedule or "manual",
        "next_run_at": scan.next_run_at.isoformat() if scan.next_run_at else None,
        "progress_pct": scan.progress_pct or 0,
        "progress_message": scan.progress_message,
        "config": scan.config,
        "summary": scan.summary,
        "created_at": scan.created_at.isoformat() if scan.created_at else None,
        "started_at": scan.started_at.isoformat() if scan.started_at else None,
        "completed_at": scan.completed_at.isoformat() if scan.completed_at else None,
        "error_message": scan.error_message,
    }


# --- Scan CRUD ---

@router.post("/")
@limiter.limit("20/minute")
async def create_scan(request: Request, req: ScanCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    from services.access import check_client_access
    # POST is destructive → check_client_access auto-enforces 'editor' minimum.
    check_client_access(req.client_id, user, db)

    if req.max_position not in (10, 30, 50):
        raise HTTPException(400, "max_position must be 10, 30 or 50")
    if req.max_urls < 100 or req.max_urls > 5000:
        raise HTTPException(400, "max_urls must be between 100 and 5000")
    if req.scan_type is not None and req.scan_type not in ("own_brand", "competitor_audit"):
        raise HTTPException(
            400,
            "scan_type must be 'own_brand', 'competitor_audit', or omitted",
        )

    clean_domain = req.domain.strip().lower().removeprefix("https://").removeprefix("http://").rstrip("/")
    config = {
        **req.config,
        "max_position": req.max_position,
        "max_urls": req.max_urls,
        "target_domains": req.target_domains or [clean_domain],
        "brand_names": req.brand_names,
    }
    scan = Scan(
        client_id=req.client_id,
        name=strip_tags(req.name) or clean_domain,
        domain=clean_domain,
        config=config,
        scan_type=req.scan_type,
        created_by=user.id,
        run_index=1,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return _serialize_scan(scan)


@router.get("/")
async def list_scans(
    client_id: str = Query(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from services.access import check_client_access
    check_client_access(client_id, user, db)

    scans = db.query(Scan).options(joinedload(Scan.focus_brand)).filter(
        Scan.client_id == client_id,
    ).order_by(Scan.created_at.desc()).all()
    return [_serialize_scan(s) for s in scans]


@router.get("/{scan_id}")
async def get_scan(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    scan = _check_scan_access(scan_id, user, db)
    return _serialize_scan(scan)


@router.delete("/{scan_id}")
async def delete_scan(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    scan = _check_scan_access(scan_id, user, db)
    # Use raw SQL DELETE to bypass SQLAlchemy ORM cascade ordering issues.
    # The DB-level ON DELETE CASCADE/SET NULL handles all FK dependencies correctly.
    from sqlalchemy import text
    db.execute(text("DELETE FROM scans WHERE id = :id"), {"id": scan_id})
    db.commit()
    return {"deleted": True}


@router.patch("/{scan_id}")
async def update_scan(scan_id: str, req: ScanUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Update mutable scan metadata: name, focus_brand_id, schedule."""
    scan = _check_scan_access(scan_id, user, db)

    if req.name is not None:
        name = strip_tags(req.name)
        if not name:
            raise HTTPException(400, "name cannot be empty")
        scan.name = name

    if req.focus_brand_id is not None:
        # Verify brand belongs to this client AND is classified for this scan
        brand = db.query(ClientBrand).filter(
            ClientBrand.id == req.focus_brand_id,
            ClientBrand.client_id == scan.client_id,
        ).first()
        if not brand:
            raise HTTPException(404, "Brand not found for this client")
        if brand.parent_id:
            raise HTTPException(400, "Focus brand must be a root brand (not a product line). Pick the parent brand instead.")
        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.brand_id == req.focus_brand_id,
        ).first()
        if not sbc:
            raise HTTPException(400, "Brand must be classified for this scan before it can be the focus")
        if sbc.classification != "my_brand":
            raise HTTPException(400, "Focus brand must be classified as 'my_brand'")
        # Clear existing focus in same transaction, then set new one (avoid unique index violation)
        db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.is_focus == True,
        ).update({ScanBrandClassification.is_focus: False})
        sbc.is_focus = True
        scan.focus_brand_id = req.focus_brand_id

    if req.schedule is not None:
        if req.schedule not in ("manual", "weekly", "monthly"):
            raise HTTPException(400, "schedule must be manual, weekly or monthly")
        scan.schedule = req.schedule

    scan.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return _serialize_scan(scan)


@router.patch("/{scan_id}/config")
async def update_scan_config(scan_id: str, req: ScanConfigUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Update scan config (providers, etc.). JSONB merge - only provided keys are updated."""
    scan = _check_scan_access(scan_id, user, db)
    from sqlalchemy.orm.attributes import flag_modified
    config = dict(scan.config or {})
    if req.providers is not None:
        valid = {"openai", "gemini"}
        providers = [p for p in req.providers if p in valid]
        if not providers:
            raise HTTPException(400, "At least one valid provider required (openai, gemini)")
        config["providers"] = providers
    scan.config = config
    flag_modified(scan, "config")
    scan.updated_at = datetime.utcnow()
    db.commit()
    return {"config": scan.config}


# --- Domain Brief ---

# Hard cap on per-scan brief regenerations. Each call fires OpenAI web_search
# (~$0.02-0.05). The first generation runs in the wizard auto-flow; legitimate
# users almost never need >2 regens. 5 leaves room for edge cases (LLM returned
# garbage on a niche domain) without enabling spam. See
# feedback_cap_user_triggered_llm_ops.
MAX_DOMAIN_BRIEF_GENERATIONS = 5


@router.post("/{scan_id}/generate-brief")
@limiter.limit("5/minute")
async def generate_brief(request: Request, scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Enqueue domain brief generation via OpenAI web search. Idempotent."""
    scan = _check_scan_access(scan_id, user, db)

    # Hard cap : 429 when this scan has already burned the budget. Counter is
    # incremented in the worker handler on success, so a failed run doesn't
    # cost an attempt. Mirror in UI via the GET /brief response.
    brief_state = (scan.config or {}).get("domain_brief") or {}
    used = int(brief_state.get("generations_count") or 0)
    if used >= MAX_DOMAIN_BRIEF_GENERATIONS:
        raise HTTPException(429, {
            "error": "brief_regen_cap_reached",
            "message": f"This scan's brief has been generated {used} times "
                       f"(max {MAX_DOMAIN_BRIEF_GENERATIONS}). Edit the brief manually "
                       f"on Gate 1 - further regenerations are blocked.",
            "generations_used": used,
            "cap": MAX_DOMAIN_BRIEF_GENERATIONS,
        })

    existing = db.query(Job).filter(
        Job.scan_id == scan_id,
        Job.job_type == "generate_domain_brief",
        Job.status.in_(["pending", "running"]),
    ).first()
    if existing:
        return {"status": "already_running", "job_id": str(existing.id)}
    _create_job(db, scan_id, "generate_domain_brief")
    db.commit()
    return {
        "status": "job_created",
        "generations_used": used,
        "cap": MAX_DOMAIN_BRIEF_GENERATIONS,
    }


@router.get("/{scan_id}/brief")
async def get_brief(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return domain brief + generation status + regen budget."""
    scan = _check_scan_access(scan_id, user, db)
    brief = (scan.config or {}).get("domain_brief")
    job = db.query(Job).filter(
        Job.scan_id == scan_id,
        Job.job_type == "generate_domain_brief",
    ).order_by(Job.created_at.desc()).first()
    used = int((brief or {}).get("generations_count") or 0)
    return {
        "domain_brief": brief,
        "generation_status": job.status if job else ("completed" if brief else None),
        "generations_used": used,
        "generations_cap": MAX_DOMAIN_BRIEF_GENERATIONS,
        "can_regenerate": used < MAX_DOMAIN_BRIEF_GENERATIONS,
    }


class DomainBriefUpdate(BaseModel):
    domain_brief: dict


@router.put("/{scan_id}/brief")
async def update_brief(scan_id: str, req: DomainBriefUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Save user-edited domain brief."""
    scan = _check_scan_access(scan_id, user, db)
    from sqlalchemy.orm.attributes import flag_modified
    config = dict(scan.config or {})
    brief = dict(req.domain_brief)
    brief["edited_by_user"] = True
    config["domain_brief"] = brief
    scan.config = config
    flag_modified(scan, "config")
    scan.updated_at = datetime.utcnow()
    db.commit()
    return {"domain_brief": brief}


# --- Per-scan brand classifications (Gate 2) ---

@router.get("/{scan_id}/brands")
async def get_scan_brands(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return per-scan brand classifications grouped by bucket.

    The `my_brand` bucket is hierarchical: parent brands carry a `.children` array
    containing their descendants that are ALSO classified as my_brand in this scan.
    Other buckets (competitor/ignored/unclassified) stay flat - hierarchy only
    matters for the focus-brand semantics (a focus brand's visibility includes
    its product lines).

    Response shape:
    {
      "focus_brand_id": "...",
      "buckets": {
        "my_brand": [
          {brand_id, name, ..., is_focus, children: [{brand_id, name, ..., is_focus}]},
          ...
        ],
        "competitor":   [{brand_id, name, ...}, ...],
        "ignored":      [...],
        "unclassified": [...]
      }
    }

    Notes:
    - Only root brands (no parent_id, or parent not in my_brand bucket) can be focus.
    - Children inside .children are draggable individually to reclassify them.
    - Orphaned children (parent not in this scan's my_brand bucket) are promoted
      to top-level items so they don't disappear from the UI.
    """
    scan = _check_scan_access(scan_id, user, db)

    rows = (
        db.query(ScanBrandClassification, ClientBrand)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(ScanBrandClassification.scan_id == scan_id)
        .order_by(ClientBrand.name.asc())
        .all()
    )

    # Build brand_id → topic names mapping from junction table
    brand_topics_map = {}
    bt_rows = (
        db.query(ScanBrandTopic, ScanTopic)
        .join(ScanTopic, ScanTopic.id == ScanBrandTopic.topic_id)
        .filter(ScanBrandTopic.scan_id == scan_id)
        .all()
    )
    for bt, topic in bt_rows:
        bid = str(bt.brand_id)
        brand_topics_map.setdefault(bid, []).append({"id": str(topic.id), "name": topic.name})

    def _serialize_brand(sbc, brand):
        bid = str(brand.id)
        return {
            "brand_id": bid,
            "name": brand.name,
            "canonical_name": brand.canonical_name,
            "domain": brand.domain,
            "aliases": brand.aliases or [],
            "parent_id": str(brand.parent_id) if brand.parent_id else None,
            "is_focus": bool(sbc.is_focus),
            "classified_by": sbc.classified_by,
            "source": sbc.source,
            "topics": brand_topics_map.get(bid, []),
        }

    # Build flat buckets first (parent_id exposed on every item)
    flat_buckets = {"my_brand": [], "competitor": [], "ignored": [], "unclassified": []}
    for sbc, brand in rows:
        bucket = sbc.classification if sbc.classification in flat_buckets else "unclassified"
        flat_buckets[bucket].append(_serialize_brand(sbc, brand))

    def _nest(items: list) -> list:
        """Group children under their parent root. Only nest if BOTH parent and child
        are in the SAME bucket - otherwise children are promoted to top-level (orphans).
        """
        ids_in_bucket = {item["brand_id"] for item in items}
        children_by_parent: dict[str, list] = {}
        for item in items:
            pid = item.get("parent_id")
            if pid and pid in ids_in_bucket:
                children_by_parent.setdefault(pid, []).append(item)

        roots = []
        for item in items:
            pid = item.get("parent_id")
            if pid and pid in ids_in_bucket:
                continue  # this is a child, nested under its parent
            item["children"] = children_by_parent.get(item["brand_id"], [])
            roots.append(item)
        return roots

    return {
        "scan_id": scan_id,
        "scan_name": scan.name or scan.domain,
        "scan_domain": scan.domain,
        "focus_brand_id": str(scan.focus_brand_id) if scan.focus_brand_id else None,
        "buckets": {
            "my_brand":     _nest(flat_buckets["my_brand"]),
            "competitor":   _nest(flat_buckets["competitor"]),
            "ignored":      _nest(flat_buckets["ignored"]),
            "unclassified": _nest(flat_buckets["unclassified"]),
        },
    }


@router.post("/{scan_id}/brands/classify")
async def classify_scan_brand(scan_id: str, req: BrandClassify, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Upsert a per-scan brand classification. Used by Gate 2 drag & drop.

    - If the brand row doesn't exist yet in scan_brand_classifications, it's created.
    - If is_focus=True, clears any existing focus for this scan in the same transaction
      (avoiding the idx_sbc_one_focus_per_scan unique-index violation).
    - Setting classification != 'my_brand' while is_focus is True is rejected.
    """
    scan = _check_scan_access(scan_id, user, db)

    if req.classification not in ("my_brand", "competitor", "ignored", "unclassified"):
        raise HTTPException(400, "classification must be my_brand | competitor | ignored | unclassified")
    if req.is_focus and req.classification != "my_brand":
        raise HTTPException(400, "Only my_brand can be the focus brand")

    brand = db.query(ClientBrand).filter(
        ClientBrand.id == req.brand_id,
        ClientBrand.client_id == scan.client_id,
    ).first()
    if not brand:
        raise HTTPException(404, "Brand not found for this client")

    if req.is_focus and brand.parent_id:
        raise HTTPException(400, "Focus brand must be a root brand (not a product line). Pick the parent brand instead.")

    sbc = db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == scan_id,
        ScanBrandClassification.brand_id == req.brand_id,
    ).first()

    if sbc is None:
        sbc = ScanBrandClassification(
            scan_id=scan_id,
            brand_id=req.brand_id,
            classification=req.classification,
            is_focus=False,
            classified_by="user",
            source=brand.detection_source,
        )
        db.add(sbc)
        db.flush()  # get sbc.id for the focus handling below

    sbc.classification = req.classification
    sbc.classified_by = "user"
    sbc.updated_at = datetime.utcnow()

    # If demoting from my_brand AND it was the focus → clear focus on the scan too
    if sbc.is_focus and req.classification != "my_brand":
        sbc.is_focus = False
        scan.focus_brand_id = None

    if req.is_focus:
        # Clear any other focus in same txn
        db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.is_focus == True,
            ScanBrandClassification.id != sbc.id,
        ).update({ScanBrandClassification.is_focus: False})
        sbc.is_focus = True
        scan.focus_brand_id = brand.id

    scan.updated_at = datetime.utcnow()
    db.commit()
    return {
        "brand_id": str(brand.id),
        "classification": sbc.classification,
        "is_focus": bool(sbc.is_focus),
        "focus_brand_id": str(scan.focus_brand_id) if scan.focus_brand_id else None,
    }


@router.get("/{scan_id}/pipeline")
async def get_pipeline(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the scan's job timeline with per-step status + ETA.

    Used by the scanning page to render a transparent progress view (Nielsen's
    "visibility of system status" + Doherty Threshold). Replaces the lone
    progress_message line that left users guessing what's happening.
    """
    scan = _check_scan_access(scan_id, user, db)
    jobs = db.query(Job).filter(Job.scan_id == scan_id).order_by(Job.created_at.asc()).all()

    STEP_META = {
        "fetch_keywords":             ("Fetching keywords from HaloScan",  30),
        "classify_topics":            ("Identifying topics",               60),
        "assign_keywords":            ("Linking keywords to topics",       30),
        "detect_competitors":         ("Detecting competitors",            45),
        "cleanup_brands":             ("Cleaning up brand catalog",        90),
        "generate_domain_brief":      ("Generating domain brief",          45),
        "generate_personas":          ("Generating personas",              120),
        "generate_persona_questions": ("Generating questions",             60),
        "classify_question_intent":   ("Classifying question intents",     30),
        "run_llm_tests":              ("Testing AI providers",             1200),
        "judge_question_responses":   ("Judging response quality",         60),
        "generate_opportunities":     ("Identifying opportunities",        45),
        "generate_editorial":         ("Drafting editorial summary",       90),
        "materialize_content_items":  ("Preparing content actions",        45),
    }

    # Re-tune the expensive "Testing AI providers" ETA from actual load.
    active_q = (
        db.query(ScanQuestion)
        .join(ScanPersona, ScanPersona.id == ScanQuestion.persona_id)
        .filter(
            ScanQuestion.scan_id == scan_id,
            ScanQuestion.is_active == True,
            ScanPersona.is_active == True,
        )
        .count()
    )
    providers = (scan.config or {}).get("providers", ["openai"]) or ["openai"]
    if active_q > 0:
        STEP_META["run_llm_tests"] = (
            STEP_META["run_llm_tests"][0],
            int(active_q * len(providers) * 3),
        )

    # Per-provider completion (run_llm_tests writes one row per question×provider)
    # so the Scan UI can split "Testing AI providers" into OpenAI / Gemini / …
    prov_done = dict(
        db.query(ScanLLMResult.provider, func.count())
        .filter(ScanLLMResult.scan_id == scan_id)
        .group_by(ScanLLMResult.provider)
        .all()
    )

    now = datetime.utcnow()

    def _serialize_job(j):
        meta = STEP_META.get(j.job_type, (j.job_type.replace("_", " ").title(), 60))
        label, eta = meta
        out = {
            "job_type": j.job_type,
            "label": label,
            "status": j.status,
            "attempts": j.attempts or 0,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        }
        if j.status == "completed" and j.started_at and j.completed_at:
            out["duration_ms"] = int((j.completed_at - j.started_at).total_seconds() * 1000)
        elif j.status == "running" and j.started_at:
            out["elapsed_ms"] = int((now - j.started_at).total_seconds() * 1000)
            out["eta_seconds"] = max(0, eta - int(out["elapsed_ms"] / 1000))
        else:
            out["eta_seconds"] = eta

        # Per-provider split for the long "Testing AI providers" step.
        if j.job_type == "run_llm_tests" and active_q > 0:
            out["providers"] = [
                {"provider": p, "done": min(int(prov_done.get(p, 0)), active_q), "total": active_q}
                for p in providers
            ]

        if isinstance(j.result, dict):
            r = j.result
            if r.get("total_tests") is not None:
                out["details"] = f"{r['total_tests']} tests · {r.get('citation_rate', 0)}% cited · {r.get('brand_mention_rate', 0)}% brand mentions"
            elif r.get("classified") is not None and r.get("batches") is not None:
                out["details"] = f"{r['classified']} questions classified"
            elif r.get("materialized") is not None:
                out["details"] = f"{r['materialized']} content items"
            elif r.get("brands") is not None and isinstance(r["brands"], list):
                out["details"] = f"{len(r['brands'])} brands processed"
            elif r.get("error"):
                out["error"] = str(r["error"])[:200]
        return out

    # Scan-phase filter - the Scan step UI should show ONLY the scan-execution
    # steps (run_llm_tests onward), not the upstream setup pipeline (keywords,
    # topics, personas) which belong to earlier wizard stages. Showing the full
    # 15-step process on the Scan view confuses users ("if setup is done, why
    # is it still listed?"). We split on the LAUNCH moment = the created_at of
    # the most recent run_llm_tests job. Everything at-or-after = scan phase.
    launch_at = None
    for j in jobs:
        if j.job_type == "run_llm_tests" and j.created_at:
            # keep the latest one (re-launch / rescan creates a fresh job)
            if launch_at is None or j.created_at > launch_at:
                launch_at = j.created_at

    if launch_at is not None:
        scan_jobs = [j for j in jobs if j.created_at and j.created_at >= launch_at]
    else:
        # No run_llm_tests yet - scan not launched. Return everything so the
        # earlier wizard stages still get a progress view if they poll.
        scan_jobs = jobs

    serialized = [_serialize_job(j) for j in scan_jobs]
    completed_count = sum(1 for s in serialized if s["status"] == "completed")
    total_completed_ms = sum(s.get("duration_ms", 0) for s in serialized)
    remaining_eta = sum(s.get("eta_seconds", 0) for s in serialized if s["status"] in ("running", "pending"))

    return {
        "scan_status": scan.status,
        "progress_pct": scan.progress_pct or 0,
        "progress_message": scan.progress_message,
        "steps": serialized,
        "total_steps": len(serialized),
        "completed_steps": completed_count,
        "total_elapsed_ms": total_completed_ms,
        "remaining_eta_seconds": remaining_eta,
        "phase": "scan" if launch_at is not None else "setup",
    }


class EmptyBucketReq(BaseModel):
    classification: str  # ignored | competitor | my_brand | unclassified


@router.post("/{scan_id}/brands/empty-bucket")
async def empty_bucket(scan_id: str, req: EmptyBucketReq, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Drop every SBC row for a given classification on this scan.

    Used by the "Empty Ignored" button - typically called with
    classification='ignored' to clear the trash bucket after the user has
    confirmed nothing valuable is in there.

    Removes the per-scan classification rows only. The `client_brands` rows
    remain so other scans / future runs aren't affected. If the same brand
    name is detected again on a rescan it re-appears in the inbox for
    re-triage.

    Refuses to touch the focus brand (safety: even if the user picks the
    "my_brand" bucket, the row marked is_focus=True is preserved).
    """
    scan = _check_scan_access(scan_id, user, db)
    if req.classification not in ("ignored", "competitor", "my_brand", "unclassified"):
        raise HTTPException(400, "classification must be ignored | competitor | my_brand | unclassified")

    deleted = (
        db.query(ScanBrandClassification)
        .filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.classification == req.classification,
            ScanBrandClassification.is_focus == False,
        )
        .delete(synchronize_session=False)
    )
    scan.updated_at = datetime.utcnow()
    db.commit()
    return {"deleted": deleted, "classification": req.classification}


@router.post("/{scan_id}/brands/bulk-classify")
async def bulk_classify_brands(scan_id: str, req: BrandBulkClassify, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Set classification on N brands at once.

    Used by the "Mark all as ignored" inbox shortcut - N≈2000 individual
    PATCH calls would burn 30s+ of round-trips, and cleanup_brands often
    fails on dense scans anyway. This is the manual escape hatch.

    Never touches focus brands or brands already classified as my_brand
    (avoids accidental demotion if the user mis-selects).
    """
    scan = _check_scan_access(scan_id, user, db)

    if req.classification not in ("my_brand", "competitor", "ignored", "unclassified"):
        raise HTTPException(400, "classification must be my_brand | competitor | ignored | unclassified")
    if not req.brand_ids:
        return {"updated": 0, "inserted": 0, "skipped_my_brand": 0}

    # Ensure all brand_ids belong to this scan's client (security boundary)
    valid_brand_ids = {
        str(bid) for (bid,) in db.query(ClientBrand.id).filter(
            ClientBrand.id.in_(req.brand_ids),
            ClientBrand.client_id == scan.client_id,
        ).all()
    }
    if not valid_brand_ids:
        return {"updated": 0, "inserted": 0, "skipped_my_brand": 0}

    # Read existing SBC rows in one query
    existing = {
        str(s.brand_id): s for s in db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.brand_id.in_(valid_brand_ids),
        ).all()
    }

    updated = inserted = skipped_my_brand = 0
    for bid in valid_brand_ids:
        sbc = existing.get(bid)
        if sbc is None:
            db.add(ScanBrandClassification(
                scan_id=scan_id,
                brand_id=bid,
                classification=req.classification,
                is_focus=False,
                classified_by="user_bulk",
                source="user_bulk",
            ))
            inserted += 1
        else:
            # Protect my_brand + focus from accidental bulk overrides.
            if sbc.classification == "my_brand" or sbc.is_focus:
                skipped_my_brand += 1
                continue
            if sbc.classification != req.classification:
                sbc.classification = req.classification
                sbc.classified_by = "user_bulk"
                sbc.source = "user_bulk"
                sbc.updated_at = datetime.utcnow()
                updated += 1

    scan.updated_at = datetime.utcnow()
    db.commit()
    return {
        "updated": updated,
        "inserted": inserted,
        "skipped_my_brand": skipped_my_brand,
    }


@router.delete("/{scan_id}/brands/{brand_id}")
async def remove_brand_from_scan(scan_id: str, brand_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Remove the brand from this scan's view.

    Drops the per-scan classification row only - the canonical `client_brands`
    row is preserved (it may be referenced by other scans or future runs).
    Used by the × button on the Ignored bucket to fully discard a brand from
    the current scan after the user has already moved it through the buckets.

    Idempotent: returns 200 even if no SBC row existed.
    """
    scan = _check_scan_access(scan_id, user, db)

    sbc = db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == scan_id,
        ScanBrandClassification.brand_id == brand_id,
    ).first()
    if sbc is None:
        return {"deleted": False, "message": "no classification existed"}

    # Refuse to drop the focus brand - protects the scan from accidental
    # focus loss when a user gets click-happy in the Ignored column.
    if sbc.is_focus:
        raise HTTPException(
            400,
            "Cannot remove the focus brand. Star a different brand first."
        )

    db.delete(sbc)
    scan.updated_at = datetime.utcnow()
    db.commit()
    return {"deleted": True, "brand_id": brand_id}


@router.patch("/{scan_id}/brands/{brand_id}/parent")
async def reparent_brand(scan_id: str, brand_id: str, req: BrandReparent, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Set or clear the parent of a brand via drag-to-parent.

    The hierarchy lives on `client_brands.parent_id` (client-scoped, not
    scan-scoped), so a successful PATCH affects every scan in the client.
    That's the intended behaviour - brand hierarchy is canonical knowledge
    about the workspace, not per-scan opinion.

    Validations:
    - Both brand and proposed parent belong to scan.client_id.
    - parent_id != brand_id (no self-loop).
    - parent_id must not be a descendant of brand (no circular hierarchy).
    - parent and brand must share the same bucket in THIS scan
      (e.g. competitor cannot nest under my_brand) - keeps the UI sane.
    """
    scan = _check_scan_access(scan_id, user, db)

    brand = db.query(ClientBrand).filter(
        ClientBrand.id == brand_id,
        ClientBrand.client_id == scan.client_id,
    ).first()
    if not brand:
        raise HTTPException(404, "Brand not found for this client")

    parent_id = req.parent_id
    if parent_id is None:
        # Promote to root - always safe (cycle-free by definition).
        brand.parent_id = None
        db.commit()
        return {"brand_id": str(brand.id), "parent_id": None}

    if str(parent_id) == str(brand_id):
        raise HTTPException(400, "A brand cannot be its own parent")

    parent = db.query(ClientBrand).filter(
        ClientBrand.id == parent_id,
        ClientBrand.client_id == scan.client_id,
    ).first()
    if not parent:
        raise HTTPException(404, "Parent brand not found for this client")

    # Cycle detection - walk up the candidate parent's ancestry; if we hit
    # `brand.id`, the proposed move would create a loop.
    cursor = parent
    depth = 0
    while cursor.parent_id is not None and depth < 64:
        if str(cursor.parent_id) == str(brand.id):
            raise HTTPException(400, "Cannot reparent: would create a cycle")
        cursor = db.query(ClientBrand).filter(ClientBrand.id == cursor.parent_id).first()
        if cursor is None:
            break
        depth += 1

    # Bucket sanity - both must be classified the same way in this scan, so a
    # competitor doesn't end up nested under a my_brand parent.
    brand_sbc = db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == scan_id,
        ScanBrandClassification.brand_id == brand.id,
    ).first()
    parent_sbc = db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == scan_id,
        ScanBrandClassification.brand_id == parent.id,
    ).first()
    brand_cls = brand_sbc.classification if brand_sbc else None
    parent_cls = parent_sbc.classification if parent_sbc else None
    if brand_cls != parent_cls:
        raise HTTPException(
            400,
            f"Brand bucket ({brand_cls}) does not match parent bucket ({parent_cls}). "
            f"Move the brand into the same bucket as the target parent first."
        )

    # Parent must be a ROOT in that bucket (single-level nesting - mirrors
    # the GET /brands tree-building logic that only nests one level).
    if parent.parent_id is not None:
        raise HTTPException(
            400,
            "Cannot nest under a brand that already has a parent - pick a root brand."
        )

    brand.parent_id = parent.id
    db.commit()
    return {"brand_id": str(brand.id), "parent_id": str(parent.id)}


@router.post("/{scan_id}/brands/import-from-brief")
async def import_competitors_from_brief(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Re-sync competitors + product lines from `scan.config.domain_brief`.

    The same logic runs automatically inside `generate_domain_brief` when the
    brief is first produced. This endpoint exposes a manual refresh so the
    user can re-pick up changes after editing the brief in the Gate 2 UI.

    Idempotent. Preserves existing my_brand / focus / ignored classifications.
    Re-parents orphan products (parent_id IS NULL) under their brief-declared
    brand; never re-parents rows that already have a parent (preserves any
    manual user reorganisation).
    """
    from sqlalchemy import func

    scan = _check_scan_access(scan_id, user, db)
    brief = (scan.config or {}).get("domain_brief") or {}
    competitors = brief.get("competitors") or []
    if not competitors:
        return {
            "created_brands": 0, "created_gammes": 0,
            "reparented": 0, "classified": 0, "skipped_existing": 0,
            "message": "no domain brief available",
        }

    def _classify_competitor(brand_id) -> str:
        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.brand_id == brand_id,
        ).first()
        if sbc is None:
            db.add(ScanBrandClassification(
                scan_id=scan_id, brand_id=brand_id,
                classification="competitor", is_focus=False,
                classified_by="brief", source="brief",
            ))
            return "classified"
        if sbc.classification == "my_brand" or sbc.is_focus:
            return "skipped_my_brand"
        if sbc.classification == "competitor":
            return "skipped_existing_competitor"
        if sbc.classification == "unclassified":
            sbc.classification = "competitor"
            sbc.classified_by = "brief"
            sbc.source = "brief"
            sbc.updated_at = datetime.utcnow()
            return "classified"
        # ignored - explicit user choice, don't override
        return "skipped_existing_ignored"

    created_brands = created_gammes = reparented = classified = skipped_existing = 0
    seen_brands: set[str] = set()
    for comp in competitors:
        comp_name = (comp.get("name") or "").strip() if isinstance(comp, dict) else ""
        from services.brand_name_norm import normalize_brand_name
        comp_norm = normalize_brand_name(comp_name)
        if not comp_norm or comp_norm in seen_brands:
            continue
        seen_brands.add(comp_norm)

        root = db.query(ClientBrand).filter(
            ClientBrand.client_id == scan.client_id,
            ClientBrand.canonical_name == comp_norm,
        ).first()
        if root is None:
            root = ClientBrand(
                client_id=scan.client_id, name=comp_name, canonical_name=comp_norm,
                detected_in_scan_id=scan_id, auto_detected=True,
                validated_by_user=False, detection_source="brief",
                last_seen_at=datetime.utcnow(),
            )
            db.add(root)
            db.flush()
            created_brands += 1
        else:
            root.last_seen_at = datetime.utcnow()

        action = _classify_competitor(root.id)
        if action == "classified":
            classified += 1
        else:
            skipped_existing += 1

        # Skip products if root resolved to anything but competitor/unclassified
        # (LLM hallucination guard - my_brand listed as competitor in brief).
        root_sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.brand_id == root.id,
        ).first()
        if root_sbc is None or root_sbc.classification not in ("competitor", "unclassified"):
            continue

        seen_gammes: set[str] = set()
        for prod_name in (comp.get("products") or []):
            prod_name = (prod_name or "").strip()
            prod_norm = normalize_brand_name(prod_name)
            if not prod_norm or prod_norm in seen_gammes:
                continue
            if prod_norm == comp_norm:
                continue
            seen_gammes.add(prod_norm)

            gamme = db.query(ClientBrand).filter(
                ClientBrand.client_id == scan.client_id,
                ClientBrand.canonical_name == prod_norm,
            ).first()
            if gamme is None:
                gamme = ClientBrand(
                    client_id=scan.client_id, name=prod_name, canonical_name=prod_norm,
                    parent_id=root.id, detected_in_scan_id=scan_id,
                    auto_detected=True, validated_by_user=False,
                    detection_source="brief", last_seen_at=datetime.utcnow(),
                )
                db.add(gamme)
                db.flush()
                created_gammes += 1
            else:
                gamme.last_seen_at = datetime.utcnow()
                # Re-parent ONLY if currently orphan (preserves user reorg)
                if gamme.parent_id is None:
                    gamme.parent_id = root.id
                    reparented += 1

            action = _classify_competitor(gamme.id)
            if action == "classified":
                classified += 1
            else:
                skipped_existing += 1

    scan.updated_at = datetime.utcnow()
    db.commit()
    return {
        "created_brands": created_brands,
        "created_gammes": created_gammes,
        "reparented": reparented,
        "classified": classified,
        "skipped_existing": skipped_existing,
    }


@router.post("/{scan_id}/brands/validate")
async def validate_scan_brands(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Gate 2: validate the per-scan brand classification and enqueue persona generation.

    Requires:
    - scan.status == 'brands_ready' (set by assign_keywords handler in J2 - until then this gate
      is not reachable from the happy path; the endpoint is wired up so Gate 2 UI can call it
      as soon as the worker refactor lands)
    - focus_brand_id IS NOT NULL
    - at least one brand classified as 'my_brand' (the focus itself counts)
    """
    scan = _check_scan_access(scan_id, user, db)

    # Sprint 15.3 hotfix : when the user skips the explicit "Validate
    # topics" Gate-1 click and lands straight on Brands tab, scan.status
    # is still 'topics_ready'. The "Continue to personas" intent is an
    # implicit topic validation — enqueue assign_keywords (which sets
    # status='brands_ready'), set status to 'assigning_keywords' so the
    # UI shows a clear progress state, and return 409 telling the UI to
    # retry in a few seconds. The worker handler is fast (~1s), so the
    # user just clicks again and it goes through. Future scans get the
    # full auto-chain from classify_topics → assign_keywords (handled
    # in worker/handlers/classify_topics.py same sprint).
    if scan.status == "topics_ready":
        active_topics = db.query(ScanTopic).filter(
            ScanTopic.scan_id == scan_id, ScanTopic.is_active == True
        ).count()
        if active_topics == 0:
            raise HTTPException(400, "Cannot validate brands : no active topics on this scan")
        _create_job(db, scan_id, "assign_keywords")
        scan.status = "assigning_keywords"
        scan.updated_at = datetime.utcnow()
        db.commit()
        raise HTTPException(
            409,
            "Linking keywords to topics... refresh the page in a few seconds and click again.",
        )

    if scan.status != "brands_ready":
        raise HTTPException(400, f"Cannot validate brands in status '{scan.status}' (expected 'brands_ready')")
    if not scan.focus_brand_id:
        raise HTTPException(400, "A focus brand must be selected before validating")

    my_brand_count = db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == scan_id,
        ScanBrandClassification.classification == "my_brand",
    ).count()
    if my_brand_count == 0:
        raise HTTPException(400, "At least one brand must be classified as 'my_brand'")

    # Enforce: the focus brand row is actually my_brand + is_focus=True
    focus_sbc = db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == scan_id,
        ScanBrandClassification.brand_id == scan.focus_brand_id,
    ).first()
    if not focus_sbc or focus_sbc.classification != "my_brand" or not focus_sbc.is_focus:
        raise HTTPException(400, "Focus brand row is inconsistent - reclassify it as my_brand with is_focus=true")

    # NOTE 2026-05-21 - auto-promote of focus_brand to client.primary_brand_ids
    # REMOVED here. The previous behavior prepended every validated focus brand
    # at index 0 of the workspace primaries, which polluted the workspace
    # whenever the user ran exploratory scans (test domains, one-off audits) -
    # foot-gun observed on Pierre Fabre when cocoonr.fr and praxedo.fr scans
    # added those non-PF brands as workspace primaries (with auto-generated
    # briefs via BB.6 chain). Documented as foot-gun #19 in
    # project_phase_brand_briefs.md.
    #
    # Explicit promotion to workspace primaries is now opt-in via
    # /app/settings/brands (drag Available → Selected → Save). The brand brief
    # auto-enqueue below stays untouched - briefs are per-brand identity, cheap,
    # and dormant on non-primary brands until they're promoted later (idempotent).

    scan.status = "generating_personas"
    scan.updated_at = datetime.utcnow()

    # Phase BB sync : enqueue generate_brand_brief FIRST, generate_personas
    # SECOND so the worker FIFO ensures the brief is in the DB when personas
    # boot (personas read focus_brand.brief via format_analysis_context).
    # Brief generation = 30-60s ; personas = 10-30 min. Order matters because
    # both jobs are queued in the same transaction - if personas is inserted
    # first, the worker picks it up first and personas miss the brief.
    # Idempotent - handler checks brand.brief IS NULL and the cap before regen.
    focus_brand = db.query(ClientBrand).filter(
        ClientBrand.id == scan.focus_brand_id
    ).first()
    if focus_brand and focus_brand.brief is None and \
            int(focus_brand.brief_generations_count or 0) < 3:
        in_flight_bb = (
            db.query(Job)
            .filter(
                Job.client_id == scan.client_id,
                Job.job_type == "generate_brand_brief",
                Job.status.in_(["pending", "running"]),
                Job.payload["brand_id"].astext == str(focus_brand.id),
            )
            .first()
        )
        if not in_flight_bb:
            db.add(Job(
                client_id=scan.client_id,
                job_type="generate_brand_brief",
                status="pending",
                payload={"brand_id": str(focus_brand.id)},
                max_attempts=2,
            ))
            logger.info(
                f"validate_brands: enqueued generate_brand_brief for focus "
                f"brand {focus_brand.id} ({focus_brand.name}) - runs BEFORE personas"
            )

    _create_job(db, scan_id, "generate_personas")

    db.commit()
    return {"status": "generating_personas", "my_brand_count": my_brand_count}


# --- Rescan + lineage ---

@router.post("/{scan_id}/rescan")
@limiter.limit("10/minute")
async def rescan(request: Request, scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a child scan that inherits topics, personas, questions and brand classifications
    from the parent. Skips Gate 1 and Gate 2 - goes straight to fetching_keywords (fresh HaloScan
    + fresh LLM) while reusing the validated setup.

    Phase 1: copies topics, personas, questions, scan_brand_classifications.
             Does NOT copy opportunities or llm_results (those are fresh per run).
    """
    parent = _check_scan_access(scan_id, user, db)

    # Count the active questions that will be copied to the child - this is
    # what `run_llm_tests` will execute, and what we must charge for.
    active_personas = db.query(ScanPersona).filter(
        ScanPersona.scan_id == parent.id, ScanPersona.is_active == True
    ).count()
    active_questions = (
        db.query(ScanQuestion)
        .join(ScanPersona, ScanPersona.id == ScanQuestion.persona_id)
        .filter(
            ScanQuestion.scan_id == parent.id,
            ScanQuestion.is_active == True,
            ScanPersona.is_active == True,
        )
        .count()
    )
    if active_personas == 0 or active_questions == 0:
        raise HTTPException(400, "Parent scan has no active personas/questions to rescan")

    # Sprint N-runs : rescan inherits parent.config (and thus runs_depth) by default.
    # Same credit formula as launch_scan : credits = questions × runs_depth.
    parent_config = parent.config or {}
    runs_depth = int(parent_config.get("runs_depth", 1)) or 1
    if runs_depth < 1:
        runs_depth = 1
    credits_needed = active_questions * runs_depth

    # Credit gate: same pattern as launch_scan - lock client, check balance,
    # then debit (with scan_id=child.id once it exists, so a worker failure
    # auto-refunds against the child).
    from routers.stripe import get_credit_balance, add_credits, lock_client_credits
    lock_client_credits(str(parent.client_id), db)
    balance = get_credit_balance(str(parent.client_id), "scan", db)
    if balance < credits_needed:
        raise HTTPException(402, {
            "error": "insufficient_credits",
            "need": credits_needed,
            "have": balance,
            "questions": active_questions,
            "runs_depth": runs_depth,
            "message": (
                f"Need {credits_needed} scan credits ({active_questions} questions "
                f"× {runs_depth} runs) but only {balance} available"
            ),
        })

    # Compute run_index: (max run_index of the lineage) + 1
    root_id = parent.parent_scan_id or parent.id
    max_run_index = db.query(func.max(Scan.run_index)).filter(
        (Scan.id == root_id) | (Scan.parent_scan_id == root_id)
    ).scalar() or 1

    child = Scan(
        client_id=parent.client_id,
        name=parent.name or parent.domain,
        domain=parent.domain,
        status="draft",
        focus_brand_id=parent.focus_brand_id,
        parent_scan_id=root_id,
        schedule=parent.schedule or "manual",
        run_index=max_run_index + 1,
        config=dict(parent.config or {}),
        created_by=user.id,
    )
    db.add(child)
    db.flush()  # need child.id for children rows

    # Copy topics (keep mapping old_topic_id → new_topic_id for persona.topic_id)
    topic_map: dict[str, str] = {}
    for t in db.query(ScanTopic).filter(ScanTopic.scan_id == parent.id).all():
        new_t = ScanTopic(
            scan_id=child.id,
            name=t.name,
            description=t.description,
            example_keywords=t.example_keywords,
            matching_terms=t.matching_terms,
            keyword_count=0,  # will be recomputed by assign_keywords handler on fresh HaloScan data
            is_active=t.is_active,
            display_order=t.display_order,
        )
        db.add(new_t)
        db.flush()
        topic_map[str(t.id)] = str(new_t.id)

    # Copy personas + questions
    for p in db.query(ScanPersona).filter(ScanPersona.scan_id == parent.id).all():
        new_p = ScanPersona(
            scan_id=child.id,
            topic_id=topic_map.get(str(p.topic_id)) if p.topic_id else None,
            name=p.name,
            data=p.data,
            is_active=p.is_active,
        )
        db.add(new_p)
        db.flush()
        for q in db.query(ScanQuestion).filter(ScanQuestion.persona_id == p.id).all():
            db.add(ScanQuestion(
                scan_id=child.id,
                persona_id=new_p.id,
                question=q.question,
                type_question=q.type_question,
                is_active=q.is_active,
                # Sprint P (migration 036): carry the 3 per-question fields on clone
                # so the duplicated scan has populated chips + judge-ready signals.
                intention_cachee=q.intention_cachee,
                signal_positif=q.signal_positif,
                signal_negatif=q.signal_negatif,
            ))

    # Copy brand classifications (same brand_ids, same focus)
    for sbc in db.query(ScanBrandClassification).filter(ScanBrandClassification.scan_id == parent.id).all():
        db.add(ScanBrandClassification(
            scan_id=child.id,
            brand_id=sbc.brand_id,
            classification=sbc.classification,
            is_focus=sbc.is_focus,
            classified_by="auto",
            source=sbc.source,
        ))

    # Copy brand-topic associations (using topic_map for new topic IDs)
    for bt in db.query(ScanBrandTopic).filter(ScanBrandTopic.scan_id == parent.id).all():
        new_topic_id = topic_map.get(str(bt.topic_id))
        if new_topic_id:
            db.add(ScanBrandTopic(
                scan_id=child.id,
                brand_id=bt.brand_id,
                topic_id=new_topic_id,
            ))

    # Rescan = benchmark tracker. Skip ALL intermediate steps (keywords, topics, brands,
    # personas). Just re-run LLM tests with the inherited setup for time-series comparison.
    child.status = "scanning"
    child.progress_message = "Re-running AI tests with same setup..."

    # Pre-debit credits - same lock from above is still held within this txn.
    add_credits(
        client_id=str(parent.client_id),
        credit_type="scan",
        amount=-credits_needed,
        description=(
            f"Rescan launched: {active_questions} questions × {runs_depth} runs"
        ),
        db=db,
        scan_id=str(child.id),
    )

    _create_job(db, str(child.id), "run_llm_tests", {
        "providers": (child.config or {}).get("providers", ["openai"]),
    })

    db.commit()
    db.refresh(child)
    return _serialize_scan(child)


@router.get("/{scan_id}/lineage")
async def get_scan_lineage(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the full lineage of a scan: root (initial scan) + all its rescans, ordered by run_index.

    Works whether scan_id points at the root or at any child - we resolve to the root first.
    """
    scan = _check_scan_access(scan_id, user, db)
    root_id = scan.parent_scan_id or scan.id

    # Fetch root + all children in one query
    lineage = db.query(Scan).filter(
        (Scan.id == root_id) | (Scan.parent_scan_id == root_id)
    ).order_by(Scan.run_index.asc(), Scan.created_at.asc()).all()

    return {
        "root_scan_id": str(root_id),
        "runs": [_serialize_scan(s) for s in lineage],
    }


# --- Fetch Keywords (Step 1) ---

@router.post("/{scan_id}/fetch-keywords")
async def fetch_keywords(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    scan = _check_scan_access(scan_id, user, db)
    if scan.status not in ("draft", "failed"):
        raise HTTPException(400, f"Cannot fetch keywords in status '{scan.status}'")

    scan.status = "fetching_keywords"
    scan.updated_at = datetime.utcnow()
    cfg = scan.config or {}
    _create_job(db, scan_id, "fetch_keywords", {
        "domain": scan.domain,
        "max_position": cfg.get("max_position", 50),
        "max_urls": cfg.get("max_urls", 2000),
    })
    db.commit()
    return {"status": "job_created", "scan_status": scan.status}


@router.get("/{scan_id}/keywords")
async def get_keywords(
    scan_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_scan_access(scan_id, user, db)
    offset = (page - 1) * limit
    keywords = (
        db.query(ScanKeyword)
        .filter(ScanKeyword.scan_id == scan_id)
        .order_by(ScanKeyword.traffic.desc().nullslast())
        .offset(offset)
        .limit(limit)
        .all()
    )
    total = db.query(ScanKeyword).filter(ScanKeyword.scan_id == scan_id).count()
    return {
        "items": [
            {"id": str(k.id), "url": k.url, "keyword": k.keyword,
             "position": k.position, "traffic": k.traffic, "search_volume": k.search_volume}
            for k in keywords
        ],
        "total": total,
        "page": page,
        "limit": limit,
    }


# --- Topics (Gate 1) ---

@router.get("/{scan_id}/topics")
async def get_topics(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_scan_access(scan_id, user, db)
    topics = db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).order_by(ScanTopic.display_order).all()

    result = []
    for t in topics:
        # Get keywords for this topic
        topic_kws = db.query(ScanKeyword).filter(ScanKeyword.topic_id == t.id).order_by(ScanKeyword.traffic.desc().nullslast()).all()

        # Top 5 keywords - deduplicated by keyword text, aggregated across pages
        # HaloScan returns 1 row per (keyword, url) pair; we collapse into 1 row per concept
        kw_agg = {}
        for k in topic_kws:
            if k.keyword not in kw_agg:
                kw_agg[k.keyword] = {"keyword": k.keyword, "traffic": 0, "position": None, "volume": k.search_volume, "pages": set()}
            agg = kw_agg[k.keyword]
            agg["traffic"] += (k.traffic or 0)
            if k.position is not None and (agg["position"] is None or k.position < agg["position"]):
                agg["position"] = k.position
            if k.search_volume and (not agg["volume"] or k.search_volume > agg["volume"]):
                agg["volume"] = k.search_volume
            if k.url:
                agg["pages"].add(k.url)
        top_keywords = sorted(
            [{"keyword": a["keyword"], "traffic": a["traffic"] or None, "position": a["position"], "volume": a["volume"], "pages_count": len(a["pages"])} for a in kw_agg.values()],
            key=lambda x: x["traffic"] or 0,
            reverse=True,
        )[:5]
        distinct_keyword_count = len(kw_agg)

        # Top 5 unique URLs by traffic, with keyword count per URL
        url_kw_count = {}
        url_traffic = {}
        for k in topic_kws:
            if k.url:
                url_kw_count[k.url] = url_kw_count.get(k.url, 0) + 1
                url_traffic[k.url] = (url_traffic.get(k.url, 0) or 0) + (k.traffic or 0)

        sorted_urls = sorted(url_kw_count.keys(), key=lambda u: url_traffic.get(u, 0), reverse=True)
        # Include top keywords per URL for drill-down
        url_top_kws = {}
        for k in topic_kws:
            if k.url:
                if k.url not in url_top_kws:
                    url_top_kws[k.url] = []
                if len(url_top_kws[k.url]) < 10:
                    url_top_kws[k.url].append({
                        "keyword": k.keyword, "traffic": k.traffic,
                        "position": k.position, "volume": k.search_volume,
                    })
        url_positions = {}
        for k in topic_kws:
            if k.url and k.position:
                if k.url not in url_positions:
                    url_positions[k.url] = []
                url_positions[k.url].append(k.position)

        url_volume = {}
        for k in topic_kws:
            if k.url:
                url_volume[k.url] = (url_volume.get(k.url, 0) or 0) + (k.search_volume or 0)

        top_urls = [{
            "url": u, "keywords_count": url_kw_count[u],
            "search_volume": url_volume.get(u, 0),
            "traffic": url_traffic[u],
            "avg_position": round(sum(url_positions.get(u, [0])) / max(len(url_positions.get(u, [1])), 1), 1),
            "top_keywords": url_top_kws.get(u, []),
        } for u in sorted_urls[:5]]

        result.append({
            "id": str(t.id), "name": t.name, "description": t.description,
            "example_keywords": t.example_keywords, "matching_terms": t.matching_terms,
            "keyword_count": distinct_keyword_count, "is_active": t.is_active, "display_order": t.display_order,
            "top_keywords": top_keywords,
            "top_urls": top_urls,
            "total_urls": len(url_kw_count),
        })

    unassigned = db.query(ScanKeyword).filter(
        ScanKeyword.scan_id == scan_id, ScanKeyword.topic_id == None
    ).count()
    return {"topics": result, "unassigned_keywords": unassigned}


@router.get("/{scan_id}/topics/{topic_id}/urls")
async def get_topic_urls(scan_id: str, topic_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all unique URLs assigned to a topic with full data."""
    _check_scan_access(scan_id, user, db)
    keywords = db.query(ScanKeyword).filter(
        ScanKeyword.topic_id == topic_id,
    ).order_by(ScanKeyword.traffic.desc().nullslast()).all()

    url_data = {}
    url_kws = {}
    for k in keywords:
        if not k.url:
            continue
        if k.url not in url_data:
            url_data[k.url] = {"keywords_count": 0, "traffic": 0, "search_volume": 0, "positions": []}
            url_kws[k.url] = []
        url_data[k.url]["keywords_count"] += 1
        url_data[k.url]["traffic"] += (k.traffic or 0)
        url_data[k.url]["search_volume"] += (k.search_volume or 0)
        if k.position:
            url_data[k.url]["positions"].append(k.position)
        if len(url_kws[k.url]) < 10:
            url_kws[k.url].append({"keyword": k.keyword, "traffic": k.traffic, "position": k.position, "volume": k.search_volume})

    urls = []
    for u in sorted(url_data.keys(), key=lambda x: url_data[x]["traffic"], reverse=True):
        d = url_data[u]
        pos_list = d["positions"]
        urls.append({
            "url": u,
            "keywords_count": d["keywords_count"],
            "search_volume": d["search_volume"],
            "traffic": d["traffic"],
            "avg_position": round(sum(pos_list) / max(len(pos_list), 1), 1) if pos_list else None,
            "top_keywords": url_kws.get(u, []),
        })
    return {"urls": urls, "total": len(urls)}


@router.get("/{scan_id}/topics/{topic_id}/keywords")
async def get_topic_keywords(scan_id: str, topic_id: str, limit: int = Query(50), user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Get keywords for a topic, deduplicated by keyword text.

    HaloScan returns 1 row per (keyword, url) pair. We aggregate:
    - traffic = SUM across all pages ranking for this keyword
    - volume = MAX (same for all URLs of a keyword, but MAX is safe)
    - position = MIN (best ranking across pages)
    - pages_count = number of distinct pages where this keyword ranks
    """
    _check_scan_access(scan_id, user, db)
    rows = db.query(
        ScanKeyword.keyword,
        func.sum(ScanKeyword.traffic).label("traffic"),
        func.max(ScanKeyword.search_volume).label("volume"),
        func.min(ScanKeyword.position).label("position"),
        func.count(func.distinct(ScanKeyword.url)).label("pages_count"),
    ).filter(
        ScanKeyword.topic_id == topic_id,
    ).group_by(ScanKeyword.keyword).order_by(func.sum(ScanKeyword.traffic).desc().nullslast()).limit(limit).all()

    total_distinct = db.query(func.count(func.distinct(ScanKeyword.keyword))).filter(
        ScanKeyword.topic_id == topic_id,
    ).scalar() or 0

    return {
        "keywords": [
            {
                "keyword": r.keyword,
                "traffic": int(r.traffic) if r.traffic else None,
                "volume": int(r.volume) if r.volume else None,
                "position": int(r.position) if r.position else None,
                "pages_count": int(r.pages_count) if r.pages_count else 0,
            }
            for r in rows
        ],
        "total": total_distinct,
    }


@router.post("/{scan_id}/topics/{topic_id}/move-url")
async def move_url_to_topic(scan_id: str, topic_id: str, req: MoveUrlRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Move all keywords with a given URL to a different topic."""
    _check_scan_access(scan_id, user, db)
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL required")

    # Capture source topic IDs before update to recalc their counts after
    source_topic_ids = [
        tid for (tid,) in db.query(ScanKeyword.topic_id).filter(
            ScanKeyword.scan_id == scan_id,
            ScanKeyword.url == url,
        ).distinct().all() if tid
    ]

    # Update all keywords with this URL to the new topic
    updated = db.query(ScanKeyword).filter(
        ScanKeyword.scan_id == scan_id,
        ScanKeyword.url == url,
    ).update({ScanKeyword.topic_id: topic_id})

    # Recalc keyword_count for source topics and target topic (distinct count)
    source_counts = {}
    for sid in source_topic_ids:
        if str(sid) != str(topic_id):
            source_counts[str(sid)] = _recalc_topic_keyword_count(sid, db)
    target_count = _recalc_topic_keyword_count(topic_id, db)

    db.commit()
    return {
        "moved": updated,
        "url": url,
        "to_topic": topic_id,
        "source_counts": source_counts,
        "target_count": target_count,
    }


@router.post("/{scan_id}/topics/{topic_id}/move-keyword")
async def move_keyword_to_topic(scan_id: str, topic_id: str, req: MoveKeywordRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Move all rows of a given keyword (across all its URLs) to a different topic.

    HaloScan may return the same keyword for multiple URLs (different positions).
    Moving the keyword moves ALL those rows → ALL their URLs flow with it to the target.
    The URL's membership in a topic is defined by its keyword rows in that topic:
    - Source: URL disappears from source only if no other keyword rows link to it there
    - Target: URL appears (if not already via another keyword)
    """
    _check_scan_access(scan_id, user, db)
    keyword = req.keyword.strip()
    source_topic_id = req.source_topic_id.strip()
    if not keyword or not source_topic_id:
        raise HTTPException(400, "keyword and source_topic_id required")

    updated = db.query(ScanKeyword).filter(
        ScanKeyword.scan_id == scan_id,
        ScanKeyword.keyword == keyword,
        ScanKeyword.topic_id == source_topic_id,
    ).update({ScanKeyword.topic_id: topic_id})

    source_count = _recalc_topic_keyword_count(source_topic_id, db)
    target_count = _recalc_topic_keyword_count(topic_id, db)
    source_total_urls = _count_topic_urls(source_topic_id, db)
    target_total_urls = _count_topic_urls(topic_id, db)

    db.commit()
    return {
        "moved": updated,
        "keyword": keyword,
        "from_topic": source_topic_id,
        "to_topic": topic_id,
        "source_count": source_count,
        "target_count": target_count,
        "source_total_urls": source_total_urls,
        "target_total_urls": target_total_urls,
    }


@router.post("/{scan_id}/topics")
async def create_topic(scan_id: str, req: TopicCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_scan_access(scan_id, user, db)
    topic = ScanTopic(scan_id=scan_id, name=strip_tags(req.name), description=strip_tags(req.description))
    db.add(topic)
    db.commit()
    db.refresh(topic)
    return {"id": str(topic.id), "name": topic.name, "description": topic.description}


@router.patch("/{scan_id}/topics/{topic_id}")
async def update_topic(scan_id: str, topic_id: str, req: TopicUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_scan_access(scan_id, user, db)
    topic = db.query(ScanTopic).filter(ScanTopic.id == topic_id, ScanTopic.scan_id == scan_id).first()
    if not topic:
        raise HTTPException(404, "Topic not found")
    if req.name is not None:
        topic.name = strip_tags(req.name)
    if req.description is not None:
        topic.description = strip_tags(req.description)

    cascaded_personas = 0
    cascaded_questions = 0
    if req.is_active is not None and bool(req.is_active) != bool(topic.is_active):
        # Cascade is_active to every persona + question under this topic so
        # the Topics and Personas pages can't disagree on which work the
        # scanner will run. Without this, a topic toggled OFF on Gate 1 still
        # leaves its personas active in the launch query (which filters on
        # persona.is_active, not topic.is_active) → ghost work + credit drift.
        # The Personas page bulk toggle is the symmetric path: it patches
        # the topic + each persona explicitly, so the two routes converge.
        new_state = bool(req.is_active)
        topic.is_active = new_state

        persona_ids = [
            p.id for p in db.query(ScanPersona.id).filter(
                ScanPersona.scan_id == scan_id,
                ScanPersona.topic_id == topic_id,
            ).all()
        ]
        if persona_ids:
            cascaded_personas = db.query(ScanPersona).filter(
                ScanPersona.id.in_(persona_ids),
                ScanPersona.is_active != new_state,
            ).update({ScanPersona.is_active: new_state}, synchronize_session=False)
            cascaded_questions = db.query(ScanQuestion).filter(
                ScanQuestion.persona_id.in_(persona_ids),
                ScanQuestion.is_active != new_state,
            ).update({ScanQuestion.is_active: new_state}, synchronize_session=False)

    db.commit()
    return {
        "id": str(topic.id),
        "name": topic.name,
        "is_active": topic.is_active,
        "cascaded_personas": cascaded_personas,
        "cascaded_questions": cascaded_questions,
    }


@router.delete("/{scan_id}/topics/{topic_id}")
async def delete_topic(scan_id: str, topic_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_scan_access(scan_id, user, db)
    topic = db.query(ScanTopic).filter(ScanTopic.id == topic_id, ScanTopic.scan_id == scan_id).first()
    if not topic:
        raise HTTPException(404, "Topic not found")
    db.delete(topic)
    db.commit()
    return {"deleted": True}


@router.post("/{scan_id}/topics/auto-classify")
async def auto_classify_topics(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    scan = _check_scan_access(scan_id, user, db)
    kw_count = db.query(ScanKeyword).filter(ScanKeyword.scan_id == scan_id).count()
    if kw_count == 0:
        raise HTTPException(400, "No keywords fetched yet")

    # Idempotency guard: don't create a duplicate classify_topics job if one is
    # already pending or running. The page auto-refreshes every 3s and re-triggers
    # this endpoint while status is 'keywords_fetched' - without this guard, each
    # refresh creates a new job that DELETES all existing topics and recreates them
    # with new IDs, breaking any user interaction that started between refreshes.
    existing = db.query(Job).filter(
        Job.scan_id == scan_id,
        Job.job_type == "classify_topics",
        Job.status.in_(["pending", "running"]),
    ).first()
    if existing:
        return {"status": "already_queued", "job_id": str(existing.id)}

    _create_job(db, scan_id, "classify_topics")
    db.commit()
    return {"status": "job_created"}


@router.post("/{scan_id}/topics/validate")
async def validate_topics(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    scan = _check_scan_access(scan_id, user, db)
    active_topics = db.query(ScanTopic).filter(
        ScanTopic.scan_id == scan_id, ScanTopic.is_active == True
    ).count()
    if active_topics == 0:
        raise HTTPException(400, "At least one active topic required")

    scan.status = "assigning_keywords"
    scan.updated_at = datetime.utcnow()
    _create_job(db, scan_id, "assign_keywords")
    db.commit()
    return {"status": "assigning_keywords", "active_topics": active_topics}


# --- Personas + Questions (Gate 2) ---

@router.get("/{scan_id}/personas")
async def get_personas(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return personas grouped by topic with per-level stats.

    Shape:
    {
      "topics": [
        {
          "topic_id": "...", "topic_name": "Eczema", "topic_is_active": true,
          "personas": [
            {
              "id": "...", "name": "...", "data": {...}, "is_active": true,
              "topic_id": "...",
              "questions": [{"id","question","type_question","is_active"}, ...],
              "stats": {"total_questions": 15, "active_questions": 14}
            }, ...
          ],
          "stats": {
            "total_personas": 6, "active_personas": 6,
            "total_questions": 90, "active_questions": 87
          }
        }, ...
      ],
      "orphan_personas": [...],   # personas with topic_id NULL or pointing to a deleted topic
      "totals": {
        "active_personas": 28, "total_personas": 30,
        "active_questions": 420, "total_questions": 450
      }
    }
    """
    _check_scan_access(scan_id, user, db)

    topics = db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).order_by(ScanTopic.display_order).all()
    personas = db.query(ScanPersona).filter(ScanPersona.scan_id == scan_id).all()
    questions = db.query(ScanQuestion).filter(ScanQuestion.scan_id == scan_id).all()

    # Index questions by persona_id for O(1) lookup
    questions_by_persona: dict[str, list] = {}
    for q in questions:
        questions_by_persona.setdefault(str(q.persona_id), []).append(q)

    def _serialize_question(q, extras_by_text):
        # Sprint P (migration 036): the 3 per-question fields are now native
        # columns. Fallback to the legacy JSONB text-lookup for rows generated
        # before the migration backfill (NULL columns), since the JSONB blob on
        # scan_personas.data.questions[] is still written by both generators
        # during the transition release.
        x = extras_by_text.get(q.question) or {}
        return {
            "id": str(q.id),
            "question": q.question,
            "type_question": q.type_question,
            "is_active": bool(q.is_active),
            "intention_cachee": (q.intention_cachee or x.get("intention_cachee") or ""),
            "signal_positif": (q.signal_positif or x.get("signal_positif") or ""),
            "signal_negatif": (q.signal_negatif or x.get("signal_negatif") or ""),
        }

    def _serialize_persona(p):
        p_questions = questions_by_persona.get(str(p.id), [])
        extras_by_text = {
            (eq.get("question") or "").strip(): eq
            for eq in ((p.data or {}).get("questions") or [])
            if isinstance(eq, dict)
        }
        return {
            "id": str(p.id),
            "name": p.name,
            "data": p.data,
            "topic_id": str(p.topic_id) if p.topic_id else None,
            "is_active": bool(p.is_active),
            "questions": [_serialize_question(q, extras_by_text) for q in p_questions],
            "stats": {
                "total_questions": len(p_questions),
                "active_questions": sum(1 for q in p_questions if q.is_active),
            },
        }

    # Group personas by topic
    topic_map = {str(t.id): t for t in topics}
    personas_by_topic: dict[str, list] = {}
    orphan_personas = []
    for p in personas:
        tid = str(p.topic_id) if p.topic_id else None
        if tid and tid in topic_map:
            personas_by_topic.setdefault(tid, []).append(p)
        else:
            orphan_personas.append(p)

    topics_out = []
    total_personas = 0
    active_personas_total = 0
    total_questions = 0
    active_questions_total = 0

    for t in topics:
        t_personas = personas_by_topic.get(str(t.id), [])
        t_personas_serialized = [_serialize_persona(p) for p in t_personas]
        t_total_q = sum(pp["stats"]["total_questions"] for pp in t_personas_serialized)
        # Effective active questions: only count questions under ACTIVE personas
        # (a persona toggled off = all its questions excluded from the scan)
        t_active_q = sum(pp["stats"]["active_questions"] for pp in t_personas_serialized if pp["is_active"])
        t_active_p = sum(1 for pp in t_personas_serialized if pp["is_active"])
        topics_out.append({
            "topic_id": str(t.id),
            "topic_name": t.name,
            "topic_is_active": bool(t.is_active),
            "keyword_count": t.keyword_count or 0,  # exposed for personas UI ("X of N keywords in topic")
            "personas": t_personas_serialized,
            "stats": {
                "total_personas": len(t_personas_serialized),
                "active_personas": t_active_p,
                "total_questions": t_total_q,
                "active_questions": t_active_q,
            },
        })
        total_personas += len(t_personas_serialized)
        active_personas_total += t_active_p
        total_questions += t_total_q
        active_questions_total += t_active_q

    orphan_serialized = [_serialize_persona(p) for p in orphan_personas]
    for pp in orphan_serialized:
        total_personas += 1
        if pp["is_active"]:
            active_personas_total += 1
            active_questions_total += pp["stats"]["active_questions"]
        total_questions += pp["stats"]["total_questions"]
        active_questions_total += pp["stats"]["active_questions"]

    return {
        "topics": topics_out,
        "orphan_personas": orphan_serialized,
        "totals": {
            "total_personas": total_personas,
            "active_personas": active_personas_total,
            "total_questions": total_questions,
            "active_questions": active_questions_total,
        },
    }


@router.post("/{scan_id}/personas")
async def create_persona(scan_id: str, req: PersonaCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_scan_access(scan_id, user, db)
    persona = ScanPersona(scan_id=scan_id, topic_id=req.topic_id, name=strip_tags(req.name), data=req.data)
    db.add(persona)
    db.commit()
    db.refresh(persona)
    return {"id": str(persona.id), "name": persona.name}


@router.patch("/{scan_id}/personas/{persona_id}")
async def update_persona(scan_id: str, persona_id: str, req: PersonaUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Update a persona - all fields optional. Used for rename, toggle, reassign."""
    _check_scan_access(scan_id, user, db)
    persona = db.query(ScanPersona).filter(ScanPersona.id == persona_id, ScanPersona.scan_id == scan_id).first()
    if not persona:
        raise HTTPException(404, "Persona not found")
    if req.name is not None:
        persona.name = strip_tags(req.name)
    if req.data is not None:
        persona.data = req.data
    if req.topic_id is not None:
        persona.topic_id = req.topic_id
    if req.is_active is not None:
        persona.is_active = req.is_active
    db.commit()
    return {
        "id": str(persona.id),
        "name": persona.name,
        "is_active": bool(persona.is_active),
        "topic_id": str(persona.topic_id) if persona.topic_id else None,
    }


@router.delete("/{scan_id}/personas/{persona_id}")
async def delete_persona(scan_id: str, persona_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_scan_access(scan_id, user, db)
    persona = db.query(ScanPersona).filter(ScanPersona.id == persona_id, ScanPersona.scan_id == scan_id).first()
    if not persona:
        raise HTTPException(404, "Persona not found")
    db.delete(persona)
    db.commit()
    return {"deleted": True}


# Hard cap on per-persona question-regen. Each call fires a Claude haiku
# request (~$0.01-0.03). The first generation runs on persona creation;
# regenerations beyond 5 indicate the user isn't getting what they want
# from the LLM and should edit questions manually.
# See feedback_cap_user_triggered_llm_ops.
MAX_PERSONA_QUESTIONS_GENERATIONS = 5


@router.post("/{scan_id}/personas/{persona_id}/generate-questions")
@limiter.limit("5/minute")
async def generate_persona_questions(request: Request, scan_id: str, persona_id: str,
                                     user=Depends(get_current_user),
                                     db: Session = Depends(get_db)):
    """Enqueue a worker job to generate 15 questions for a custom persona.

    Capped at MAX_PERSONA_QUESTIONS_GENERATIONS to bound LLM spend if the
    user spam-clicks regenerate. Counter lives in persona.data and is
    incremented by the worker on success.
    """
    _check_scan_access(scan_id, user, db)
    persona = db.query(ScanPersona).filter(ScanPersona.id == persona_id, ScanPersona.scan_id == scan_id).first()
    if not persona:
        raise HTTPException(404, "Persona not found")

    # Hard cap : 429 once the persona has burned the budget.
    used = int((persona.data or {}).get("questions_generations_count") or 0)
    if used >= MAX_PERSONA_QUESTIONS_GENERATIONS:
        raise HTTPException(429, {
            "error": "persona_questions_regen_cap_reached",
            "message": f"Questions have been regenerated {used} times for this persona "
                       f"(max {MAX_PERSONA_QUESTIONS_GENERATIONS}). Edit questions manually "
                       f"- further regenerations are blocked.",
            "generations_used": used,
            "cap": MAX_PERSONA_QUESTIONS_GENERATIONS,
        })

    # In-flight dedupe : don't double-enqueue if the user clicks twice.
    in_flight = db.query(Job).filter(
        Job.scan_id == scan_id,
        Job.job_type == "generate_persona_questions",
        Job.status.in_(["pending", "running"]),
    ).all()
    for j in in_flight:
        if (j.payload or {}).get("persona_id") == persona_id:
            return {"status": "already_running", "persona_id": persona_id,
                    "job_id": str(j.id),
                    "generations_used": used, "cap": MAX_PERSONA_QUESTIONS_GENERATIONS}

    db.add(Job(scan_id=scan_id, job_type="generate_persona_questions", payload={"persona_id": persona_id}))
    db.commit()
    return {"status": "generating", "persona_id": persona_id,
            "generations_used": used, "cap": MAX_PERSONA_QUESTIONS_GENERATIONS}


@router.post("/{scan_id}/questions")
async def create_question(scan_id: str, req: QuestionCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_scan_access(scan_id, user, db)
    q = ScanQuestion(scan_id=scan_id, persona_id=req.persona_id, question=strip_tags(req.question), type_question=req.type_question)
    db.add(q)
    db.commit()
    db.refresh(q)
    return {"id": str(q.id), "question": q.question}


@router.patch("/{scan_id}/questions/{question_id}")
async def update_question(scan_id: str, question_id: str, req: QuestionUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Update a question - all fields optional. Used for toggle, inline edit, retype."""
    _check_scan_access(scan_id, user, db)
    q = db.query(ScanQuestion).filter(ScanQuestion.id == question_id, ScanQuestion.scan_id == scan_id).first()
    if not q:
        raise HTTPException(404, "Question not found")
    if req.question is not None:
        text = strip_tags(req.question)
        if not text:
            raise HTTPException(400, "question cannot be empty")
        q.question = text
    if req.type_question is not None:
        q.type_question = req.type_question
    if req.is_active is not None:
        q.is_active = req.is_active
    db.commit()
    return {
        "id": str(q.id),
        "question": q.question,
        "type_question": q.type_question,
        "is_active": bool(q.is_active),
    }


@router.delete("/{scan_id}/questions/{question_id}")
async def delete_question(scan_id: str, question_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_scan_access(scan_id, user, db)
    q = db.query(ScanQuestion).filter(ScanQuestion.id == question_id, ScanQuestion.scan_id == scan_id).first()
    if not q:
        raise HTTPException(404, "Question not found")
    db.delete(q)
    db.commit()
    return {"deleted": True}


@router.post("/{scan_id}/launch")
@limiter.limit("10/minute")
async def launch_scan(request: Request, scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    scan = _check_scan_access(scan_id, user, db)

    active_personas = db.query(ScanPersona).filter(
        ScanPersona.scan_id == scan_id, ScanPersona.is_active == True
    ).count()
    # Only count questions whose persona is also active (same logic as run_llm_tests)
    active_questions = (
        db.query(ScanQuestion)
        .join(ScanPersona, ScanPersona.id == ScanQuestion.persona_id)
        .filter(
            ScanQuestion.scan_id == scan_id,
            ScanQuestion.is_active == True,
            ScanPersona.is_active == True,
        )
        .count()
    )
    if active_personas == 0 or active_questions == 0:
        raise HTTPException(400, "Need at least one active persona and question")

    # Sprint N-runs : credits scale with runs_depth (statistical multi-sampling).
    # 1 credit = 1 question × 1 run. Default runs_depth=1 keeps legacy semantics
    # until Sprint 3 flips the default to 10.
    config = scan.config or {}
    runs_depth = int(config.get("runs_depth", 1)) or 1
    if runs_depth < 1:
        runs_depth = 1
    credits_needed = active_questions * runs_depth

    # Credit check: debit scan credits (credits_needed = questions × runs).
    # Lock the client row FIRST so the balance read + debit are atomic.
    # Without the lock, two concurrent launches could both observe the
    # same balance and each pass the check, causing a double-spend.
    #
    # Bypass when scan.config.credits_already_debited = True - set by
    # import scripts (e.g. worker/scripts/import_seollm_avene.py) that
    # bring in personas + questions whose underlying work was already
    # paid for upstream. The real API costs still apply, but the sen-ai
    # ledger doesn't double-charge.
    bypass_credits = bool(config.get("credits_already_debited"))

    from routers.stripe import get_credit_balance, add_credits, lock_client_credits
    if not bypass_credits:
        lock_client_credits(str(scan.client_id), db)
        balance = get_credit_balance(str(scan.client_id), "scan", db)
        if balance < credits_needed:
            raise HTTPException(402, {
                "error": "insufficient_credits",
                "need": credits_needed,
                "have": balance,
                "questions": active_questions,
                "runs_depth": runs_depth,
                "message": (
                    f"Need {credits_needed} scan credits ({active_questions} questions "
                    f"× {runs_depth} runs) but only {balance} available"
                ),
            })

        # Pre-debit credits (re-uses the same lock - re-entrant within this txn)
        add_credits(
            client_id=str(scan.client_id),
            credit_type="scan",
            amount=-credits_needed,
            description=(
                f"Scan launched: {active_questions} questions × {runs_depth} runs"
            ),
            db=db,
            scan_id=scan_id,
        )
    else:
        balance = get_credit_balance(str(scan.client_id), "scan", db)
        import logging
        logging.getLogger(__name__).info(
            f"launch_scan: bypassing credit debit for scan {scan_id} "
            f"({active_questions} questions × {runs_depth} runs, "
            f"import_origin={config.get('import_origin')})"
        )

    scan.status = "scanning"
    scan.started_at = datetime.utcnow()
    scan.updated_at = datetime.utcnow()
    scan.progress_pct = 0
    scan.progress_message = "Démarrage du scan..."
    _create_job(db, scan_id, "run_llm_tests", {
        "providers": (scan.config or {}).get("providers", ["openai"]),
    })
    audit_log(db, action="scan.launch", user_id=str(user.id),
              target_type="scan", target_id=scan_id,
              ip=request.client.host if request.client else None,
              details={
                  "questions": active_questions,
                  "runs_depth": runs_depth,
                  "credits_used": credits_needed,
              })
    db.commit()
    return {
        "status": "scanning",
        "credits_used": credits_needed,
        "credits_remaining": balance - credits_needed,
        "questions": active_questions,
        "runs_depth": runs_depth,
    }


@router.post("/{scan_id}/retry")
@limiter.limit("10/minute")
async def retry_scan(request: Request, scan_id: str,
                     user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Re-enqueue failed jobs for a scan instead of starting from scratch.

    When a scan fails - typically because a downstream Claude/OpenAI call hits
    an API outage or quota limit - the worker marks the scan failed and
    auto-refunds credits. This endpoint reverses that: pending jobs that ran
    successfully (e.g., run_llm_tests, generate_opportunities) keep their
    completed state, and only the actually-failed jobs are reset to pending.
    Scan status flips back to whatever it should be given the job state.

    Note on credits: when a scan is auto-refunded, retrying gets the deferred
    work for free. We accept this small leak - the case is rare (only when
    upstream succeeded but downstream failed) and the user already paid the
    cost in waiting / dealing with a failure they didn't cause. Re-debiting
    here would require knowing how much was refunded, which means scanning
    the credit ledger - overkill for a recovery path.
    """
    scan = _check_scan_access(scan_id, user, db)

    if scan.status != "failed":
        raise HTTPException(400, {
            "error": "scan_not_failed",
            "message": f"Scan is in status '{scan.status}', nothing to retry",
        })

    # Permanent failures (e.g., HaloScan has no data for the domain) are
    # flagged at handler level via PermanentScanError. Retrying would just
    # re-fail with the same message - block it.
    if (scan.summary or {}).get("retryable") is False:
        raise HTTPException(400, {
            "error": "permanent_failure",
            "message": "This scan can't be retried - the failure is permanent (e.g., the domain has no data). Start a new scan with different inputs.",
        })

    failed_jobs = (
        db.query(Job)
        .filter(Job.scan_id == scan_id, Job.status == "failed")
        .all()
    )
    if not failed_jobs:
        raise HTTPException(400, {
            "error": "no_failed_jobs",
            "message": "Scan is marked failed but has no failed jobs to retry",
        })

    # Reset failed jobs so the worker picks them up next poll
    for j in failed_jobs:
        j.status = "pending"
        j.attempts = 0
        j.completed_at = None
        j.started_at = None
        j.result = None

    # Did upstream succeed? If run_llm_tests is completed, the scan is
    # essentially done - only the chained post-processing failed. Status
    # should be "scanning" so the UI shows progress (worker flips it to
    # completed once all chained jobs finish, in run_llm_tests' final step).
    # If run_llm_tests itself failed, status stays "scanning" too - same flow.
    scan.status = "scanning"
    scan.error_message = None
    scan.updated_at = datetime.utcnow()

    audit_log(db, action="scan.retry", user_id=str(user.id),
              target_type="scan", target_id=scan_id,
              ip=request.client.host if request.client else None,
              details={"reset_jobs": [j.job_type for j in failed_jobs]})
    db.commit()

    return {
        "status": "scanning",
        "reset_jobs": [j.job_type for j in failed_jobs],
        "message": f"Re-queued {len(failed_jobs)} failed job(s)",
    }


# --- Results ---

@router.get("/{scan_id}/results")
async def get_results(scan_id: str, provider: str | None = Query(None), user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Get scan results: overview, per-persona, competitors. Optional provider filter."""
    scan = _check_scan_access(scan_id, user, db)

    q = db.query(ScanLLMResult).filter(ScanLLMResult.scan_id == scan_id)
    if provider and provider != 'all':
        q = q.filter(ScanLLMResult.provider == provider)
    results = q.all()
    if not results:
        return {"overview": None, "by_persona": [], "competitors": [], "details": []}

    # --- Overview KPIs ---
    # Primary metric: brand mention rate (brand name in AI response text)
    # Secondary: domain citation rate (domain URL in sources)
    total = len(results)
    brand_mentioned = sum(1 for r in results if (r.brand_analysis or {}).get("marque_cible_mentionnee"))
    brand_mention_rate = round(brand_mentioned / total * 100, 1) if total else 0
    domain_cited = sum(1 for r in results if r.target_cited)
    domain_citation_rate = round(domain_cited / total * 100, 1) if total else 0
    avg_position = None
    positions = [r.target_position for r in results if r.target_position]
    if positions:
        avg_position = round(sum(positions) / len(positions), 1)

    # Sprint J: fetch judgments for these llm_results in one query (avoid N+1).
    # Indexed by scan_llm_result_id ; UI uses these to render Pos/Neg chips.
    # NOTE: this block was duplicated further down and the variable was being
    # used before assignment - moved up here so the aggregations below can use it.
    judgments_by_result_id = {
        str(j.scan_llm_result_id): j
        for j in db.query(ScanQuestionJudgment).filter(
            ScanQuestionJudgment.scan_id == scan_id
        ).all()
    }

    # Sprint M aggregations: SOV by entity_type + 4-issues funnel from judgments.
    from services.composite_scores import aggregate_entity_sov, aggregate_judgment_funnel
    entity_sov = aggregate_entity_sov(r.brand_mentions or [] for r in results)
    all_judgments_for_scan = list(judgments_by_result_id.values())
    judgment_funnel = aggregate_judgment_funnel(all_judgments_for_scan)
    judgment_coverage = {
        "judged_responses": len(all_judgments_for_scan),
        "total_responses": total,
        "coverage_pct": round(len(all_judgments_for_scan) / total * 100, 1) if total else 0,
    }

    # --- By Persona ---
    personas = db.query(ScanPersona).filter(ScanPersona.scan_id == scan_id).all()
    persona_map = {str(p.id): p for p in personas}

    persona_stats = {}
    for r in results:
        qid = str(r.question_id)
        q = db.query(ScanQuestion).filter(ScanQuestion.id == r.question_id).first()
        if not q:
            continue
        pid = str(q.persona_id)
        if pid not in persona_stats:
            persona = persona_map.get(pid)
            persona_stats[pid] = {
                "persona_name": persona.name if persona else "?",
                "topic": None,
                "total": 0,
                "cited": 0,
                "positions": [],
            }
            if persona and persona.topic_id:
                topic = db.query(ScanTopic).filter(ScanTopic.id == persona.topic_id).first()
                if topic:
                    persona_stats[pid]["topic"] = topic.name

        persona_stats[pid]["total"] += 1
        mentioned = (r.brand_analysis or {}).get("marque_cible_mentionnee", False)
        if mentioned:
            persona_stats[pid]["cited"] += 1
            pos = (r.brand_analysis or {}).get("position_marque_cible")
            if pos:
                persona_stats[pid]["positions"].append(pos)

    by_persona = []
    for pid, stats in persona_stats.items():
        rate = round(stats["cited"] / stats["total"] * 100, 1) if stats["total"] else 0
        avg_pos = round(sum(stats["positions"]) / len(stats["positions"]), 1) if stats["positions"] else None
        by_persona.append({
            "persona": stats["persona_name"],
            "topic": stats["topic"],
            "tests": stats["total"],
            "cited": stats["cited"],
            "citation_rate": rate,
            "avg_position": avg_pos,
        })
    by_persona.sort(key=lambda x: -x["citation_rate"])

    # --- Competitors - cross-referenced with Gate 2 classifications ---
    focus_brand_obj = None
    if scan.focus_brand_id:
        focus_brand_obj = db.query(ClientBrand).filter(ClientBrand.id == scan.focus_brand_id).first()
    focus_names_lower = set()
    if focus_brand_obj:
        focus_names_lower.add(focus_brand_obj.name.lower())
        if focus_brand_obj.canonical_name:
            focus_names_lower.add(focus_brand_obj.canonical_name.lower())
        children = db.query(ClientBrand).filter(ClientBrand.parent_id == focus_brand_obj.id).all()
        for child in children:
            focus_names_lower.add(child.name.lower())
            if child.canonical_name:
                focus_names_lower.add(child.canonical_name.lower())

    classification_map = _build_brand_classification_map(scan_id, db)

    classified_competitors = {}
    discovered_brands = {}
    for r in results:
        for bm in (r.brand_mentions or []):
            if not bm.get("est_marque_cible") and bm.get("contexte_valide", True):
                name = bm.get("brand_name_groupby") or bm.get("brand_name", "")
                if not name or name.lower() in focus_names_lower:
                    continue
                cls = _classify_brand_mention(name, classification_map, focus_names_lower)
                mentions = bm.get("nb_mentions") or 1
                if cls == "ignored" or cls == "my_brand":
                    continue
                elif cls == "competitor":
                    classified_competitors[name] = classified_competitors.get(name, 0) + mentions
                else:
                    discovered_brands[name] = discovered_brands.get(name, 0) + mentions

    competitors = [
        {"name": name, "mentions": c, "classification": "competitor"}
        for name, c in sorted(classified_competitors.items(), key=lambda x: -x[1])[:15]
    ]
    discovered = [
        {"name": name, "mentions": c, "classification": "discovered"}
        for name, c in sorted(discovered_brands.items(), key=lambda x: -x[1])[:10]
    ]

    # --- Details (each test) - enriched with brand_mentions, brand_analysis, intention_cachee ---
    topics_map = {str(t.id): t for t in db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).all()}

    # Sprint P (migration 036): intention_cachee is now a native column on
    # scan_questions. Keep the JSONB lookup as fallback for legacy rows whose
    # backfill might have missed them (e.g. edited question text breaking the
    # lookup-by-text join the migration uses).
    intent_lookup = {}  # question_text_lower → intention_cachee (legacy fallback)
    for p in personas:
        for pq in (p.data or {}).get("questions", []):
            if pq.get("question") and pq.get("intention_cachee"):
                intent_lookup[pq["question"].strip().lower()] = pq["intention_cachee"]

    # judgments_by_result_id is now fetched earlier (see top of get_results)
    # so the overview aggregations can use it.

    from services.composite_scores import compute_scores

    details = []
    for r in results:
        q = db.query(ScanQuestion).filter(ScanQuestion.id == r.question_id).first()
        persona = persona_map.get(str(q.persona_id)) if q else None
        topic = topics_map.get(str(persona.topic_id)) if persona and persona.topic_id else None
        intention = None
        if q:
            intention = q.intention_cachee or intent_lookup.get((q.question or "").strip().lower())
        bm_mentioned = (r.brand_analysis or {}).get("marque_cible_mentionnee", False)
        j = judgments_by_result_id.get(str(r.id))
        # Sprint M: composite scores derived from brand_mentions + judgment +
        # intent_category. None on any axis means "not computable" (legacy row,
        # no judgment yet, or not a safety intent for defensive).
        judgment_dict = None
        if j is not None:
            judgment_dict = {
                "positive_signal_hit": j.positive_signal_hit,
                "negative_signal_hit": j.negative_signal_hit,
                "intent_addressed": j.intent_addressed,
                "enveloppement_score": j.enveloppement_score,
            }
        scores = compute_scores(
            brand_mentions=r.brand_mentions,
            judgment=judgment_dict,
            intent_category=(q.intent_category if q else None),
        )
        details.append({
            "question": q.question if q else "?",
            "type": q.type_question if q else "?",
            "intention_cachee": intention,
            "persona": persona.name if persona else "?",
            "persona_id": str(q.persona_id) if q else None,
            "topic_name": topic.name if topic else None,
            "provider": r.provider,
            "model": r.model,
            "brand_mentioned": bm_mentioned,
            "target_cited": r.target_cited,
            "target_position": r.target_position,
            "total_citations": r.total_citations,
            "citations": r.citations or [],
            "brand_mentions": [
                {**bm, "classification": _classify_brand_mention(
                    bm.get("brand_name_groupby") or bm.get("brand_name", ""),
                    classification_map, focus_names_lower
                )}
                for bm in (r.brand_mentions or [])
            ],
            "brand_analysis": r.brand_analysis or {},
            "response_text": r.response_text or "",
            "duration_ms": r.duration_ms,
            # Sprint J judgment payload - None when not yet judged.
            "judgment": None if j is None else {
                "positive_signal_hit": j.positive_signal_hit,
                "positive_signal_evidence": j.positive_signal_evidence or "",
                "negative_signal_hit": j.negative_signal_hit,
                "negative_signal_evidence": j.negative_signal_evidence or "",
                "intent_addressed": j.intent_addressed,
                "intent_evidence": j.intent_evidence or "",
                "citation_quality": j.citation_quality,
                "enveloppement_score": j.enveloppement_score,
            },
            # Sprint M composite scores per response (visibility / quality /
            # defensive / composite). Any field can be None when not computable.
            "scores": scores,
        })

    # Focus brand name
    focus_brand_name = None
    if scan.focus_brand_id:
        fb = db.query(ClientBrand).filter(ClientBrand.id == scan.focus_brand_id).first()
        if fb:
            focus_brand_name = fb.name

    return {
        "overview": {
            "domain": scan.domain,
            "scan_name": scan.name,
            "focus_brand": focus_brand_name,
            "scan_id": str(scan.id),
            "parent_scan_id": str(scan.parent_scan_id) if scan.parent_scan_id else None,
            "total_tests": total,
            "target_cited": brand_mentioned,
            "citation_rate": brand_mention_rate,
            "domain_cited": domain_cited,
            "domain_citation_rate": domain_citation_rate,
            "avg_position": avg_position,
            "providers": list({r.provider for r in results}),
            "scan_date": scan.completed_at.isoformat() if scan.completed_at else None,
            "editorial": (scan.summary or {}).get("editorial"),
            "position_distribution": (scan.summary or {}).get("position_distribution"),
            "position_distribution_delta": (scan.summary or {}).get("position_distribution_delta"),
            "provider_status": (scan.summary or {}).get("provider_status"),
            "refund_info": (scan.summary or {}).get("refund_info"),
            # Sprint M aggregations exposed at scan level for the
            # dashboard. entity_sov keys are entity_type → {total, targets, sov}.
            # judgment_funnel is the 4-issues breakdown across all judged responses.
            "entity_sov": entity_sov,
            "judgment_funnel": judgment_funnel,
            "judgment_coverage": judgment_coverage,
        },
        "by_persona": by_persona,
        "competitors": competitors,
        "discovered_brands": discovered,
        "details": details,
    }


@router.get("/{scan_id}/results/aggregated")
async def get_results_aggregated(
    scan_id: str,
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
    provider: str | None = Query(None),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Aggregated results across multiple runs in a lineage.

    Merges ScanLLMResult data from all completed runs, matched by (question_text, persona_name).
    Returns trend data per metric, per-question evolution, and merged competitors/citations.
    Default: all completed runs. Filter with from_date/to_date (ISO format).
    """
    scan = _check_scan_access(scan_id, user, db)

    # 1. Resolve lineage (root + all children)
    root_id = scan.parent_scan_id or scan.id
    lineage = db.query(Scan).filter(
        ((Scan.id == root_id) | (Scan.parent_scan_id == root_id)),
        Scan.status == "completed",
    ).order_by(Scan.created_at.asc()).all()

    if not lineage:
        return {"mode": "aggregated", "included_runs": [], "overview": None, "by_persona": [], "by_topic": [], "competitors": [], "details": []}

    # 2. Filter by date range
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date)
            lineage = [s for s in lineage if s.completed_at and s.completed_at >= from_dt]
        except ValueError:
            pass
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date)
            lineage = [s for s in lineage if s.completed_at and s.completed_at <= to_dt]
        except ValueError:
            pass

    if not lineage:
        return {"mode": "aggregated", "included_runs": [], "overview": None, "by_persona": [], "by_topic": [], "competitors": [], "details": []}

    included_runs = [
        {"id": str(s.id), "run_index": s.run_index or 0, "completed_at": s.completed_at.isoformat() if s.completed_at else None}
        for s in lineage
    ]
    latest_scan = lineage[-1]
    scan_ids = [s.id for s in lineage]

    # 3. Batch-fetch all LLM results
    q_results = db.query(ScanLLMResult).filter(ScanLLMResult.scan_id.in_(scan_ids))
    if provider and provider != 'all':
        q_results = q_results.filter(ScanLLMResult.provider == provider)
    all_results = q_results.all()

    # Build scan_id → run_index map
    scan_run_map = {s.id: (s.run_index or i) for i, s in enumerate(lineage)}
    scan_date_map = {s.id: s.completed_at for s in lineage}

    # 4. Load personas and questions for matching
    all_personas = {}
    all_questions = {}
    all_topics = {}
    for s in lineage:
        personas = db.query(ScanPersona).filter(ScanPersona.scan_id == s.id).all()
        for p in personas:
            all_personas[str(p.id)] = p
        questions = db.query(ScanQuestion).filter(ScanQuestion.scan_id == s.id).all()
        for q in questions:
            all_questions[str(q.id)] = q
        topics = db.query(ScanTopic).filter(ScanTopic.scan_id == s.id).all()
        for t in topics:
            all_topics[str(t.id)] = t

    # Focus brand (from latest scan)
    focus_brand_name = None
    if latest_scan.focus_brand_id:
        fb = db.query(ClientBrand).filter(ClientBrand.id == latest_scan.focus_brand_id).first()
        if fb:
            focus_brand_name = fb.name

    # Also build focus brand exclusion set for competitors
    focus_names_lower = set()
    if latest_scan.focus_brand_id:
        fb_obj = db.query(ClientBrand).filter(ClientBrand.id == latest_scan.focus_brand_id).first()
        if fb_obj:
            focus_names_lower.add(fb_obj.name.lower())
            if fb_obj.canonical_name:
                focus_names_lower.add(fb_obj.canonical_name.lower())
            children = db.query(ClientBrand).filter(ClientBrand.parent_id == fb_obj.id).all()
            for child in children:
                focus_names_lower.add(child.name.lower())

    # 5. Group results by (question_text, persona_name) across runs
    question_groups = {}  # key: (question_text, persona_name) → { runs: [...] }
    per_run_stats = {s.id: {"total": 0, "brand_mentioned": 0, "domain_cited": 0} for s in lineage}

    for r in all_results:
        q = all_questions.get(str(r.question_id))
        if not q:
            continue
        persona = all_personas.get(str(q.persona_id))
        topic = all_topics.get(str(persona.topic_id)) if persona and persona.topic_id else None
        persona_name = persona.name if persona else "?"
        topic_name = topic.name if topic else None
        q_text = (q.question or "").strip()
        bm_mentioned = (r.brand_analysis or {}).get("marque_cible_mentionnee", False)

        key = (q_text, persona_name)
        if key not in question_groups:
            question_groups[key] = {
                "question": q_text,
                "persona": persona_name,
                "topic_name": topic_name,
                "type": q.type_question,
                "runs": [],
            }

        run_idx = scan_run_map.get(r.scan_id, 0)
        run_date = scan_date_map.get(r.scan_id)
        question_groups[key]["runs"].append({
            "run_index": run_idx,
            "scan_id": str(r.scan_id),
            "completed_at": run_date.isoformat() if run_date else None,
            "brand_mentioned": bm_mentioned,
            "target_cited": r.target_cited,
            "target_position": r.target_position,
            "provider": r.provider,
            "model": r.model,
            "brand_mentions": r.brand_mentions or [],
            "brand_analysis": r.brand_analysis or {},
            "citations": r.citations or [],
            "response_text": r.response_text or "",
            "duration_ms": r.duration_ms,
        })

        # Per-run stats
        if r.scan_id in per_run_stats:
            per_run_stats[r.scan_id]["total"] += 1
            if bm_mentioned:
                per_run_stats[r.scan_id]["brand_mentioned"] += 1
            if r.target_cited:
                per_run_stats[r.scan_id]["domain_cited"] += 1

    # 6. Build overview with trend
    trend = []
    for s in lineage:
        stats = per_run_stats[s.id]
        rate = round(stats["brand_mentioned"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        trend.append({
            "run_index": scan_run_map.get(s.id, 0),
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            "citation_rate": rate,
            "total_tests": stats["total"],
            "brand_mentioned": stats["brand_mentioned"],
        })

    latest_stats = per_run_stats[latest_scan.id]
    latest_total = latest_stats["total"] or 1
    latest_rate = round(latest_stats["brand_mentioned"] / latest_total * 100, 1)
    prev_rate = trend[-2]["citation_rate"] if len(trend) >= 2 else None
    delta = round(latest_rate - prev_rate, 1) if prev_rate is not None else None

    all_rates = [t["citation_rate"] for t in trend]
    avg_rate = round(sum(all_rates) / len(all_rates), 1) if all_rates else 0

    # 7. By persona (aggregated)
    persona_agg = {}  # persona_name → { topic, per_run: { run_idx: {cited, total} } }
    for key, group in question_groups.items():
        pname = group["persona"]
        if pname not in persona_agg:
            persona_agg[pname] = {"topic": group["topic_name"], "per_run": {}}
        for run_data in group["runs"]:
            ri = run_data["run_index"]
            if ri not in persona_agg[pname]["per_run"]:
                persona_agg[pname]["per_run"][ri] = {"cited": 0, "total": 0}
            persona_agg[pname]["per_run"][ri]["total"] += 1
            if run_data["brand_mentioned"]:
                persona_agg[pname]["per_run"][ri]["cited"] += 1

    by_persona = []
    for pname, pagg in persona_agg.items():
        latest_ri = max(pagg["per_run"].keys()) if pagg["per_run"] else 0
        latest_p = pagg["per_run"].get(latest_ri, {"cited": 0, "total": 1})
        p_trend = []
        for ri in sorted(pagg["per_run"].keys()):
            pd = pagg["per_run"][ri]
            p_trend.append({"run_index": ri, "citation_rate": round(pd["cited"] / pd["total"] * 100, 1) if pd["total"] > 0 else 0})
        all_p_rates = [t["citation_rate"] for t in p_trend]
        by_persona.append({
            "persona": pname,
            "topic": pagg["topic"],
            "tests": latest_p["total"],
            "cited": latest_p["cited"],
            "citation_rate": round(latest_p["cited"] / latest_p["total"] * 100, 1) if latest_p["total"] > 0 else 0,
            "avg_citation_rate": round(sum(all_p_rates) / len(all_p_rates), 1) if all_p_rates else 0,
            "trend": p_trend,
        })
    by_persona.sort(key=lambda x: -x["citation_rate"])

    # 8. By topic (aggregated)
    topic_agg = {}
    for key, group in question_groups.items():
        tname = group["topic_name"] or "Other"
        if tname not in topic_agg:
            topic_agg[tname] = {"per_run": {}}
        for run_data in group["runs"]:
            ri = run_data["run_index"]
            if ri not in topic_agg[tname]["per_run"]:
                topic_agg[tname]["per_run"][ri] = {"cited": 0, "total": 0}
            topic_agg[tname]["per_run"][ri]["total"] += 1
            if run_data["brand_mentioned"]:
                topic_agg[tname]["per_run"][ri]["cited"] += 1

    by_topic = []
    for tname, tagg in topic_agg.items():
        latest_ri = max(tagg["per_run"].keys()) if tagg["per_run"] else 0
        latest_t = tagg["per_run"].get(latest_ri, {"cited": 0, "total": 1})
        t_trend = []
        for ri in sorted(tagg["per_run"].keys()):
            td = tagg["per_run"][ri]
            t_trend.append({"run_index": ri, "citation_rate": round(td["cited"] / td["total"] * 100, 1) if td["total"] > 0 else 0})
        all_t_rates = [t["citation_rate"] for t in t_trend]
        by_topic.append({
            "topic": tname,
            "citation_rate": round(latest_t["cited"] / latest_t["total"] * 100, 1) if latest_t["total"] > 0 else 0,
            "avg_citation_rate": round(sum(all_t_rates) / len(all_t_rates), 1) if all_t_rates else 0,
            "trend": t_trend,
        })
    by_topic.sort(key=lambda x: x["citation_rate"])

    # 9. Competitors (merged across all runs, cross-referenced with classifications)
    classification_map = _build_brand_classification_map(str(latest_scan.id), db)
    classified_competitors = {}
    discovered_brands = {}
    for r in all_results:
        for bm in (r.brand_mentions or []):
            if not bm.get("est_marque_cible") and bm.get("contexte_valide", True):
                name = bm.get("brand_name_groupby") or bm.get("brand_name", "")
                if not name or name.lower() in focus_names_lower:
                    continue
                cls = _classify_brand_mention(name, classification_map, focus_names_lower)
                mentions = bm.get("nb_mentions") or 1
                if cls in ("ignored", "my_brand"):
                    continue
                elif cls == "competitor":
                    classified_competitors[name] = classified_competitors.get(name, 0) + mentions
                else:
                    discovered_brands[name] = discovered_brands.get(name, 0) + mentions

    competitors = [
        {"name": name, "mentions": c, "classification": "competitor"}
        for name, c in sorted(classified_competitors.items(), key=lambda x: -x[1])[:15]
    ]
    discovered = [
        {"name": name, "mentions": c, "classification": "discovered"}
        for name, c in sorted(discovered_brands.items(), key=lambda x: -x[1])[:10]
    ]

    # 10. Details (grouped by question, with per-run data)
    details = []
    for key, group in question_groups.items():
        runs_sorted = sorted(group["runs"], key=lambda r: r["run_index"])
        latest_run = runs_sorted[-1] if runs_sorted else None
        ever_mentioned = any(r["brand_mentioned"] for r in runs_sorted)
        mention_count = sum(1 for r in runs_sorted if r["brand_mentioned"])

        # Only include response_text for latest run (performance)
        for r in runs_sorted[:-1]:
            r["response_text"] = ""

        # Enrich brand_mentions with classification
        for r in runs_sorted:
            r["brand_mentions"] = [
                {**bm, "classification": _classify_brand_mention(
                    bm.get("brand_name_groupby") or bm.get("brand_name", ""),
                    classification_map, focus_names_lower
                )}
                for bm in (r.get("brand_mentions") or [])
            ]

        details.append({
            "question": group["question"],
            "persona": group["persona"],
            "topic_name": group["topic_name"],
            "type": group["type"],
            "latest": latest_run,
            "runs": runs_sorted,
            "ever_mentioned": ever_mentioned,
            "mention_count": mention_count,
            "total_runs": len(runs_sorted),
        })

    return {
        "mode": "aggregated",
        "included_runs": included_runs,
        "overview": {
            "domain": latest_scan.domain,
            "scan_name": latest_scan.name,
            "focus_brand": focus_brand_name,
            "total_tests": latest_stats["total"],
            "target_cited": latest_stats["brand_mentioned"],
            "citation_rate": latest_rate,
            "avg_citation_rate": avg_rate,
            "delta": delta,
            "trend": trend,
            "providers": list({r.provider for r in all_results}),
            "scan_date": latest_scan.completed_at.isoformat() if latest_scan.completed_at else None,
            "editorial": (latest_scan.summary or {}).get("editorial"),
            "position_distribution": (latest_scan.summary or {}).get("position_distribution"),
            "position_distribution_delta": (latest_scan.summary or {}).get("position_distribution_delta"),
            "provider_status": (latest_scan.summary or {}).get("provider_status"),
            "refund_info": (latest_scan.summary or {}).get("refund_info"),
        },
        "by_persona": by_persona,
        "by_topic": by_topic,
        "competitors": competitors,
        "discovered_brands": discovered,
        "details": details,
    }


@router.get("/{scan_id}/persona-insights")
async def get_persona_insights(scan_id: str, provider: str | None = Query(None), user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Per-persona deep dive: full profile + visibility + brand perception + competitors + opportunities.

    Returns rich Persona Cards data - the centerpiece of the AI Brand Audit deep dive.
    """
    scan = _check_scan_access(scan_id, user, db)

    q_res = db.query(ScanLLMResult).filter(ScanLLMResult.scan_id == scan_id)
    if provider and provider != 'all':
        q_res = q_res.filter(ScanLLMResult.provider == provider)
    results = q_res.all()
    personas = db.query(ScanPersona).filter(ScanPersona.scan_id == scan_id, ScanPersona.is_active == True).all()
    questions = db.query(ScanQuestion).filter(ScanQuestion.scan_id == scan_id).all()
    topics = {str(t.id): t for t in db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).all()}
    opportunities = db.query(ScanOpportunity).filter(ScanOpportunity.scan_id == scan_id).all()

    # Compute traffic weight per topic (sum of keyword traffic)
    topic_traffic = {}
    kw_rows = db.query(ScanKeyword.topic_id, func.sum(ScanKeyword.traffic)).filter(
        ScanKeyword.scan_id == scan_id, ScanKeyword.topic_id != None,
    ).group_by(ScanKeyword.topic_id).all()
    total_traffic = sum(t or 0 for _, t in kw_rows)
    for tid, traffic in kw_rows:
        topic_traffic[str(tid)] = traffic or 0

    # Build lookups
    q_map = {str(q.id): q for q in questions}
    q_by_persona = {}  # persona_id → [question]
    for q in questions:
        q_by_persona.setdefault(str(q.persona_id), []).append(q)

    # Sprint P (migration 036): intention_cachee is now a column on
    # scan_questions. Lookup kept as fallback for legacy rows that the
    # backfill couldn't match by text.
    intent_lookup = {}
    for p in personas:
        for pq in (p.data or {}).get("questions", []):
            if pq.get("question") and pq.get("intention_cachee"):
                intent_lookup[pq["question"].strip().lower()] = pq["intention_cachee"]

    # Map results by question_id
    results_by_qid = {}
    for r in results:
        results_by_qid.setdefault(str(r.question_id), []).append(r)

    # Map opportunities by persona_name
    opps_by_persona = {}
    for o in opportunities:
        opps_by_persona.setdefault(o.persona_name, []).append(o)

    insights = []
    for persona in personas:
        pid = str(persona.id)
        topic = topics.get(str(persona.topic_id)) if persona.topic_id else None
        persona_questions = q_by_persona.get(pid, [])
        data = persona.data or {}

        # Aggregate visibility per question type
        by_type = {}
        total_tests = 0
        total_cited = 0
        positions = []
        all_brand_mentions = []
        all_competitors = {}

        for q in persona_questions:
            qid = str(q.id)
            q_results = results_by_qid.get(qid, [])
            qtype = q.type_question or "other"
            if qtype not in by_type:
                by_type[qtype] = {"total": 0, "cited": 0}

            for r in q_results:
                total_tests += 1
                by_type[qtype]["total"] += 1
                mentioned = (r.brand_analysis or {}).get("marque_cible_mentionnee", False)
                if mentioned:
                    total_cited += 1
                    by_type[qtype]["cited"] += 1
                    pos = (r.brand_analysis or {}).get("position_marque_cible")
                    if pos:
                        positions.append(pos)
                # Collect brand mentions
                for bm in (r.brand_mentions or []):
                    if bm.get("est_marque_cible"):
                        all_brand_mentions.append(bm)
                # Collect competitor domains
                for domain, count in (r.competitor_domains or {}).items():
                    all_competitors[domain] = all_competitors.get(domain, 0) + count

        citation_rate = round(total_cited / total_tests * 100, 1) if total_tests else 0
        avg_pos = round(sum(positions) / len(positions), 1) if positions else None

        # Brand perception aggregation
        sentiments = [bm.get("sentiment") for bm in all_brand_mentions if bm.get("sentiment")]
        rec_types = [bm.get("type_recommandation") for bm in all_brand_mentions if bm.get("type_recommandation")]
        brand_perception = {
            "times_mentioned": len(all_brand_mentions),
            "dominant_sentiment": max(set(sentiments), key=sentiments.count) if sentiments else None,
            "sentiment_breakdown": {s: sentiments.count(s) for s in set(sentiments)} if sentiments else {},
            "dominant_recommendation": max(set(rec_types), key=rec_types.count) if rec_types else None,
            "recommendation_breakdown": {r: rec_types.count(r) for r in set(rec_types)} if rec_types else {},
            "avg_position": avg_pos,
        }

        # Top competitors for this persona
        persona_competitors = [
            {"name": d, "mentions": c}
            for d, c in sorted(all_competitors.items(), key=lambda x: -x[1])[:5]
        ]

        # Opportunities for this persona
        persona_opps = opps_by_persona.get(persona.name, [])
        opps_serialized = [
            {
                "id": str(o.id),
                "question": q_map.get(str(o.question_id), ScanQuestion()).question if o.question_id else None,
                "priority": o.priority,
                "score": o.opportunity_score,
                "recommended_action": o.recommended_action,
                "best_competitor": o.best_competitor_name,
            }
            for o in sorted(persona_opps, key=lambda o: -(o.opportunity_score or 0))
        ]

        # Questions with results summary
        questions_enriched = []
        for q in persona_questions:
            q_results = results_by_qid.get(str(q.id), [])
            cited_in_any = any((r.brand_analysis or {}).get("marque_cible_mentionnee") for r in q_results)
            competitors_cited = set()
            for r in q_results:
                for domain in (r.competitor_domains or {}):
                    competitors_cited.add(domain)
            questions_enriched.append({
                "id": str(q.id),
                "question": q.question,
                "type": q.type_question,
                "intention_cachee": q.intention_cachee or intent_lookup.get((q.question or "").strip().lower()),
                "is_active": q.is_active,
                "cited": cited_in_any,
                "competitors_cited": list(competitors_cited)[:5],
            })

        # Traffic weight for this persona's topic
        t_traffic = topic_traffic.get(str(persona.topic_id), 0) if persona.topic_id else 0
        t_share = round(t_traffic / total_traffic * 100, 1) if total_traffic > 0 else 0

        insights.append({
            "persona_id": pid,
            "name": persona.name,
            "topic_name": topic.name if topic else None,
            "weight": {
                "topic_traffic": t_traffic,
                "traffic_share": t_share,
                "keyword_count": topic.keyword_count if topic else 0,
                "total_searchvolume": data.get("metriques", {}).get("total_searchvolume_segment", 0),
                "avg_position": data.get("metriques", {}).get("avg_position_segment"),
                "ranking_score": data.get("metriques", {}).get("ranking_score_segment"),
            },
            "profile": {
                "age": data.get("profil_demographique", {}).get("age"),
                "profession": data.get("profil_demographique", {}).get("situation_professionnelle"),
                "expertise": data.get("profil_demographique", {}).get("niveau_expertise"),
                "pain_points": data.get("points_douleur", []),
                "customer_journey": data.get("parcours_type"),
                "search_intents": data.get("intentions_recherche", []),
                "content_opportunities": data.get("opportunites", []),
                "keywords": data.get("mots_cles_associes", []),
            },
            "visibility": {
                "total_questions": total_tests,
                "cited": total_cited,
                "citation_rate": citation_rate,
                "avg_position": avg_pos,
                "by_question_type": by_type,
            },
            "brand_perception": brand_perception,
            "competitors": persona_competitors,
            "opportunities": opps_serialized,
            "questions": questions_enriched,
        })

    # Sort: worst visibility first (most actionable)
    insights.sort(key=lambda x: x["visibility"]["citation_rate"])
    return insights


@router.get("/{scan_id}/opportunities")
async def get_opportunities(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """All scored opportunities for a scan, grouped by priority.

    Each opportunity = 1 question where the brand is weak + recommended action.
    """
    scan = _check_scan_access(scan_id, user, db)

    opportunities = db.query(ScanOpportunity).filter(
        ScanOpportunity.scan_id == scan_id,
    ).order_by(ScanOpportunity.opportunity_score.desc()).all()

    # Enrich with question text + brand classification
    q_ids = [o.question_id for o in opportunities if o.question_id]
    questions = {str(q.id): q for q in db.query(ScanQuestion).filter(ScanQuestion.id.in_(q_ids)).all()} if q_ids else {}
    classification_map = _build_brand_classification_map(scan_id, db)

    summary = {"critique": 0, "haute": 0, "moyenne": 0}
    items = []
    for o in opportunities:
        summary[o.priority] = summary.get(o.priority, 0) + 1
        q = questions.get(str(o.question_id))
        best_comp_cls = classification_map.get((o.best_competitor_name or "").lower(), "discovered") if o.best_competitor_name else None
        items.append({
            "id": str(o.id),
            "question": q.question if q else None,
            "question_type": q.type_question if q else None,
            "persona_name": o.persona_name,
            "topic_name": o.topic_name,
            "priority": o.priority,
            "score": o.opportunity_score,
            "brand_cited": o.brand_cited,
            "brand_position": o.brand_position,
            "brand_sentiment": o.brand_sentiment,
            "brand_recommended": o.brand_recommended,
            "best_competitor_name": o.best_competitor_name,
            "best_competitor_classification": best_comp_cls,
            "best_competitor_position": o.best_competitor_position,
            "best_competitor_domain": o.best_competitor_domain,
            "nb_competitors_cited": o.nb_competitors_cited,
            "recommended_action": o.recommended_action,
            "target_url": o.target_url,
            "media_domain": o.media_domain,
        })

    return {"summary": summary, "opportunities": items}


# --- Sprint 4 : Wikipedia Entity Action -------------------------------------
# Surfaces the Wikipedia presence cache (client_brands.wikipedia JSONB,
# migration 046) for all brands attached to the scan : focus + competitors +
# my_brand children. ChatGPT cites Wikipedia ~48% of the time so a missing
# page is structurally costly. Cf. project_10_action_features.md #1.

@router.get("/{scan_id}/wikipedia")
async def get_scan_wikipedia(
    scan_id: str,
    include_all: bool = False,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return Wikipedia presence per brand attached to this scan.

    Response shape :
        {
            "scan_id": "...",
            "brands": [
                {
                    "brand_id": "...",
                    "name": "Ducray",
                    "canonical_name": "ducray",
                    "classification": "target_brand",
                    "is_focus": True,
                    "wikipedia": { "checked_at": ..., "by_lang": {...} },
                    "stale": False  # true if older than the TTL or never checked
                },
                ...
            ],
            "last_refresh": "..."  # max(checked_at) across brands, or null
        }

    Always returns 200, even when wikipedia={} for every brand - the UI
    handles the empty state via a "Refresh" button that POSTs to the
    /refresh endpoint below.
    """
    scan = _check_scan_access(scan_id, user, db)

    # Sprint 4.7 - surface only the brands worth auditing on Wikipedia :
    # the focus brand of the scan + classified MASTER brands (parent_id NULL).
    # Sub-products / gammes (parent_id NOT NULL) are excluded because they
    # virtually never have a dedicated Wikipedia page and they generate
    # constant name-collision false positives.
    # include_all=true returns every brand attached to the scan (debug view).
    DEFAULT_CLASSIFICATIONS = ("my_brand", "competitor")
    from sqlalchemy import or_ as _or_
    q = (
        db.query(ScanBrandClassification, ClientBrand)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(ScanBrandClassification.scan_id == scan_id)
    )
    if not include_all:
        q = q.filter(
            ScanBrandClassification.classification.in_(DEFAULT_CLASSIFICATIONS)
        ).filter(
            _or_(
                ScanBrandClassification.is_focus.is_(True),
                ClientBrand.parent_id.is_(None),
            )
        )
    sbc_rows = q.all()

    from datetime import datetime as _dt, timedelta as _td
    TTL_DAYS = 7
    now = _dt.utcnow()
    items = []
    last_refresh = None
    for sbc, brand in sbc_rows:
        wiki = brand.wikipedia or {}
        checked = wiki.get("checked_at")
        stale = True
        if checked:
            try:
                ts = _dt.fromisoformat(str(checked).rstrip("Z"))
                stale = ts < (now - _td(days=TTL_DAYS))
                if not last_refresh or ts > last_refresh:
                    last_refresh = ts
            except ValueError:
                stale = True
        items.append({
            "brand_id": str(brand.id),
            "name": brand.name,
            "canonical_name": brand.canonical_name,
            "domain": brand.domain,
            "classification": sbc.classification,
            "is_focus": bool(sbc.is_focus),
            "wikipedia": wiki,
            "stale": stale,
        })

    # Sort : focus first, then my_brand → competitor → unclassified → ignored.
    classif_order = {"my_brand": 0, "competitor": 1, "unclassified": 5, "ignored": 9}
    items.sort(key=lambda i: (
        0 if i["is_focus"] else 1,
        classif_order.get(i["classification"], 6),
        (i["name"] or "").lower(),
    ))

    return {
        "scan_id": scan_id,
        "brands": items,
        "last_refresh": last_refresh.isoformat() + "Z" if last_refresh else None,
        "ttl_days": TTL_DAYS,
    }


@router.post("/{scan_id}/wikipedia/refresh")
async def refresh_scan_wikipedia(
    scan_id: str,
    force: bool = False,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue a check_brand_wikipedia job for this scan.

    No-cost (Wikipedia REST + Action API are free). If `force=true`, ignore the
    7-day TTL and re-check every brand. Idempotent at the job level -
    re-clicking enqueues a fresh job, but the handler itself is no-op for
    fresh entries unless force=true.
    """
    scan = _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "check_brand_wikipedia", {"force": bool(force)})
    db.commit()
    return {"status": "enqueued", "scan_id": scan_id, "force": bool(force)}


# --- Sprint 5 : Princeton GEO content audit -------------------------------
# Audit each URL of the user's own site that was cited by at least one LLM
# during the scan, scoring it against the 7 patterns from Aggarwal et al.
# (KDD 24). Heuristic, deterministic, no LLM cost. Cf.
# project_10_action_features.md #4 + worker/handlers/audit_scan_pages.py.

@router.get("/{scan_id}/page-audit")
async def get_scan_page_audit(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the audited pages for a scan, sorted by GEO score ASC (worst
    first) so the action surface highlights the highest-leverage fixes."""
    scan = _check_scan_access(scan_id, user, db)

    rows = (
        db.query(ScanPageAudit)
        .filter(ScanPageAudit.scan_id == scan_id)
        .order_by(
            ScanPageAudit.fetch_error.is_(None).desc(),   # successful audits first
            ScanPageAudit.geo_score.asc().nullslast(),     # worst score first inside successes
            ScanPageAudit.citation_count.desc(),
        )
        .all()
    )

    # Aggregate KPIs for the hero banner.
    success_rows = [r for r in rows if r.fetch_error is None and r.geo_score is not None]
    avg_score = round(sum(r.geo_score for r in success_rows) / len(success_rows), 1) if success_rows else None
    top_issues: dict[str, int] = {}
    for r in success_rows:
        for issue in (r.audit or {}).get("issues", []):
            top_issues[issue["pattern"]] = top_issues.get(issue["pattern"], 0) + 1

    items = []
    last_fetched = None
    for r in rows:
        if r.fetched_at and (last_fetched is None or r.fetched_at > last_fetched):
            last_fetched = r.fetched_at
        items.append({
            "url": r.url,
            "title": r.title,
            "lang": r.lang,
            "fetch_status": r.fetch_status,
            "fetch_error": r.fetch_error,
            "geo_score": r.geo_score,
            "citation_count": r.citation_count,
            "fetched_at": r.fetched_at.isoformat() + "Z" if r.fetched_at else None,
            "signals": (r.audit or {}).get("signals", {}),
            "scores": (r.audit or {}).get("scores", {}),
            "issues": (r.audit or {}).get("issues", []),
        })

    return {
        "scan_id": scan_id,
        "pages": items,
        "summary": {
            "audited": len(rows),
            "succeeded": len(success_rows),
            "failed": len(rows) - len(success_rows),
            "avg_geo_score": avg_score,
            "issues_by_pattern": top_issues,
        },
        "last_fetched": last_fetched.isoformat() + "Z" if last_fetched else None,
    }


@router.post("/{scan_id}/page-audit/refresh")
async def refresh_scan_page_audit(
    scan_id: str,
    reset: bool = False,
    limit: int = 400,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue an `audit_scan_pages` worker job. Free (no LLM). If `reset=true`,
    the handler wipes prior audit rows before re-running so a stale page that
    has since 404'd doesn't linger in the UI."""
    scan = _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "audit_scan_pages", {
        "reset": bool(reset),
        "limit": int(limit) if limit else 400,
    })
    db.commit()
    return {"status": "enqueued", "scan_id": scan_id, "reset": bool(reset), "limit": limit}


# --- Sprint 6 : schema.org / JSON-LD audit + generator --------------------
# Same source URLs as Sprint 5 (the user's own pages cited by an LLM during
# the scan). For each one we extract existing JSON-LD blocks, detect the
# page type, generate the missing schema.org blocks, validate them locally
# against the schema.org required-property spec. No LLM cost. Cf.
# project_10_action_features.md #5 + worker/handlers/audit_scan_schemas.py.

@router.get("/{scan_id}/schema-audit")
async def get_scan_schema_audit(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the schema audit per cited URL, sorted by schema_score ASC
    (worst first) so the user attacks the highest-leverage fixes first."""
    scan = _check_scan_access(scan_id, user, db)

    rows = (
        db.query(ScanSchemaAudit)
        .filter(ScanSchemaAudit.scan_id == scan_id)
        .order_by(
            ScanSchemaAudit.fetch_error.is_(None).desc(),
            ScanSchemaAudit.schema_score.asc().nullslast(),
            ScanSchemaAudit.citation_count.desc(),
        )
        .all()
    )

    success_rows = [r for r in rows if r.fetch_error is None and r.schema_score is not None]
    avg_score = round(sum(r.schema_score for r in success_rows) / len(success_rows), 1) if success_rows else None

    # Tally missing schemas across the scan so the hero banner can surface
    # the biggest opportunity ("FAQPage missing on 14 pages").
    missing_tally: dict[str, int] = {}
    for r in success_rows:
        for t in (r.missing_schemas or []):
            missing_tally[t] = missing_tally.get(t, 0) + 1

    by_page_type: dict[str, int] = {}
    for r in success_rows:
        if r.page_type:
            by_page_type[r.page_type] = by_page_type.get(r.page_type, 0) + 1

    items = []
    last_fetched = None
    for r in rows:
        if r.fetched_at and (last_fetched is None or r.fetched_at > last_fetched):
            last_fetched = r.fetched_at
        items.append({
            "url": r.url,
            "title": r.title,
            "page_type": r.page_type,
            "fetch_status": r.fetch_status,
            "fetch_error": r.fetch_error,
            "schema_score": r.schema_score,
            "citation_count": r.citation_count,
            "fetched_at": r.fetched_at.isoformat() + "Z" if r.fetched_at else None,
            "existing_schemas": r.existing_schemas or [],
            "missing_schemas": list(r.missing_schemas or []),
            "generated_blocks": r.generated_blocks or {},
        })

    return {
        "scan_id": scan_id,
        "pages": items,
        "summary": {
            "audited": len(rows),
            "succeeded": len(success_rows),
            "failed": len(rows) - len(success_rows),
            "avg_schema_score": avg_score,
            "missing_by_type": missing_tally,
            "by_page_type": by_page_type,
        },
        "last_fetched": last_fetched.isoformat() + "Z" if last_fetched else None,
    }


@router.post("/{scan_id}/schema-audit/refresh")
async def refresh_scan_schema_audit(
    scan_id: str,
    reset: bool = False,
    limit: int = 400,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue an `audit_scan_schemas` worker job. Free (no LLM)."""
    scan = _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "audit_scan_schemas", {
        "reset": bool(reset),
        "limit": int(limit) if limit else 400,
    })
    db.commit()
    return {"status": "enqueued", "scan_id": scan_id, "reset": bool(reset), "limit": limit}


# --- Sprint 7 : competitor reverse-engineering ----------------------------
# For each scan we surface the top 5 competitors by win-count then audit
# the pages LLMs already cite for them (Princeton GEO + JSON-LD schemas +
# Babbar backlinks). The API aggregates per-competitor and computes a
# "pattern delta" vs the user's own pages (scan_page_audits) so the UI
# can render "what they have that you don't." Cf.
# project_10_action_features.md #6 + worker/handlers/audit_competitor_pages.py.

@router.get("/{scan_id}/competitor-reverse")
async def get_competitor_reverse(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return per-competitor cards : the brand, its winning URLs with GEO
    score / schemas / backlinks, and a pattern delta vs the user's own
    pages on the same scan (from scan_page_audits + scan_schema_audits)."""
    scan = _check_scan_access(scan_id, user, db)

    rows = (
        db.query(ScanCompetitorPage, ClientBrand)
        .join(ClientBrand, ClientBrand.id == ScanCompetitorPage.brand_id)
        .filter(ScanCompetitorPage.scan_id == scan_id)
        .order_by(
            ScanCompetitorPage.fetch_error.is_(None).desc(),
            ScanCompetitorPage.geo_score.desc().nullslast(),
            ScanCompetitorPage.citation_count.desc(),
        )
        .all()
    )

    # ---- Baseline : the user's own scan_page_audits + scan_schema_audits.
    own_pages = (
        db.query(ScanPageAudit).filter(ScanPageAudit.scan_id == scan_id).all()
    )
    own_schemas = (
        db.query(ScanSchemaAudit).filter(ScanSchemaAudit.scan_id == scan_id).all()
    )
    own_geo_avg = None
    if own_pages:
        scored = [p.geo_score for p in own_pages if p.geo_score is not None]
        own_geo_avg = round(sum(scored) / len(scored), 1) if scored else None
    own_schema_avg = None
    if own_schemas:
        scored = [s.schema_score for s in own_schemas if s.schema_score is not None]
        own_schema_avg = round(sum(scored) / len(scored), 1) if scored else None
    own_schema_types: dict[str, int] = {}
    for s in own_schemas:
        for b in (s.existing_schemas or []):
            if b.get("valid") and b.get("type"):
                own_schema_types[b["type"]] = own_schema_types.get(b["type"], 0) + 1

    # ---- Group competitor pages by brand and compute aggregates.
    by_brand: dict[str, dict] = {}
    last_fetched = None
    for row, brand in rows:
        if row.fetched_at and (last_fetched is None or row.fetched_at > last_fetched):
            last_fetched = row.fetched_at
        bid = str(brand.id)
        bucket = by_brand.get(bid)
        if bucket is None:
            bucket = {
                "brand_id": bid,
                "brand_name": brand.name,
                "domain": brand.domain,
                "wikipedia": (brand.wikipedia or {}).get("found") or (brand.wikipedia or {}).get("title"),
                "pages": [],
                "backlinks": row.backlinks or {},
                "_geo_scores": [],
                "_schema_scores": [],
                "_schema_types": {},
            }
            by_brand[bid] = bucket

        bucket["pages"].append({
            "url": row.url,
            "title": row.title,
            "fetch_status": row.fetch_status,
            "fetch_error": row.fetch_error,
            "geo_score": row.geo_score,
            "schema_score": row.schema_score,
            "citation_count": row.citation_count,
            "winning_questions": row.winning_questions or [],
            "signals": (row.geo_audit or {}).get("signals", {}),
            "scores": (row.geo_audit or {}).get("scores", {}),
            "issues": (row.geo_audit or {}).get("issues", []),
            "audit_source": (row.geo_audit or {}).get("source"),  # "page" | "contexte" | null
            "schemas": row.schemas or [],
            "fetched_at": row.fetched_at.isoformat() + "Z" if row.fetched_at else None,
        })
        if row.geo_score is not None:
            bucket["_geo_scores"].append(row.geo_score)
        if row.schema_score is not None:
            bucket["_schema_scores"].append(row.schema_score)
        for b in (row.schemas or []):
            if b.get("valid") and b.get("type"):
                bucket["_schema_types"][b["type"]] = bucket["_schema_types"].get(b["type"], 0) + 1

    competitors = []
    for bucket in by_brand.values():
        geo_avg = (round(sum(bucket["_geo_scores"]) / len(bucket["_geo_scores"]), 1)
                   if bucket["_geo_scores"] else None)
        schema_avg = (round(sum(bucket["_schema_scores"]) / len(bucket["_schema_scores"]), 1)
                      if bucket["_schema_scores"] else None)
        # Pattern delta : schema types the competitor has at least once AND
        # the user doesn't have on any of their own pages.
        their_types = set(bucket["_schema_types"].keys())
        my_types = set(own_schema_types.keys())
        schemas_they_have_you_dont = sorted(list(their_types - my_types))

        # Aggregate winning_questions across this competitor's pages, dedupe by question_id.
        seen_qids: set[str] = set()
        winning_questions: list[dict] = []
        for p in bucket["pages"]:
            for q in p["winning_questions"]:
                qid = q.get("question_id")
                if not qid or qid in seen_qids:
                    continue
                seen_qids.add(qid)
                winning_questions.append(q)

        competitors.append({
            "brand_id": bucket["brand_id"],
            "brand_name": bucket["brand_name"],
            "domain": bucket["domain"],
            "wikipedia_present": bool(bucket["wikipedia"]),
            "pages": bucket["pages"],
            "page_count": len(bucket["pages"]),
            "winning_questions": winning_questions,
            "winning_questions_count": len(winning_questions),
            "geo_avg": geo_avg,
            "schema_avg": schema_avg,
            "schema_types": bucket["_schema_types"],
            "backlinks": bucket["backlinks"],
            "delta": {
                "geo_vs_yours": (round(geo_avg - own_geo_avg, 1) if (geo_avg is not None and own_geo_avg is not None) else None),
                "schema_vs_yours": (round(schema_avg - own_schema_avg, 1) if (schema_avg is not None and own_schema_avg is not None) else None),
                "schemas_they_have_you_dont": schemas_they_have_you_dont,
            },
        })

    # Sort competitors : most winning_questions first, then geo_avg desc.
    competitors.sort(key=lambda c: (-c["winning_questions_count"], -(c["geo_avg"] or 0)))

    return {
        "scan_id": scan_id,
        "competitors": competitors,
        "baseline": {
            "own_geo_avg": own_geo_avg,
            "own_schema_avg": own_schema_avg,
            "own_schema_types": own_schema_types,
            "own_page_count": len(own_pages),
        },
        "summary": {
            "competitors_audited": len(competitors),
            "pages_audited": sum(c["page_count"] for c in competitors),
        },
        "last_fetched": last_fetched.isoformat() + "Z" if last_fetched else None,
    }


@router.get("/{scan_id}/llm-result/{slr_id}")
async def get_llm_result(
    scan_id: str,
    slr_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the full LLM response text + the question that prompted it.
    Used by the Competitors tab's "View full response" modal so the user
    can read the surrounding context, not just the 200-char citation
    snippet. Light-weight read - no LLM call."""
    _check_scan_access(scan_id, user, db)

    row = (
        db.query(ScanLLMResult)
        .filter(ScanLLMResult.id == slr_id, ScanLLMResult.scan_id == scan_id)
        .first()
    )
    if not row:
        from fastapi import HTTPException
        raise HTTPException(404, "LLM result not found for this scan")

    question_text = None
    if row.question_id:
        q = db.query(ScanQuestion).filter(ScanQuestion.id == row.question_id).first()
        if q:
            question_text = q.question

    return {
        "slr_id": str(row.id),
        "scan_id": str(row.scan_id),
        "provider": row.provider,
        "model": row.model,
        "question_id": str(row.question_id) if row.question_id else None,
        "question": question_text,
        "response_text": row.response_text or "",
        "citations": row.citations or [],
    }


# --- Sprint 8 : Reddit opportunity finder ---------------------------------
# For each scan we surface the Reddit threads LLMs already cite, classify
# them (competitor_wins / you_win / neutral), score sentiment via Haiku
# per thread, and rank by a leverage_score that points the user at the
# highest-opportunity conversations. Cf.
# project_10_action_features.md #3 + worker/handlers/audit_reddit_threads.py.

def _reddit_action(classification: str | None, sentiment: str | None) -> dict:
    """Recommended action per row, derived deterministically from
    (classification, sentiment). Kept in the API layer (not stored) so
    the formula can evolve without a DB migration. Mirror of the
    handler's _recommended_action function (migration 052)."""
    cls = classification or "neutral"
    sent = sentiment or "unclear"
    if cls == "you_lost":
        return {"label": "Defend your position", "tone": "urgent"}
    if cls == "shared_crisis":
        return {"label": "Crisis response", "tone": "urgent"}
    if cls == "competitor_wins":
        if sent == "negative":
            return {"label": "Engage now", "tone": "urgent"}
        if sent == "mixed":
            return {"label": "Engage thoughtfully", "tone": "high"}
        if sent in ("neutral", "unclear"):
            return {"label": "Add your perspective", "tone": "medium"}
        return {"label": "Skip - they win", "tone": "low"}
    if cls == "head_to_head":
        return {"label": "Investigate", "tone": "high"}
    if cls == "shared_win":
        return {"label": "Co-consideration", "tone": "positive"}
    if cls == "you_win_strong":
        return {"label": "Amplify", "tone": "positive"}
    if cls == "you_win":
        if sent == "negative":
            return {"label": "Monitor crisis", "tone": "urgent"}
        return {"label": "Keep monitoring", "tone": "positive"}
    return {"label": "Context only", "tone": "low"}

@router.get("/{scan_id}/reddit-opportunities")
async def get_reddit_opportunities(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return audited Reddit threads for this scan, sorted by leverage_score
    DESC (highest opportunity first). Honest fetch errors surfaced so the
    UI can chip them as 'fetch failed'."""
    _check_scan_access(scan_id, user, db)

    rows = (
        db.query(ScanRedditThread)
        .filter(ScanRedditThread.scan_id == scan_id)
        .order_by(
            ScanRedditThread.leverage_score.desc().nullslast(),
            ScanRedditThread.citation_count.desc(),
        )
        .all()
    )

    by_class: dict[str, int] = {}
    by_sentiment: dict[str, int] = {}
    last_fetched = None
    items = []
    for r in rows:
        if r.fetched_at and (last_fetched is None or r.fetched_at > last_fetched):
            last_fetched = r.fetched_at
        if r.classification:
            by_class[r.classification] = by_class.get(r.classification, 0) + 1
        if r.sentiment:
            by_sentiment[r.sentiment] = by_sentiment.get(r.sentiment, 0) + 1
        items.append({
            "url": r.url,
            "subreddit": r.subreddit,
            "title": r.title,
            "author": r.author,
            "score": r.score,
            "num_comments": r.num_comments,
            "posted_at": r.posted_at.isoformat() + "Z" if r.posted_at else None,
            "fetch_status": r.fetch_status,
            "fetch_error": r.fetch_error,
            "citation_count": r.citation_count,
            "target_mentioned": r.target_mentioned,
            "competitors_mentioned": list(r.competitors_mentioned or []),
            "classification": r.classification,
            "sentiment": r.sentiment,
            "sentiment_summary": r.sentiment_summary,
            "target_sentiment": r.target_sentiment,
            "competitor_sentiment": r.competitor_sentiment,
            "body_excerpt": r.body_excerpt,
            "top_comments": r.top_comments or [],
            "winning_questions": r.winning_questions or [],
            "leverage_score": r.leverage_score,
            "recommended_action": _reddit_action(r.classification, r.sentiment),
        })

    # Per-classification counts for the filter chips + Peak-End hero card.
    # "Opportunities" aggregates the action-worthy buckets so the user has
    # one number to look at first.
    opp_buckets = ("competitor_wins", "you_lost", "shared_crisis", "head_to_head")
    opportunities_total = sum(by_class.get(b, 0) for b in opp_buckets)

    # Focus brand name for the UI to pass to the URL highlighter when
    # rendering inline LLM responses. Read from the scan's focus_brand
    # relationship ; fall back to the scan name.
    scan_row = db.query(Scan).filter(Scan.id == scan_id).first()
    focus_brand_name = ""
    if scan_row:
        if getattr(scan_row, "focus_brand_name", None):
            focus_brand_name = scan_row.focus_brand_name or ""
        elif getattr(scan_row, "focus_brand", None) and getattr(scan_row.focus_brand, "name", None):
            focus_brand_name = scan_row.focus_brand.name or ""

    return {
        "scan_id": scan_id,
        "focus_brand_name": focus_brand_name,
        "threads": items,
        "summary": {
            "total": len(rows),
            "by_classification": by_class,
            "by_sentiment": by_sentiment,
            "competitor_wins": by_class.get("competitor_wins", 0),
            "you_lost": by_class.get("you_lost", 0),
            "shared_crisis": by_class.get("shared_crisis", 0),
            "head_to_head": by_class.get("head_to_head", 0),
            "you_win": by_class.get("you_win", 0) + by_class.get("you_win_strong", 0),
            "shared_win": by_class.get("shared_win", 0),
            "opportunities": opportunities_total,
        },
        "last_fetched": last_fetched.isoformat() + "Z" if last_fetched else None,
    }


@router.post("/{scan_id}/reddit-opportunities/refresh")
async def refresh_reddit_opportunities(
    scan_id: str,
    reset: bool = False,
    limit: int = 100,
    sentiment: bool = True,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue an `audit_reddit_threads` worker job. Default 100 threads,
    sentiment ON. LLM cost capped via the per-thread Haiku call (~$0.001
    each) and the thread count cap."""
    _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "audit_reddit_threads", {
        "reset": bool(reset),
        "limit": int(limit),
        "sentiment": bool(sentiment),
    })
    db.commit()
    return {
        "status": "enqueued", "scan_id": scan_id,
        "reset": bool(reset), "limit": limit, "sentiment": bool(sentiment),
    }


@router.post("/{scan_id}/competitor-reverse/refresh")
async def refresh_competitor_reverse(
    scan_id: str,
    reset: bool = False,
    competitors: int = 5,
    urls_per_competitor: int = 10,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue an `audit_competitor_pages` worker job. Free (no LLM).
    Defaults to 5 competitors × 10 URLs = 50 fetches, ~25 s on the wire."""
    scan = _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "audit_competitor_pages", {
        "reset": bool(reset),
        "competitors": int(competitors),
        "urls_per_competitor": int(urls_per_competitor),
    })
    db.commit()
    return {
        "status": "enqueued",
        "scan_id": scan_id,
        "reset": bool(reset),
        "competitors": competitors,
        "urls_per_competitor": urls_per_competitor,
    }


# --- Sprint 9 : PR / journalist outreach list ----------------------------
# For each scan we surface the media domains LLMs cite for competitors but
# not the focus brand (or both, = lost ground). Authority signals copied
# from media_catalog when the domain is enriched. Cf.
# project_10_action_features.md #7 + worker/handlers/build_pr_outreach.py.

def _pr_action(classification: str | None, in_catalog: bool, target_cited: bool) -> dict:
    """Recommended action chip per row, derived from the classification +
    whether we have an enriched media_catalog row yet. Kept in the API
    layer so the formula can evolve without a DB migration."""
    cls = classification or "target_only"
    if cls == "competitor_only":
        return {"label": "Pitch them", "tone": "high"}
    if cls == "shared":
        return {"label": "Defend visibility", "tone": "medium"}
    if cls == "target_only" and target_cited:
        return {"label": "Maintain relationship", "tone": "positive"}
    return {"label": "Context only", "tone": "low"}


@router.get("/{scan_id}/pr-outreach")
async def get_pr_outreach(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return media domains the LLMs already cite for this scan's
    competitors or focus brand, sorted by leverage_score DESC. Excludes
    forums, encyclopedias, e-commerce and brand-own sites - those are
    covered by other tabs (Reddit / Wikipedia / Competitors)."""
    _check_scan_access(scan_id, user, db)

    rows = (
        db.query(ScanPROutreach)
        .filter(ScanPROutreach.scan_id == scan_id)
        .order_by(
            ScanPROutreach.leverage_score.desc().nullslast(),
            ScanPROutreach.citation_count.desc(),
        )
        .all()
    )

    by_class: dict[str, int] = {}
    by_site_type: dict[str, int] = {}
    last_seen = None
    items = []
    for r in rows:
        if r.created_at and (last_seen is None or r.created_at > last_seen):
            last_seen = r.created_at
        if r.classification:
            by_class[r.classification] = by_class.get(r.classification, 0) + 1
        if r.site_type:
            by_site_type[r.site_type] = by_site_type.get(r.site_type, 0) + 1
        items.append({
            "domain": r.domain,
            "site_type": r.site_type,
            "citation_count": r.citation_count,
            "competitor_brands": list(r.competitor_brands or []),
            "target_cited": r.target_cited,
            "classification": r.classification,
            "top_pages": r.top_pages or [],
            "winning_questions": r.winning_questions or [],
            "da": r.da, "tf": r.tf, "cf": r.cf, "rd": r.rd,
            "price_eur": float(r.price_eur) if r.price_eur is not None else None,
            "vertical": list(r.vertical or []),
            "audience_tags": list(r.audience_tags or []),
            "editorial_voice": r.editorial_voice,
            "in_catalog": r.in_catalog,
            "leverage_score": r.leverage_score,
            "recommended_action": _pr_action(r.classification, r.in_catalog, r.target_cited),
        })

    opportunities = by_class.get("competitor_only", 0) + by_class.get("shared", 0)
    return {
        "scan_id": scan_id,
        "domains": items,
        "summary": {
            "total": len(rows),
            "by_classification": by_class,
            "by_site_type": by_site_type,
            "competitor_only": by_class.get("competitor_only", 0),
            "shared": by_class.get("shared", 0),
            "target_only": by_class.get("target_only", 0),
            "opportunities": opportunities,
            "enriched": sum(1 for r in rows if r.in_catalog),
        },
        "last_fetched": last_seen.isoformat() + "Z" if last_seen else None,
    }


@router.post("/{scan_id}/pr-outreach/refresh")
async def refresh_pr_outreach(
    scan_id: str,
    reset: bool = False,
    limit: int = 200,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue a `build_pr_outreach` worker job. Free (no LLM) - the
    handler is a pure aggregation over scan_llm_results + media_catalog.
    Typically completes in <2s on a 100-question scan."""
    _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "build_pr_outreach", {
        "reset": bool(reset),
        "limit": int(limit),
    })
    db.commit()
    return {"status": "enqueued", "scan_id": scan_id, "reset": bool(reset), "limit": limit}


# --- Sprint 11 : internal linking audit ---------------------------------
# Per-page audit of the internal link graph of the user's own pages cited
# by LLMs in this scan. Topology stats (orphans / hubs / dead-ends) are
# computed at read time from the persisted outbound_internal_links arrays.
# Cf. project_10_action_features.md #9 + worker/handlers/audit_internal_links.py.

def _internal_link_action(
    linking_score: int | None,
    internal_count: int,
    generic_ratio: float,
    topology: str = "healthy",
) -> dict:
    """Recommended action per row. Free formula - tweak without migration.

    Topology-aware (Sprint 11.1 fix) : orphan + dead-end override the
    linking_score read, otherwise an orphan page with many healthy
    outbound links shows up as "Looking healthy" - which contradicts the
    ORPHAN status chip rendered right next to it on the row.
    """
    if linking_score is None:
        return {"label": "-", "tone": "low"}
    if topology == "dead_end" or internal_count == 0:
        return {"label": "Add internal links", "tone": "urgent"}
    if topology == "orphan":
        return {"label": "Add inbound links", "tone": "high"}
    if generic_ratio > 0.5:
        return {"label": "Rewrite anchors", "tone": "high"}
    if linking_score < 50:
        return {"label": "Polish anchors", "tone": "medium"}
    return {"label": "Looking healthy", "tone": "positive"}


@router.get("/{scan_id}/internal-linking")
async def get_internal_linking(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return per-page internal linking audit + topology stats. Topology
    (orphan / hub / dead-end) is computed on the fly from the inbound link
    counts across the scan's audited URL set."""
    _check_scan_access(scan_id, user, db)

    rows = (
        db.query(ScanInternalLink)
        .filter(ScanInternalLink.scan_id == scan_id)
        .order_by(
            ScanInternalLink.linking_score.asc().nullslast(),
            ScanInternalLink.citation_count.desc(),
        )
        .all()
    )

    def _canon(u: str | None) -> str:
        """Sprint 11.1 fix : strip query strings + fragments before
        comparing. Without this, LLM-cited URLs that carry tracking
        params (e.g. `?utm_source=openai` added by ChatGPT to the URLs
        it surfaces) are treated as distinct from the same page linked
        without params elsewhere on the site → false orphans (9/30 on
        the initial Avène smoke ; 4 of those resolved after this fix).
        """
        if not u:
            return ""
        # Cut anything starting at '?' or '#'.
        base = u
        for sep in ("#", "?"):
            i = base.find(sep)
            if i >= 0:
                base = base[:i]
        return base.rstrip("/").lower()

    audited_urls = {_canon(r.url) for r in rows}
    inbound: dict[str, int] = {u: 0 for u in audited_urls}
    inbound_sources: dict[str, set[str]] = {u: set() for u in audited_urls}
    for r in rows:
        src_canon = _canon(r.url)
        for link in (r.outbound_internal_links or []):
            tc = _canon(link.get("target"))
            if tc and tc != src_canon and tc in inbound:
                inbound[tc] = inbound.get(tc, 0) + 1
                inbound_sources.setdefault(tc, set()).add(src_canon)

    last_seen = None
    by_status: dict[str, int] = {"dead_end": 0, "orphan": 0, "hub": 0, "healthy": 0}
    items = []
    for r in rows:
        if r.fetched_at and (last_seen is None or r.fetched_at > last_seen):
            last_seen = r.fetched_at
        url_canon = _canon(r.url)
        inbound_count = inbound.get(url_canon, 0)
        is_dead_end = (r.outbound_internal_count or 0) == 0 and not r.fetch_error
        is_orphan = (inbound_count == 0) and len(audited_urls) > 1
        topo_status = "healthy"
        if is_dead_end:
            topo_status = "dead_end"
        elif is_orphan:
            topo_status = "orphan"

        generic_ratio = (
            (r.generic_anchor_count or 0) / r.outbound_internal_count
            if r.outbound_internal_count and r.outbound_internal_count > 0 else 0.0
        )

        items.append({
            "url": r.url,
            "title": r.title,
            "fetch_status": r.fetch_status,
            "fetch_error": r.fetch_error,
            "outbound_internal_count": r.outbound_internal_count,
            "outbound_external_count": r.outbound_external_count,
            "generic_anchor_count": r.generic_anchor_count,
            "empty_anchor_count": r.empty_anchor_count,
            "duplicate_anchor_count": r.duplicate_anchor_count,
            "avg_anchor_length": r.avg_anchor_length,
            "inbound_count": inbound_count,
            "inbound_sources": sorted(list(inbound_sources.get(url_canon, set())))[:10],
            "outbound_internal_links": r.outbound_internal_links or [],
            "issues": r.issues or [],
            "linking_score": r.linking_score,
            "citation_count": r.citation_count,
            "topology": topo_status,
            "generic_ratio": generic_ratio,
            # recommended_action filled in after the hub pass so it can
            # be topology-aware (hub / orphan / dead-end override the
            # linking_score read).
        })

    inbound_values = sorted([i["inbound_count"] for i in items if i["inbound_count"] >= 2], reverse=True)
    hub_threshold = None
    if inbound_values:
        cutoff_index = max(0, int(len(inbound_values) * 0.2) - 1)
        hub_threshold = max(5, inbound_values[cutoff_index] if cutoff_index < len(inbound_values) else 5)
    for it in items:
        if it["topology"] == "healthy" and hub_threshold is not None and it["inbound_count"] >= hub_threshold and it["inbound_count"] >= 2:
            it["topology"] = "hub"
        it["recommended_action"] = _internal_link_action(
            it["linking_score"], it["outbound_internal_count"] or 0,
            it["generic_ratio"], it["topology"],
        )
        # Drop the helper field before serializing.
        it.pop("generic_ratio", None)
        by_status[it["topology"]] = by_status.get(it["topology"], 0) + 1

    total = len(items)
    avg_score = None
    if items:
        scored = [i["linking_score"] for i in items if i["linking_score"] is not None]
        avg_score = round(sum(scored) / len(scored), 1) if scored else None

    issues_by_type: dict[str, int] = {}
    for it in items:
        for issue in it["issues"]:
            t = issue.get("type") or "unknown"
            issues_by_type[t] = issues_by_type.get(t, 0) + 1

    return {
        "scan_id": scan_id,
        "pages": items,
        "summary": {
            "total": total,
            "avg_linking_score": avg_score,
            "dead_end": by_status["dead_end"],
            "orphan": by_status["orphan"],
            "hub": by_status["hub"],
            "healthy": by_status["healthy"],
            "issues_by_type": issues_by_type,
            "hub_threshold": hub_threshold,
        },
        "last_fetched": last_seen.isoformat() + "Z" if last_seen else None,
    }


@router.post("/{scan_id}/internal-linking/refresh")
async def refresh_internal_linking(
    scan_id: str,
    reset: bool = False,
    limit: int = 200,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue an `audit_internal_links` worker job. Free (no LLM) -
    plain HTTP + BeautifulSoup. ~1-2s per page over the wire, capped at
    200 URLs per run."""
    _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "audit_internal_links", {
        "reset": bool(reset),
        "limit": int(limit),
    })
    db.commit()
    return {"status": "enqueued", "scan_id": scan_id, "reset": bool(reset), "limit": limit}


# --- Sprint 10 : YouTube creator mapping ---------------------------------
# Channels LLMs already cite for the scan's brands. Same mine-LLM-citations
# pattern as Sprints 7/8/9 ; enrichment via YouTube oEmbed (free, no key)
# to recover channel name + author_url. Cf. project_10_action_features.md
# #10 + worker/handlers/audit_youtube_creators.py.

def _youtube_action(classification: str | None, target_cited: bool) -> dict:
    """Recommended action per channel row."""
    cls = classification or "target_only"
    if cls == "competitor_only":
        return {"label": "Pitch creator", "tone": "high"}
    if cls == "shared":
        return {"label": "Defend visibility", "tone": "medium"}
    if cls == "target_only" and target_cited:
        return {"label": "Nurture relationship", "tone": "positive"}
    return {"label": "Context only", "tone": "low"}


@router.get("/{scan_id}/youtube-creators")
async def get_youtube_creators(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return YouTube channels LLMs cite for this scan, sorted by
    leverage_score DESC. Channels enriched via oEmbed at audit time -
    `channel_url = '(unknown channel)'` collects videos that failed
    oEmbed (private / deleted / age-gated)."""
    _check_scan_access(scan_id, user, db)

    rows = (
        db.query(ScanYouTubeCreator)
        .filter(ScanYouTubeCreator.scan_id == scan_id)
        .order_by(
            ScanYouTubeCreator.leverage_score.desc().nullslast(),
            ScanYouTubeCreator.citation_count.desc(),
        )
        .all()
    )

    by_class: dict[str, int] = {}
    last_seen = None
    items = []
    total_videos = 0
    for r in rows:
        if r.fetched_at and (last_seen is None or r.fetched_at > last_seen):
            last_seen = r.fetched_at
        if r.classification:
            by_class[r.classification] = by_class.get(r.classification, 0) + 1
        total_videos += r.video_count or 0
        items.append({
            "channel_url": r.channel_url,
            "channel_name": r.channel_name,
            "channel_handle": r.channel_handle,
            "citation_count": r.citation_count,
            "video_count": r.video_count,
            "competitor_brands": list(r.competitor_brands or []),
            "target_cited": r.target_cited,
            "classification": r.classification,
            "top_videos": r.top_videos or [],
            "winning_questions": r.winning_questions or [],
            "leverage_score": r.leverage_score,
            "recommended_action": _youtube_action(r.classification, r.target_cited),
        })

    opportunities = by_class.get("competitor_only", 0) + by_class.get("shared", 0)
    return {
        "scan_id": scan_id,
        "channels": items,
        "summary": {
            "total": len(rows),
            "total_videos": total_videos,
            "by_classification": by_class,
            "competitor_only": by_class.get("competitor_only", 0),
            "shared": by_class.get("shared", 0),
            "target_only": by_class.get("target_only", 0),
            "opportunities": opportunities,
        },
        "last_fetched": last_seen.isoformat() + "Z" if last_seen else None,
    }


@router.post("/{scan_id}/youtube-creators/refresh")
async def refresh_youtube_creators(
    scan_id: str,
    reset: bool = False,
    limit: int = 200,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue an `audit_youtube_creators` worker job. Free (no LLM, no
    YouTube API key). oEmbed call per unique cited video URL, ~0.4s each
    polite throttle. 55 videos = ~25s ; 200-video cap = ~90s worst case."""
    _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "audit_youtube_creators", {
        "reset": bool(reset),
        "limit": int(limit),
    })
    db.commit()
    return {"status": "enqueued", "scan_id": scan_id, "reset": bool(reset), "limit": limit}


# --- Sprint 12 : crisis monitoring snapshot ------------------------------
# Per-brand counts of negative brand_mentions[] in this scan, categorized
# by content theme (safety / efficacy / ingredients / pricing / service /
# quality / other), with static playbook templates per category. Mirrors
# Sprint 9.1 noise gate by trusting the brand classification table.

# Static, vertical-agnostic playbook templates. Keyed by dominant_category.
# Each template returns one card the UI renders next to the brand row.
_CRISIS_PLAYBOOKS: dict[str, dict] = {
    "safety": {
        "title": "Safety / adverse-event signal",
        "tone": "urgent",
        "investigate": [
            "Pull the exact contexte snippet and verify the claim against pharmacovigilance, regulatory recall lists, and your own quality / adverse-event database.",
            "Identify whether the signal originates from the brand_mentions source (LLM-summarised public discussion) or a verified incident.",
            "Check the timeline : is this a freshly-emerging signal or recycled historical content the LLM is still surfacing?",
        ],
        "respond": [
            "Loop in pharmacovigilance / regulatory affairs before any external communication.",
            "Coordinate with medical / scientific affairs to prepare an evidence-backed correction.",
            "If the claim is materially false, file a takedown / correction request with the cited sources.",
        ],
        "channels": ["pharmacovigilance@", "regulatory affairs", "medical comms", "PR (gated until cleared)"],
        "timeline": "Same day triage. 48h cross-functional response if signal is verified.",
    },
    "ingredients": {
        "title": "Ingredient / composition concern",
        "tone": "high",
        "investigate": [
            "Confirm whether the disputed ingredient is actually in the current formulation (the LLM may be citing a deprecated formula).",
            "Map the concern to the underlying source : NGO list, regulatory body, viral consumer post, or pseudo-scientific blog.",
            "Check whether competitors face the same concern (shared crisis vs target-only).",
        ],
        "respond": [
            "Prepare an ingredient-level explainer (why we use it, at what concentration, what the science says).",
            "If the formula has changed, push corrected references to Wikipedia, brand site, and key media that the LLM cites.",
            "Brief customer-facing teams with a 3-sentence response talk track.",
        ],
        "channels": ["formulation / R&D", "scientific affairs", "digital comms", "Wikipedia edit guidelines"],
        "timeline": "Within 1 week, ahead of the next scan refresh.",
    },
    "efficacy": {
        "title": "Efficacy doubt",
        "tone": "medium",
        "investigate": [
            "Sample the negative excerpts : is it dosage / duration of use / wrong indication, or genuine product underperformance?",
            "Check whether positive mentions cluster on the same product line or a different one (signal might be product-specific).",
            "Pull clinical / consumer-test evidence available for the contested benefit.",
        ],
        "respond": [
            "Add measurable efficacy claims + clinical data references to the product pages that LLMs already cite (cf. Page Audit tab).",
            "Brief influencer / partner network to feature before-after / how-to-use content correcting the misuse pattern.",
            "If a competitor wins on a specific benefit claim, prepare comparative content for the PR / Media tab.",
        ],
        "channels": ["product marketing", "clinical / consumer studies", "influencer / PR", "SEO / content"],
        "timeline": "2-4 weeks rolling. Re-measure on the next scan.",
    },
    "pricing": {
        "title": "Price / value perception",
        "tone": "medium",
        "investigate": [
            "Identify the comparison anchor : which competitor / channel is the LLM citing as the cheaper alternative?",
            "Check whether the price perception is driven by promotion gaps or by structurally cheaper rivals.",
        ],
        "respond": [
            "Surface value-justification content (premium-grade ingredients, manufacturing origin, certifications) on the cited pages.",
            "Review promotion calendar with the channel team if the gap is structural.",
            "If the issue is comparator-specific, prepare a 'why we cost more' talk track for customer-facing teams.",
        ],
        "channels": ["pricing / revenue", "ecommerce", "customer support"],
        "timeline": "1-2 weeks for content, longer for pricing decisions.",
    },
    "service": {
        "title": "Customer service / fulfilment",
        "tone": "medium",
        "investigate": [
            "Cross-reference negative excerpts with internal CS ticket volumes and NPS detractors for the same time window.",
            "Identify whether the signal is about delivery, returns, support latency, or refund process.",
        ],
        "respond": [
            "Fix the underlying CS process before any external comms (silent fix > public apology).",
            "Update help-centre FAQ pages that LLMs cite, with explicit policy language.",
        ],
        "channels": ["customer service ops", "ecommerce / logistics", "help-centre / content"],
        "timeline": "Operational fix in 2-4 weeks ; content update same week.",
    },
    "quality": {
        "title": "Product quality / defect",
        "tone": "high",
        "investigate": [
            "Pull the defect type from the contexte (packaging leak, breakage, batch issue).",
            "Tie to internal quality returns + check whether a specific batch / SKU shows over-representation.",
        ],
        "respond": [
            "If batch-isolated, brief CS with the affected SKU range and replacement protocol.",
            "If systemic, escalate to quality / supply with an action plan.",
        ],
        "channels": ["quality assurance", "supply chain", "customer service"],
        "timeline": "Same week triage. Cross-functional plan within 2 weeks.",
    },
    "other": {
        "title": "General negative signal",
        "tone": "low",
        "investigate": [
            "Pull the top contexts and identify the underlying theme - the keyword categorisation didn't match a known bucket.",
            "Decide whether the signal warrants its own playbook category in a future iteration.",
        ],
        "respond": [
            "If it's an emerging theme (e.g. sustainability, animal welfare, social positioning), prepare a position statement.",
            "Otherwise monitor on the next scan.",
        ],
        "channels": ["brand strategy", "corporate communications"],
        "timeline": "Monitor on next scan refresh.",
    },
}


def _crisis_action(severity_label: str | None, classification: str) -> dict:
    """Recommended action chip per brand row."""
    label_map = {
        "critical": ("Crisis war room", "urgent"),
        "high":     ("Investigate now", "high"),
        "medium":   ("Triage this week", "medium"),
        "low":      ("Monitor", "low"),
        "none":     ("All clear", "positive"),
    }
    sev = severity_label or "none"
    text, tone = label_map.get(sev, ("Monitor", "low"))
    # Slightly different framing for competitors : their crisis is intelligence,
    # not a fire to put out.
    if classification == "competitor" and sev in ("high", "critical"):
        return {"label": "Competitive opening", "tone": "positive"}
    return {"label": text, "tone": tone}


@router.get("/{scan_id}/crisis")
async def get_crisis(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Per-scan crisis snapshot. One row per (scan, brand) with severity
    label + dominant category + top contexts + matching playbook."""
    _check_scan_access(scan_id, user, db)

    # Sort : severity DESC, then negative_count DESC, then total_mentions
    # DESC as the tiebreaker. Without the third key the 110 zero-severity
    # competitor rows on a healthy scan render in DB insert order, so
    # product-line variants (which are rarely cited by exact name) drown
    # the top-cited master brands (Bioderma, LRP, Vichy etc.).
    rows = (
        db.query(ScanCrisisSignal)
        .filter(ScanCrisisSignal.scan_id == scan_id)
        .order_by(
            ScanCrisisSignal.severity.desc().nullslast(),
            ScanCrisisSignal.negative_count.desc(),
            ScanCrisisSignal.total_mentions.desc(),
        )
        .all()
    )

    # Pick the focus brand explicitly for the hero row : when the scan
    # has many my_brand entries (product lines) we want the master brand
    # in the headline, not whichever variant happened to sort first.
    scan_row = db.query(Scan).filter(Scan.id == scan_id).first()
    focus_brand_id = str(scan_row.focus_brand_id) if (scan_row and scan_row.focus_brand_id) else None

    target_row = None
    target_candidate_by_mentions = None
    target_product_lines: list[dict] = []  # non-focus my_brand rows with crisis signal
    competitor_rows = []
    last_seen = None
    by_severity: dict[str, int] = {"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
    for r in rows:
        if r.created_at and (last_seen is None or r.created_at > last_seen):
            last_seen = r.created_at
        item = {
            "brand_id": str(r.brand_id),
            "brand_name": r.brand_name,
            "classification": r.brand_classification,
            "negative_count": r.negative_count,
            "positive_count": r.positive_count,
            "neutral_count": r.neutral_count,
            "total_mentions": r.total_mentions,
            "negative_ratio": r.negative_ratio,
            "severity": r.severity,
            "severity_label": r.severity_label,
            "dominant_category": r.dominant_category,
            "category_breakdown": r.category_breakdown or {},
            "top_contexts": r.top_contexts or [],
            "topic_clusters": r.topic_clusters or [],
            "shared_with": r.shared_with or [],
            "recommended_action": _crisis_action(r.severity_label, r.brand_classification),
            "playbook": _CRISIS_PLAYBOOKS.get(r.dominant_category or "other") if r.negative_count > 0 else None,
        }
        if r.severity_label:
            by_severity[r.severity_label] = by_severity.get(r.severity_label, 0) + 1
        if r.brand_classification == "my_brand":
            if focus_brand_id and str(r.brand_id) == focus_brand_id:
                target_row = item
            else:
                # Non-focus my_brand row : a product / range variant. Surface
                # it ONLY when it carries a real signal (any negative mention
                # OR severity above 'none'). Otherwise it's just one more row
                # of tied-zero noise the user doesn't need to see.
                if (item["negative_count"] or 0) > 0 or (item["severity"] or 0) > 15:
                    target_product_lines.append(item)
            # Fallback : the my_brand row with the most total mentions = the
            # master brand most of the time (a master brand collects citations
            # across all its product mentions via the brand resolver).
            if target_candidate_by_mentions is None or (item["total_mentions"] or 0) > (target_candidate_by_mentions["total_mentions"] or 0):
                target_candidate_by_mentions = item
        else:
            competitor_rows.append(item)
    if target_row is None:
        target_row = target_candidate_by_mentions
        # If focus_brand wasn't wired we may have moved the fallback into
        # target_product_lines too ; remove duplicate if it's there.
        if target_row is not None:
            target_product_lines = [
                p for p in target_product_lines
                if p.get("brand_id") != target_row.get("brand_id")
            ]
    # Order product lines : highest severity first.
    target_product_lines.sort(key=lambda p: -((p["severity"] or 0)))

    # Overall severity = the target's severity, or the worst severity
    # across all rows if no target row exists.
    overall_label = "none"
    overall_severity = 0
    if target_row:
        overall_label = target_row["severity_label"] or "none"
        overall_severity = target_row["severity"] or 0
    elif rows:
        overall_severity = rows[0].severity or 0
        overall_label = rows[0].severity_label or "none"

    # Aggregate signal across all my_brand rows (focus + product variants).
    # The hero focus brand may show "All clear" while a product variant
    # carries the actual negative - surface that count in the summary so
    # the user knows to look at the breakdown section.
    pl_negative_total = sum((p["negative_count"] or 0) for p in target_product_lines)
    pl_in_crisis = sum(1 for p in target_product_lines if (p["severity"] or 0) >= 36)

    return {
        "scan_id": scan_id,
        "target": target_row,
        "target_product_lines": target_product_lines,
        "competitors": competitor_rows,
        "summary": {
            "overall_severity": overall_severity,
            "overall_label": overall_label,
            "by_severity": by_severity,
            "brand_count": len(rows),
            "target_negative_count": target_row["negative_count"] if target_row else 0,
            "target_product_line_negative_count": pl_negative_total,
            "target_product_lines_in_crisis": pl_in_crisis,
            "competitor_in_crisis": sum(1 for c in competitor_rows if (c["severity"] or 0) >= 36),
            "shared_crisis_topics": sum(len(s.get("shared_topics", [])) for s in (target_row["shared_with"] if target_row else [])),
        },
        "playbooks_library": _CRISIS_PLAYBOOKS,
        "last_fetched": last_seen.isoformat() + "Z" if last_seen else None,
    }


@router.post("/{scan_id}/crisis/refresh")
async def refresh_crisis(
    scan_id: str,
    reset: bool = True,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue a `build_crisis_radar` worker job. Free (no LLM) - pure
    aggregation + keyword categorisation over brand_mentions[] already
    in the DB. Completes in <2s on a 100-question scan."""
    _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "build_crisis_radar", {"reset": bool(reset)})
    db.commit()
    return {"status": "enqueued", "scan_id": scan_id, "reset": bool(reset)}


# --- Sprint 14 : AI Act compliance pack ---------------------------------
# Single aggregator endpoint that returns everything the per-scan
# Compliance report needs : providers used, models, runs_depth, dates,
# triggered-by user, counts, distinct cited domains, brand classification
# snapshot. The page renders this plus a static methodology + sub-processors
# block, then a print stylesheet lets the user save as PDF.
# EU AI Act applies 2 August 2026 (general purpose AI). Doc Methodology +
# audit trail + transparency satisfy the procurement asks that block EU
# enterprise deals.

@router.get("/{scan_id}/compliance-data")
async def get_compliance_data(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Aggregated payload for the per-scan AI Act compliance report.
    Pulls scan + provider mix + counts + citation domains + brand snapshot
    in one round trip. Read-only."""
    from sqlalchemy import text as _text
    scan = _check_scan_access(scan_id, user, db)

    # User who triggered the scan.
    from models import User
    triggered_by_email = None
    triggered_by_name = None
    if scan.created_by:
        u = db.query(User).filter(User.id == scan.created_by).first()
        if u:
            triggered_by_email = u.email
            triggered_by_name = u.name

    # Client + organization name resolution (org via clients_orgs if wired).
    client_row = db.query(Client).filter(Client.id == scan.client_id).first()
    client_name = client_row.name if client_row else None

    # Provider mix + per-provider run count.
    provider_rows = db.execute(_text(
        """
        SELECT provider, COUNT(*) AS n_responses,
               COUNT(DISTINCT run_index) AS distinct_runs
          FROM scan_llm_results
         WHERE scan_id = :sid
         GROUP BY provider
         ORDER BY n_responses DESC
        """
    ), {"sid": scan_id}).fetchall()
    providers = [
        {"provider": r.provider, "responses": int(r.n_responses), "distinct_runs": int(r.distinct_runs)}
        for r in provider_rows
    ]

    # runs_depth : max run_index across all rows. The summary field may
    # carry the configured value too, prefer it when set.
    max_run = db.execute(_text(
        "SELECT MAX(run_index) FROM scan_llm_results WHERE scan_id = :sid"
    ), {"sid": scan_id}).scalar() or 1
    configured_runs = None
    if scan.summary and isinstance(scan.summary, dict):
        configured_runs = scan.summary.get("runs_depth")

    # Aggregate counts.
    counts = {
        "topics": db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).count(),
        "personas": db.query(ScanPersona).filter(ScanPersona.scan_id == scan_id).count(),
        "questions": db.query(ScanQuestion).filter(ScanQuestion.scan_id == scan_id).count(),
        "llm_responses": db.query(ScanLLMResult).filter(ScanLLMResult.scan_id == scan_id).count(),
    }

    # Distinct citation domains - the user's "data sources" for AI Act
    # transparency. Group by domain, with counts.
    citation_rows = db.execute(_text(
        """
        SELECT lower(citation->>'domaine') AS domain, COUNT(*) AS n
          FROM scan_llm_results,
               LATERAL jsonb_array_elements(citations) AS citation
         WHERE scan_id = :sid
           AND citation->>'domaine' IS NOT NULL
         GROUP BY lower(citation->>'domaine')
         ORDER BY n DESC
         LIMIT 50
        """
    ), {"sid": scan_id}).fetchall()
    cited_domains = [{"domain": r.domain, "citations": int(r.n)} for r in citation_rows]
    cited_domains_total = db.execute(_text(
        """
        SELECT COUNT(DISTINCT lower(citation->>'domaine')) AS n
          FROM scan_llm_results, LATERAL jsonb_array_elements(citations) AS citation
         WHERE scan_id = :sid AND citation->>'domaine' IS NOT NULL
        """
    ), {"sid": scan_id}).scalar() or 0

    # Brand classification snapshot at the time of the report.
    brand_rows = db.execute(_text(
        """
        SELECT cb.name, cb.canonical_name, sbc.classification
          FROM scan_brand_classifications sbc
          JOIN client_brands cb ON cb.id = sbc.brand_id
         WHERE sbc.scan_id = :sid
           AND sbc.classification IN ('my_brand','competitor','ignored')
         ORDER BY sbc.classification, cb.name
        """
    ), {"sid": scan_id}).fetchall()
    by_class: dict[str, list[dict]] = {"my_brand": [], "competitor": [], "ignored": []}
    for r in brand_rows:
        by_class[r.classification].append({"name": r.canonical_name or r.name})

    duration_seconds = None
    if scan.completed_at and scan.started_at:
        try:
            duration_seconds = int((scan.completed_at - scan.started_at).total_seconds())
        except Exception:
            duration_seconds = None

    return {
        "scan_id": scan_id,
        "scan": {
            "id": str(scan.id),
            "name": scan.name,
            "domain": scan.domain,
            "status": scan.status,
            "scan_type": scan.scan_type,
            "run_index": scan.run_index,
            "schedule": scan.schedule,
            "created_at": scan.created_at.isoformat() + "Z" if scan.created_at else None,
            "started_at": scan.started_at.isoformat() + "Z" if scan.started_at else None,
            "completed_at": scan.completed_at.isoformat() + "Z" if scan.completed_at else None,
            "duration_seconds": duration_seconds,
            "focus_brand_name": (scan.focus_brand.canonical_name or scan.focus_brand.name) if scan.focus_brand else None,
        },
        "client": {"id": str(scan.client_id), "name": client_name},
        "triggered_by": {"email": triggered_by_email, "name": triggered_by_name},
        "providers": providers,
        "runs_depth": {
            "configured": configured_runs,
            "observed_max": int(max_run),
        },
        "counts": counts,
        "data_sources": {
            "distinct_domains_total": int(cited_domains_total),
            "top_50": cited_domains,
        },
        "brand_classifications": by_class,
    }


# --- Sentiment Judge : Haiku-as-judge layer on top of BrandAnalyzer ----
# Refreshes per-mention judgements for negative brand_mentions on this
# scan. Crisis radar + Overview chip prefer the judged sentiment when
# verdict='overturn'. Cap : $0.05/scan, 200 mentions max per run.

@router.post("/{scan_id}/sentiment-judge/refresh")
async def refresh_sentiment_judge(
    scan_id: str,
    reset: bool = False,
    limit: int = 200,
    budget: float = 0.05,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue a `judge_sentiment` worker job. Re-reads every
    `brand_mentions[].sentiment = 'négatif'` with Claude Haiku 4.5 and
    decides confirm / overturn / hedge. ~$0.001 per call ; per-scan
    budget cap of $0.05 by default."""
    _check_scan_access(scan_id, user, db)
    _create_job(db, scan_id, "judge_sentiment", {
        "reset": bool(reset),
        "limit": int(limit),
        "budget": float(budget),
    })
    db.commit()
    return {"status": "enqueued", "scan_id": scan_id, "reset": bool(reset), "limit": limit, "budget": budget}


@router.get("/{scan_id}/sentiment-judgements")
async def get_sentiment_judgements(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the latest sentiment judgement per (slr_id, mention_index)
    on this scan. Used by the Crisis tab for transparency : show the user
    how many false positives were caught + the reasoning per overturn."""
    _check_scan_access(scan_id, user, db)

    # Latest judgement per (slr_id, mention_index) using a DISTINCT ON
    # window. PostgreSQL specific but cleaner than a self-join.
    from sqlalchemy import text as _text
    rows = db.execute(_text(
        """
        SELECT DISTINCT ON (slr_id, mention_index)
               id::text, scan_id::text, slr_id::text, mention_index,
               brand_name, raw_sentiment, raw_justification,
               judge_verdict, judged_sentiment, judge_reasoning,
               judge_model, judge_cost_usd, judge_run_at
          FROM scan_sentiment_judgements
         WHERE scan_id = :sid
         ORDER BY slr_id, mention_index, judge_run_at DESC
        """
    ), {"sid": scan_id}).fetchall()

    items = []
    by_verdict: dict[str, int] = {"confirm": 0, "overturn": 0, "hedge": 0}
    total_cost = 0.0
    for r in rows:
        by_verdict[r.judge_verdict] = by_verdict.get(r.judge_verdict, 0) + 1
        total_cost += float(r.judge_cost_usd or 0)
        items.append({
            "id": r.id,
            "slr_id": r.slr_id,
            "mention_index": r.mention_index,
            "brand_name": r.brand_name,
            "raw_sentiment": r.raw_sentiment,
            "raw_justification": r.raw_justification,
            "judge_verdict": r.judge_verdict,
            "judged_sentiment": r.judged_sentiment,
            "judge_reasoning": r.judge_reasoning,
            "judge_model": r.judge_model,
            "judge_cost_usd": float(r.judge_cost_usd or 0),
            "judge_run_at": r.judge_run_at.isoformat() + "Z" if r.judge_run_at else None,
        })

    return {
        "scan_id": scan_id,
        "judgements": items,
        "summary": {
            "total": len(items),
            "by_verdict": by_verdict,
            "confirmed": by_verdict.get("confirm", 0),
            "overturned": by_verdict.get("overturn", 0),
            "hedged": by_verdict.get("hedge", 0),
            "total_cost_usd": round(total_cost, 6),
        },
    }
