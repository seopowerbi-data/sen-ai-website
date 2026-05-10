from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sqlalchemy import func
from sqlalchemy.orm import joinedload
from models import (
    Scan, ScanKeyword, ScanTopic, ScanPersona, ScanQuestion, ScanLLMResult,
    ScanBrandClassification, ScanBrandTopic, ScanOpportunity, ClientBrand,
    Job, UserClient, get_db,
)
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


class ScanUpdate(BaseModel):
    """PATCH payload — all fields optional."""
    name: str | None = None
    focus_brand_id: str | None = None
    schedule: str | None = None  # manual | weekly | monthly


class ScanConfigUpdate(BaseModel):
    """PATCH /scans/{id}/config — update scan configuration."""
    providers: list[str] | None = None


class BrandClassify(BaseModel):
    brand_id: str
    classification: str  # my_brand | competitor | ignored | unclassified
    is_focus: bool = False


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
    """PATCH payload — every field optional. Used for rename / toggle / reassign."""
    name: str | None = None
    data: dict | None = None
    topic_id: str | None = None
    is_active: bool | None = None


class QuestionCreate(BaseModel):
    persona_id: str
    question: str
    type_question: str | None = None


class QuestionUpdate(BaseModel):
    """PATCH payload — every field optional."""
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
_ROLE_RANK = {"viewer": 0, "editor": 1, "owner": 2}
_DESTRUCTIVE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _check_scan_access(scan_id: str, user, db: Session) -> Scan:
    """Verify the current user has access to a scan, with method-aware RBAC.

    H6: on destructive HTTP methods (POST/PUT/PATCH/DELETE) the caller's
    `UserClient.role` must be at least 'editor'. On GET it just needs any
    link (viewer is enough). The HTTP method is read from a contextvar set
    by `request_method_middleware` in main.py — that lets every existing
    call site (38 of them across this router) gain RBAC enforcement
    without changing its signature.

    Note that this enforcement only matters once team features ship — today
    every account is solo-owner of its own client(s), so the check is a
    no-op in practice. It's wired up now so future invitee accounts in
    'viewer' role can never silently bypass write protections.
    """
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(404, "Scan not found")
    link = db.query(UserClient).filter(
        UserClient.user_id == user.id,
        UserClient.client_id == scan.client_id,
    ).first()
    if not link:
        raise HTTPException(403, "Access denied")

    method = current_request_method.get()
    if method in _DESTRUCTIVE_METHODS:
        rank = _ROLE_RANK.get(link.role, -1)
        if rank < _ROLE_RANK["editor"]:
            raise HTTPException(
                403,
                f"Insufficient role: '{link.role}' cannot {method} on this scan "
                f"(requires 'editor' or 'owner')",
            )
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
    link = db.query(UserClient).filter(
        UserClient.user_id == user.id,
        UserClient.client_id == req.client_id,
    ).first()
    if not link:
        raise HTTPException(403, "Access denied to this client")
    # H6: creating a scan is a destructive write — viewers can't.
    if _ROLE_RANK.get(link.role, -1) < _ROLE_RANK["editor"]:
        raise HTTPException(
            403,
            f"Insufficient role: '{link.role}' cannot create scans (requires 'editor' or 'owner')",
        )

    if req.max_position not in (10, 30, 50):
        raise HTTPException(400, "max_position must be 10, 30 or 50")
    if req.max_urls < 100 or req.max_urls > 5000:
        raise HTTPException(400, "max_urls must be between 100 and 5000")

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
    link = db.query(UserClient).filter(
        UserClient.user_id == user.id,
        UserClient.client_id == client_id,
    ).first()
    if not link:
        raise HTTPException(403, "Access denied")

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
    """Update scan config (providers, etc.). JSONB merge — only provided keys are updated."""
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

@router.post("/{scan_id}/generate-brief")
@limiter.limit("5/minute")
async def generate_brief(request: Request, scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Enqueue domain brief generation via OpenAI web search. Idempotent."""
    scan = _check_scan_access(scan_id, user, db)
    existing = db.query(Job).filter(
        Job.scan_id == scan_id,
        Job.job_type == "generate_domain_brief",
        Job.status.in_(["pending", "running"]),
    ).first()
    if existing:
        return {"status": "already_running", "job_id": str(existing.id)}
    _create_job(db, scan_id, "generate_domain_brief")
    db.commit()
    return {"status": "job_created"}


@router.get("/{scan_id}/brief")
async def get_brief(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return domain brief + generation status."""
    scan = _check_scan_access(scan_id, user, db)
    brief = (scan.config or {}).get("domain_brief")
    job = db.query(Job).filter(
        Job.scan_id == scan_id,
        Job.job_type == "generate_domain_brief",
    ).order_by(Job.created_at.desc()).first()
    return {
        "domain_brief": brief,
        "generation_status": job.status if job else ("completed" if brief else None),
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
    Other buckets (competitor/ignored/unclassified) stay flat — hierarchy only
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
        are in the SAME bucket — otherwise children are promoted to top-level (orphans).
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


@router.post("/{scan_id}/brands/validate")
async def validate_scan_brands(scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Gate 2: validate the per-scan brand classification and enqueue persona generation.

    Requires:
    - scan.status == 'brands_ready' (set by assign_keywords handler in J2 — until then this gate
      is not reachable from the happy path; the endpoint is wired up so Gate 2 UI can call it
      as soon as the worker refactor lands)
    - focus_brand_id IS NOT NULL
    - at least one brand classified as 'my_brand' (the focus itself counts)
    """
    scan = _check_scan_access(scan_id, user, db)

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
        raise HTTPException(400, "Focus brand row is inconsistent — reclassify it as my_brand with is_focus=true")

    scan.status = "generating_personas"
    scan.updated_at = datetime.utcnow()
    _create_job(db, scan_id, "generate_personas")
    db.commit()
    return {"status": "generating_personas", "my_brand_count": my_brand_count}


# --- Rescan + lineage ---

@router.post("/{scan_id}/rescan")
@limiter.limit("10/minute")
async def rescan(request: Request, scan_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a child scan that inherits topics, personas, questions and brand classifications
    from the parent. Skips Gate 1 and Gate 2 — goes straight to fetching_keywords (fresh HaloScan
    + fresh LLM) while reusing the validated setup.

    Phase 1: copies topics, personas, questions, scan_brand_classifications.
             Does NOT copy opportunities or llm_results (those are fresh per run).
    """
    parent = _check_scan_access(scan_id, user, db)

    # Count the active questions that will be copied to the child — this is
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

    # Credit gate: same pattern as launch_scan — lock client, check balance,
    # then debit (with scan_id=child.id once it exists, so a worker failure
    # auto-refunds against the child).
    from routers.stripe import get_credit_balance, add_credits, lock_client_credits
    lock_client_credits(str(parent.client_id), db)
    balance = get_credit_balance(str(parent.client_id), "scan", db)
    if balance < active_questions:
        raise HTTPException(402, {
            "error": "insufficient_credits",
            "need": active_questions,
            "have": balance,
            "message": f"Need {active_questions} scan credits but only {balance} available",
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

    # Pre-debit credits — same lock from above is still held within this txn.
    add_credits(
        client_id=str(parent.client_id),
        credit_type="scan",
        amount=-active_questions,
        description=f"Rescan launched: {active_questions} questions",
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

    Works whether scan_id points at the root or at any child — we resolve to the root first.
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

        # Top 5 keywords — deduplicated by keyword text, aggregated across pages
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
    if req.is_active is not None:
        topic.is_active = req.is_active
    db.commit()
    return {"id": str(topic.id), "name": topic.name, "is_active": topic.is_active}


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
    # this endpoint while status is 'keywords_fetched' — without this guard, each
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

    def _serialize_question(q):
        return {
            "id": str(q.id),
            "question": q.question,
            "type_question": q.type_question,
            "is_active": bool(q.is_active),
        }

    def _serialize_persona(p):
        p_questions = questions_by_persona.get(str(p.id), [])
        return {
            "id": str(p.id),
            "name": p.name,
            "data": p.data,
            "topic_id": str(p.topic_id) if p.topic_id else None,
            "is_active": bool(p.is_active),
            "questions": [_serialize_question(q) for q in p_questions],
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
    """Update a persona — all fields optional. Used for rename, toggle, reassign."""
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


@router.post("/{scan_id}/personas/{persona_id}/generate-questions")
async def generate_persona_questions(scan_id: str, persona_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Enqueue a worker job to generate 15 questions for a custom persona."""
    _check_scan_access(scan_id, user, db)
    persona = db.query(ScanPersona).filter(ScanPersona.id == persona_id, ScanPersona.scan_id == scan_id).first()
    if not persona:
        raise HTTPException(404, "Persona not found")
    db.add(Job(scan_id=scan_id, job_type="generate_persona_questions", payload={"persona_id": persona_id}))
    db.commit()
    return {"status": "generating", "persona_id": persona_id}


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
    """Update a question — all fields optional. Used for toggle, inline edit, retype."""
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

    # Credit check: debit scan credits (1 credit = 1 question).
    # Lock the client row FIRST so the balance read + debit are atomic.
    # Without the lock, two concurrent launches could both observe the
    # same balance and each pass the check, causing a double-spend.
    from routers.stripe import get_credit_balance, add_credits, lock_client_credits
    lock_client_credits(str(scan.client_id), db)
    balance = get_credit_balance(str(scan.client_id), "scan", db)
    if balance < active_questions:
        raise HTTPException(402, {
            "error": "insufficient_credits",
            "need": active_questions,
            "have": balance,
            "message": f"Need {active_questions} scan credits but only {balance} available",
        })

    # Pre-debit credits (re-uses the same lock — re-entrant within this txn)
    add_credits(
        client_id=str(scan.client_id),
        credit_type="scan",
        amount=-active_questions,
        description=f"Scan launched: {active_questions} questions",
        db=db,
        scan_id=scan_id,
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
              details={"questions": active_questions, "credits_used": active_questions})
    db.commit()
    return {"status": "scanning", "credits_used": active_questions, "credits_remaining": balance - active_questions}


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

    # --- Competitors — cross-referenced with Gate 2 classifications ---
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

    # --- Details (each test) — enriched with brand_mentions, brand_analysis, intention_cachee ---
    topics_map = {str(t.id): t for t in db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).all()}

    # Build question_text → intention_cachee lookup from persona.data.questions
    intent_lookup = {}  # question_text_lower → intention_cachee
    for p in personas:
        for pq in (p.data or {}).get("questions", []):
            if pq.get("question") and pq.get("intention_cachee"):
                intent_lookup[pq["question"].strip().lower()] = pq["intention_cachee"]

    details = []
    for r in results:
        q = db.query(ScanQuestion).filter(ScanQuestion.id == r.question_id).first()
        persona = persona_map.get(str(q.persona_id)) if q else None
        topic = topics_map.get(str(persona.topic_id)) if persona and persona.topic_id else None
        intention = intent_lookup.get((q.question or "").strip().lower()) if q else None
        bm_mentioned = (r.brand_analysis or {}).get("marque_cible_mentionnee", False)
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

    Returns rich Persona Cards data — the centerpiece of the AI Brand Audit deep dive.
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

    # Build question_text → intention_cachee lookup
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
                "intention_cachee": intent_lookup.get((q.question or "").strip().lower()),
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
