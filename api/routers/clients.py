from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from services.rate_limit import limiter
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sqlalchemy.orm.attributes import flag_modified

from models import Client, ClientBrand, ClientBrandPage, ClientCredit, Job, ScanBrandClassification, UserClient, get_db
from services.access import (
    check_client_access, list_user_clients,
    resolve_active_client_id, resolve_active_organization_id,
)
from services.auth_service import get_current_user
from services.request_context import current_request_method
from services.sanitize import strip_tags

router = APIRouter()

# Phase E.C.5.1 — the duplicated _ROLE_RANK / _DESTRUCTIVE_METHODS pair lived
# here, in brands.py, and in scans.py back when each router did its own
# _check_client_access(). After the migration to services/access.py these
# constants are unused — services/access.ROLE_RANK is the single source of
# truth. Kept the comment to flag the cleanup if anyone re-introduces a
# local rank dict.


def _check_client_access(client_id: str, user, db: Session):
    """Thin wrapper over services.access.check_client_access (Phase E.C).

    Kept under the legacy name so existing call sites don't need to change.
    Centralized helper reads Organization + UserClient (legacy fallback)
    and surfaces consistent 403 messages.
    """
    from services.access import check_client_access
    check_client_access(client_id, user, db)


class ClientResponse(BaseModel):
    id: str
    name: str
    brand: str | None
    apps: dict | None = None

    model_config = {"from_attributes": True}


class ClientCreate(BaseModel):
    name: str
    brand: str | None = None


@router.get("/", response_model=list[ClientResponse])
async def list_clients(
    active_organization_id: str | None = Cookie(default=None),
    active_client_id: str | None = Cookie(default=None),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Phase E.C.2 — resolve the effective active org : explicit cookie (if
    # user is still a member) > personal org > None. Defaulting to the
    # personal org is critical because most legacy dashboard pages do
    # `clients[0]?.id` to find "the user's workspace" ; without a default,
    # adding a second org would silently push the personal client out of
    # position 0 when sorted alphabetically (observed 2026-05-15 smoke).
    effective_org_id = resolve_active_organization_id(user, db, active_organization_id)
    clients = list_user_clients(user, db, organization_id=effective_org_id)
    if not clients and effective_org_id:
        # Defensive : personal org has zero clients (shouldn't happen post-C.1
        # backfill, but if it does, fall back to the union so the user is
        # never stuck with zero workspaces).
        clients = list_user_clients(user, db, organization_id=None)

    # Phase E.C.3 — when the user has pinned an active client (via the
    # workspace switcher on /app/org), float it to index 0 so legacy pages
    # that do `clients[0]?.id` automatically pick the right workspace.
    # Stale cookie pointing at another org's client → resolve_active_client_id
    # returns None, no reordering happens.
    effective_client_id = resolve_active_client_id(
        user, db, active_client_id, scope_organization_id=effective_org_id,
    )
    if effective_client_id:
        clients = sorted(clients, key=lambda c: 0 if str(c.id) == effective_client_id else 1)

    return [ClientResponse(id=str(c.id), name=c.name, brand=c.brand, apps=c.apps) for c in clients]


class SetActiveClientRequest(BaseModel):
    client_id: str


@router.post("/active")
async def set_active_client(
    req: SetActiveClientRequest,
    response: Response,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Pin a client as the active workspace. Same HttpOnly-cookie pattern as
    /api/organizations/active. Use case : agency user with 5 clients in one
    org wants /app/dashboard to land on client #3 every time."""
    check_client_access(req.client_id, user, db)
    response.set_cookie(
        "active_client_id", req.client_id,
        httponly=True, secure=True, samesite="lax",
        max_age=60 * 60 * 24 * 180, path="/",
    )
    return {"ok": True, "client_id": req.client_id}


@router.delete("/active")
async def clear_active_client(response: Response, user=Depends(get_current_user)):
    response.delete_cookie("active_client_id", path="/")
    return {"ok": True}


@router.post("/")
async def create_client(req: ClientCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if user already has a client
    existing = db.query(UserClient).filter(UserClient.user_id == user.id).first()
    if existing:
        client = db.query(Client).filter(Client.id == existing.client_id).first()
        return ClientResponse(id=str(client.id), name=client.name, brand=client.brand, apps=client.apps)

    # Sprint 15.2 : agency-intent users get an auto-created Organization
    # at their first workspace creation so the header org-switcher pops up
    # and the "use the header dropdown to add more clients" promise from
    # /welcome holds. Non-agency users stay on the legacy UserClient-only
    # path (1 user = 1 personal client) which is fine for solo brand
    # owners. The org auto-creation only fires when the user has NO
    # existing org and signup_intent='agency'.
    from models import Organization, OrganizationUser
    org_id = None
    if (user.signup_intent or "").lower() == "agency":
        already_owns_org = db.query(OrganizationUser).filter(
            OrganizationUser.user_id == user.id
        ).first()
        if not already_owns_org:
            base_name = (user.name or "").strip() or (user.email.split("@")[0] if user.email else "")
            org_name = f"{base_name}'s agency" if base_name else "My agency"
            org = Organization(
                name=org_name[:120],
                is_personal=False,
                pool_billing=True,  # agency model = shared credit pool across clients
            )
            db.add(org)
            db.flush()
            db.add(OrganizationUser(user_id=user.id, organization_id=org.id, role="owner"))
            org_id = org.id

    # Create new client + link user as owner
    # Welcome bonus is now granted on email verification (H3), not here
    client = Client(
        name=strip_tags(req.name),
        brand=strip_tags(req.brand),
        organization_id=org_id,
    )
    db.add(client)
    db.flush()

    db.add(UserClient(user_id=user.id, client_id=client.id, role="manager"))

    # Sprint 15.2 fix - welcome bonus grant moved here from the auth flow.
    # Background : _grant_welcome_bonus in auth.py looks up UserClient to
    # find a client_id, but at signup-time the user has NO client yet
    # (the client is created AFTER signup, via this endpoint, by the
    # welcome wizard). So the bonus silently no-op'd for every fresh
    # account. Granting at first-client-creation guarantees the 50
    # credits land where the user expects them to. Idempotent on the
    # description so a re-create-client edge case doesn't double-grant.
    if user.is_email_verified:
        already_granted = db.query(ClientCredit).filter(
            ClientCredit.client_id == client.id,
            ClientCredit.description == "Welcome bonus — 50 free scan credits",
        ).first()
        if not already_granted:
            db.add(ClientCredit(
                client_id=client.id,
                credit_type="scan",
                amount=50,
                balance_after=50,
                description="Welcome bonus — 50 free scan credits",
            ))

    db.commit()
    db.refresh(client)

    return ClientResponse(id=str(client.id), name=client.name, brand=client.brand, apps=client.apps)


# ── Brand promotion settings ────────────────────────────────────────────
# These endpoints back the Workspace Settings → "My primary brands" UI.
# primary_brand_ids is the cross-scan default for content-gen promotion
# (FAQ / Article generation). Resolution chain documented in
# worker/services/brand_resolver.py.

class PromotionUpdate(BaseModel):
    primary_brand_ids: list[str]  # ordered, [0] = lead brand


@router.get("/{client_id}/promotion")
async def get_client_promotion(client_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return current promotion settings + all client brands + auto-detected suggestions.

    Brands are nested : children (gammes / product lines linked via parent_id)
    appear inside their parent's `children` array rather than as flat entries.
    Mirrors the GET /scans/{id}/brands shape so the workspace settings UI can
    reuse the same hierarchical drag-drop pattern (Gate 3).
    """
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    primary_ids = list(client.primary_brand_ids or [])
    primary_id_set = {str(bid) for bid in primary_ids}

    all_brands = (
        db.query(ClientBrand)
        .filter(ClientBrand.client_id == client_id)
        .order_by(ClientBrand.name)
        .all()
    )

    # Auto-detected my_brand brands (across all scans for this client)
    suggested_rows = (
        db.query(ScanBrandClassification.brand_id)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(
            ClientBrand.client_id == client_id,
            ScanBrandClassification.classification == "my_brand",
        )
        .distinct()
        .all()
    )
    suggested_id_set = {str(r.brand_id) for r in suggested_rows}

    by_id = {str(b.id): b for b in all_brands}

    def _to_dict(b: ClientBrand) -> dict:
        return {
            "id": str(b.id),
            "name": b.name,
            "domain": b.domain,
            "parent_id": str(b.parent_id) if b.parent_id else None,
            "is_primary": str(b.id) in primary_id_set,
            "is_suggested": str(b.id) in suggested_id_set,
            "children": [],
        }

    # Children-by-parent map (only for nesting under a primary OR another root).
    # Children are not added to the root flat list; they live inside parent.children.
    child_ids_to_skip: set[str] = set()
    children_by_parent: dict[str, list[dict]] = {}
    for b in all_brands:
        if not b.parent_id:
            continue
        pid_str = str(b.parent_id)
        children_by_parent.setdefault(pid_str, []).append(_to_dict(b))
        child_ids_to_skip.add(str(b.id))

    # Emit roots in deterministic order : primary brands first (in primary_ids
    # order = drag-reorder authority), then everything else alphabetical.
    serialized: list[dict] = []
    for bid in primary_ids:
        bid_str = str(bid)
        b = by_id.get(bid_str)
        if not b or bid_str in child_ids_to_skip:
            continue
        d = _to_dict(b)
        d["children"] = children_by_parent.get(bid_str, [])
        serialized.append(d)
    for b in all_brands:
        bid_str = str(b.id)
        if bid_str in primary_id_set:
            continue
        if bid_str in child_ids_to_skip:
            continue
        d = _to_dict(b)
        d["children"] = children_by_parent.get(bid_str, [])
        serialized.append(d)

    return {
        "primary_brand_ids": [str(bid) for bid in primary_ids],
        "all_brands": serialized,
    }


class BrandParentUpdate(BaseModel):
    parent_id: str | None = None  # null = detach (top-level)


@router.patch("/{client_id}/brands/{brand_id}/parent")
async def update_brand_parent(client_id: str, brand_id: str, req: BrandParentUpdate,
                              user=Depends(get_current_user),
                              db: Session = Depends(get_db)):
    """Set or clear the parent_id on a client_brand.

    Used by the workspace-settings drag-drop UI : when the user drops a brand
    onto a primary brand, we PATCH parent_id; when they detach a child to the
    Available column, we PATCH parent_id=NULL.

    Same brand graph the per-scan classifier (Gate 3) uses — a brand has at
    most one parent, children can't have grand-children (enforced here).
    """
    _check_client_access(client_id, user, db)
    brand = (
        db.query(ClientBrand)
        .filter(ClientBrand.id == brand_id, ClientBrand.client_id == client_id)
        .first()
    )
    if not brand:
        raise HTTPException(404, "Brand not found")

    new_parent_id = req.parent_id
    if new_parent_id is None or new_parent_id == "":
        brand.parent_id = None
        db.commit()
        return {"ok": True, "brand_id": str(brand.id), "parent_id": None}

    if new_parent_id == brand_id:
        raise HTTPException(400, "A brand cannot be its own parent")

    parent = (
        db.query(ClientBrand)
        .filter(ClientBrand.id == new_parent_id, ClientBrand.client_id == client_id)
        .first()
    )
    if not parent:
        raise HTTPException(400, "Parent brand not found in this client")

    # Prevent grand-children : if `parent` is itself a child, refuse.
    if parent.parent_id is not None:
        raise HTTPException(400, {
            "error": "grand_child_disallowed",
            "message": "Can't make a gamme a parent. Attach to the top-level brand instead.",
        })

    brand.parent_id = parent.id
    db.commit()
    return {
        "ok": True, "brand_id": str(brand.id),
        "parent_id": str(parent.id), "parent_name": parent.name,
    }


@router.put("/{client_id}/promotion")
async def update_client_promotion(client_id: str, req: PromotionUpdate,
                                  user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Replace the client's primary_brand_ids (workspace default for content gen).

    Cross-scan consistency : a brand the user just declared as their OWN
    (workspace level) must never be classified as `competitor` on any
    existing scan — that would skew visibility metrics (Avène mentions
    counted as competitor mentions) AND poison generated content (the
    promote-vs-avoid resolver would tell the LLM to actively NOT recommend
    the brand). Sweep all of this client's ScanBrandClassification rows
    and flip any matching `competitor` to `my_brand`.

    Inverse direction (brand removed from primary_brand_ids) is left alone
    : per-scan classifications stay so the user can keep treating a brand
    as my_brand on a specific scan without re-adding it to the workspace
    primaries (e.g. a one-off audit of a no-longer-owned subsidiary).
    """
    from models import Scan
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    valid_ids = {
        str(b.id) for b in
        db.query(ClientBrand).filter(ClientBrand.client_id == client_id).all()
    }
    invalid = [bid for bid in req.primary_brand_ids if bid not in valid_ids]
    if invalid:
        raise HTTPException(400, f"Brand IDs not in this client: {invalid[:3]}")

    try:
        new_primary_uuids = [UUID(bid) for bid in req.primary_brand_ids]
    except ValueError as e:
        raise HTTPException(400, f"Malformed UUID: {e}")

    old_primary_ids = {str(b) for b in (client.primary_brand_ids or [])}
    new_primary_ids = {str(b) for b in new_primary_uuids}
    added = new_primary_ids - old_primary_ids

    client.primary_brand_ids = new_primary_uuids

    # Sweep competitor classifications for newly-added primaries — only
    # touch rows that are currently `competitor`, never demote my_brand
    # or focus rows (idempotent on re-clicks).
    flipped = 0
    if added:
        scan_ids_for_client = [
            s.id for s in db.query(Scan).filter(Scan.client_id == UUID(client_id)).all()
        ]
        if scan_ids_for_client:
            sbc_rows = db.query(ScanBrandClassification).filter(
                ScanBrandClassification.brand_id.in_([UUID(b) for b in added]),
                ScanBrandClassification.scan_id.in_(scan_ids_for_client),
                ScanBrandClassification.classification == "competitor",
            ).all()
            for sbc in sbc_rows:
                sbc.classification = "my_brand"
                sbc.classified_by = "primary_brand_sync"
                sbc.source = "primary_brand_sync"
                sbc.updated_at = datetime.utcnow()
                flipped += 1

    # Phase BB sync : when a brand is added to primary_brand_ids and has no
    # brief yet, auto-enqueue generate_brand_brief. Idempotent — the worker
    # handler checks `brand.brief IS NULL` before regenerating, and the API
    # endpoint hard-caps regens. Failure here is best-effort (rollback the
    # job insert but keep the primary_brand_ids change).
    enqueued = 0
    if added:
        try:
            for bid_str in added:
                bid_uuid = UUID(bid_str)
                brand = db.query(ClientBrand).filter(ClientBrand.id == bid_uuid).first()
                if brand and (brand.brief is None) and \
                        (int(brand.brief_generations_count or 0) < MAX_BRAND_BRIEF_GENERATIONS):
                    # De-dup against in-flight jobs for the same brand
                    in_flight = (
                        db.query(Job)
                        .filter(
                            Job.client_id == client_id,
                            Job.job_type == "generate_brand_brief",
                            Job.status.in_(["pending", "running"]),
                            Job.payload["brand_id"].astext == bid_str,
                        )
                        .first()
                    )
                    if in_flight:
                        continue
                    db.add(Job(
                        client_id=UUID(client_id),
                        job_type="generate_brand_brief",
                        status="pending",
                        payload={"brand_id": bid_str},
                        max_attempts=2,
                    ))
                    enqueued += 1
            if enqueued:
                db.commit()
        except Exception:
            db.rollback()

    db.commit()

    return {
        "ok": True,
        "primary_brand_ids": req.primary_brand_ids,
        "count": len(req.primary_brand_ids),
        "competitor_to_my_brand_flipped": flipped,
        "brand_briefs_enqueued": enqueued,
    }


# ── Workspace brief (client.apps.client_brief) ──────────────────────────
# The workspace brief describes the COMPANY (not any single scanned domain).
# It's injected into FAQ + article generation so the output sounds like the
# user's brand even when generated from a competitor scan opportunity.

class BriefUpdate(BaseModel):
    brief: dict  # full brief object — caller is responsible for shape


@router.get("/{client_id}/brief")
async def get_client_brief(client_id: str, user=Depends(get_current_user),
                           db: Session = Depends(get_db)):
    """Return the workspace brief + regen budget, or null brief if not yet generated."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")
    apps = client.apps or {}
    brief = apps.get("client_brief")
    used = int((brief or {}).get("generations_count") or 0)
    return {
        "brief": brief,
        "generations_used": used,
        "generations_cap": MAX_CLIENT_BRIEF_GENERATIONS,
        "can_regenerate": used < MAX_CLIENT_BRIEF_GENERATIONS,
    }


# Hard cap on per-client workspace brief regenerations. Each call fires
# OpenAI web_search (~$0.02-0.05). Workspace brief is regenerated rarely
# in practice (1-2x to seed, then user edits). 5 leaves room for LLM
# garbage on a brand-new workspace without enabling spam.
# See feedback_cap_user_triggered_llm_ops.
MAX_CLIENT_BRIEF_GENERATIONS = 5


@router.post("/{client_id}/brief/generate")
@limiter.limit("5/minute")
async def generate_client_brief(request: Request, client_id: str,
                                user=Depends(get_current_user),
                                db: Session = Depends(get_db)):
    """Enqueue a generate_client_brief worker job. Returns the job id for polling."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    # Block regeneration if user has explicitly edited — they should DELETE
    # via PUT (with edited_by_user=false) before regenerating.
    apps = client.apps or {}
    existing = apps.get("client_brief") or {}
    if existing.get("edited_by_user"):
        raise HTTPException(
            409,
            "Brief has been manually edited — delete edited_by_user via PUT before regenerating",
        )

    # Hard cap : 429 once the workspace brief has burned the regen budget.
    # Counter incremented in worker on success (failed runs don't count).
    used = int(existing.get("generations_count") or 0)
    if used >= MAX_CLIENT_BRIEF_GENERATIONS:
        raise HTTPException(429, {
            "error": "client_brief_regen_cap_reached",
            "message": f"Workspace brief has been generated {used} times "
                       f"(max {MAX_CLIENT_BRIEF_GENERATIONS}). Edit the brief manually "
                       f"in workspace settings — further regenerations are blocked.",
            "generations_used": used,
            "cap": MAX_CLIENT_BRIEF_GENERATIONS,
        })

    # Avoid duplicate in-flight jobs
    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "generate_client_brief",
            Job.status.in_(["pending", "running"]),
        )
        .first()
    )
    if in_flight:
        return {"ok": True, "job_id": str(in_flight.id), "status": in_flight.status,
                "message": "Already in flight",
                "generations_used": used, "cap": MAX_CLIENT_BRIEF_GENERATIONS}

    job = Job(
        client_id=client_id,
        job_type="generate_client_brief",
        status="pending",
        payload={"client_id": client_id},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"ok": True, "job_id": str(job.id), "status": "pending",
            "generations_used": used, "cap": MAX_CLIENT_BRIEF_GENERATIONS}


@router.put("/{client_id}/brief")
async def update_client_brief(client_id: str, req: BriefUpdate,
                              user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Replace the workspace brief with a manual edit (sets edited_by_user=true).

    Pass an explicit `edited_by_user: false` inside `brief` to re-allow regeneration.
    """
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    new_brief = dict(req.brief)
    # Default edited_by_user=true unless caller explicitly opts out (e.g., to clear flag)
    if "edited_by_user" not in new_brief:
        new_brief["edited_by_user"] = True

    apps = dict(client.apps or {})
    apps["client_brief"] = new_brief
    client.apps = apps
    flag_modified(client, "apps")
    db.commit()
    return {"ok": True, "edited_by_user": new_brief.get("edited_by_user", True)}


# ── Per-brand brief (client_brands.brief JSONB) ───────────────────────────
# Phase BB — surcharges the workspace brief per-primary-brand. Same regen
# cap pattern as the workspace brief but tighter ($0.05/run × N brands).
# See worker/handlers/generate_brand_brief.py + project_phase_brand_briefs.md.

MAX_BRAND_BRIEF_GENERATIONS = 3  # mirror worker/handlers/generate_brand_brief.py


def _resolve_brand_for_client(client_id: str, brand_id: str, db: Session) -> ClientBrand:
    try:
        UUID(client_id); UUID(brand_id)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, "Invalid client_id or brand_id") from e
    brand = (
        db.query(ClientBrand)
        .filter(ClientBrand.id == brand_id, ClientBrand.client_id == client_id)
        .first()
    )
    if not brand:
        raise HTTPException(404, "Brand not found in this client")
    return brand


@router.get("/{client_id}/brands/{brand_id}/brief")
async def get_brand_brief(client_id: str, brand_id: str,
                          user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the per-brand brief + regen budget. ``brief`` is null when not generated yet."""
    _check_client_access(client_id, user, db)
    brand = _resolve_brand_for_client(client_id, brand_id, db)
    used = int(brand.brief_generations_count or 0)
    return {
        "brief": brand.brief,
        "brand_id": str(brand.id),
        "brand_name": brand.name,
        "generated_at": brand.brief_generated_at.isoformat() + "Z" if brand.brief_generated_at else None,
        "generations_used": used,
        "generations_cap": MAX_BRAND_BRIEF_GENERATIONS,
        "can_regenerate": used < MAX_BRAND_BRIEF_GENERATIONS and not (brand.brief or {}).get("edited_by_user"),
    }


@router.post("/{client_id}/brands/{brand_id}/brief/generate")
@limiter.limit("5/minute")
async def generate_brand_brief(request: Request, client_id: str, brand_id: str,
                               user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Enqueue a generate_brand_brief worker job. Returns the job id for polling."""
    _check_client_access(client_id, user, db)
    brand = _resolve_brand_for_client(client_id, brand_id, db)

    existing = brand.brief or {}
    if existing.get("edited_by_user"):
        raise HTTPException(
            409,
            "Brand brief has been manually edited — clear edited_by_user via PATCH before regenerating",
        )

    used = int(brand.brief_generations_count or 0)
    if used >= MAX_BRAND_BRIEF_GENERATIONS:
        raise HTTPException(429, {
            "error": "brand_brief_regen_cap_reached",
            "message": f"Brand brief regenerated {used} times "
                       f"(max {MAX_BRAND_BRIEF_GENERATIONS}). Edit the brief manually below — "
                       f"further regenerations are blocked.",
            "generations_used": used,
            "cap": MAX_BRAND_BRIEF_GENERATIONS,
        })

    # De-dup in-flight jobs for the same brand
    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "generate_brand_brief",
            Job.status.in_(["pending", "running"]),
            # Job.payload JSONB filter — Postgres @> operator
            Job.payload["brand_id"].astext == str(brand.id),
        )
        .first()
    )
    if in_flight:
        return {"ok": True, "job_id": str(in_flight.id), "status": in_flight.status,
                "message": "Already in flight",
                "generations_used": used, "cap": MAX_BRAND_BRIEF_GENERATIONS}

    job = Job(
        client_id=client_id,
        job_type="generate_brand_brief",
        status="pending",
        payload={"brand_id": str(brand.id)},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"ok": True, "job_id": str(job.id), "status": "pending",
            "generations_used": used, "cap": MAX_BRAND_BRIEF_GENERATIONS}


class BrandBriefPatch(BaseModel):
    brief: dict  # full brief object — Pydantic shape enforced at worker boundary


@router.patch("/{client_id}/brands/{brand_id}/brief")
async def update_brand_brief(client_id: str, brand_id: str, req: BrandBriefPatch,
                             user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Replace the per-brand brief with a manual edit (sets edited_by_user=true by default).

    Pass an explicit ``edited_by_user: false`` inside ``brief`` to re-allow regeneration.
    """
    _check_client_access(client_id, user, db)
    brand = _resolve_brand_for_client(client_id, brand_id, db)

    new_brief = dict(req.brief)
    # Default edited_by_user=true unless caller explicitly opts out (e.g., to clear flag)
    if "edited_by_user" not in new_brief:
        new_brief["edited_by_user"] = True
    # Ensure name stays in sync with the row (in case user blanked it)
    if not new_brief.get("name"):
        new_brief["name"] = brand.name

    brand.brief = new_brief
    flag_modified(brand, "brief")
    db.commit()
    return {"ok": True,
            "edited_by_user": new_brief.get("edited_by_user", True),
            "brand_id": str(brand.id)}


# ─── Trust sources (per-client authoritative reference domains) ──────────
# Discovery is automatically chained from generate_client_brief on success,
# but these endpoints expose manual control for: seeding existing clients
# whose brief predates the trust-sources feature, refreshing when the
# discovered list looks off, and future Settings UI integration.


@router.get("/{client_id}/trust-sources")
async def get_trust_sources(client_id: str, user=Depends(get_current_user),
                            db: Session = Depends(get_db)):
    """Read the current trust_sources payload for a client.

    Returns the persisted structure verbatim (or an empty stub if discovery
    has never run). The Settings UI uses this to render the list + last
    refresh date + refresh button.
    """
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    payload = (client.apps or {}).get("trust_sources") or {}
    return {
        "ok": True,
        "trust_sources": {
            "domains": payload.get("domains") or [],
            "details": payload.get("details") or [],
            "extra_domains": payload.get("extra_domains") or [],
            "industry_text": payload.get("industry_text") or "",
            "discovered_at": payload.get("discovered_at"),
            "sources_count": payload.get("sources_count") or 0,
        },
    }


@router.post("/{client_id}/trust-sources/discover")
@limiter.limit("3/minute")
async def discover_trust_sources(request: Request, client_id: str,
                                  force: bool = False,
                                  user=Depends(get_current_user),
                                  db: Session = Depends(get_db)):
    """Enqueue a discover_trust_sources worker job.

    Requires `client_brief.industry` to be set (otherwise the worker no-ops).
    Idempotent on the worker side : returns 'fresh' if the cached payload is
    still within TTL and industry hasn't changed. Pass ?force=true to bypass.
    """
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    apps = client.apps or {}
    brief = apps.get("client_brief") or {}
    if not (brief.get("industry") or "").strip():
        raise HTTPException(
            409,
            {
                "error": "missing_industry",
                "message": "Trust source discovery needs a workspace brief with "
                           "an `industry` field. Generate the workspace brief first.",
            },
        )

    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "discover_trust_sources",
            Job.status.in_(["pending", "running"]),
        )
        .first()
    )
    if in_flight:
        return {
            "ok": True, "job_id": str(in_flight.id), "status": in_flight.status,
            "message": "Already in flight",
        }

    job = Job(
        client_id=client_id,
        job_type="discover_trust_sources",
        status="pending",
        payload={"client_id": client_id, "force": bool(force)},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"ok": True, "job_id": str(job.id), "status": "pending"}


# ─── extra_domains : user-managed prefer-hint extension slot ─────────────
# These let the user widen the soft prefer-hint list when the discovered
# domains miss something they trust (e.g., an internal scientific publisher,
# a niche industry portal). HARD denylist (competitor) is unaffected — these
# only feed the SOFT prefer-hint list returned by
# `get_trust_sources_for_client`. The discover handler carries them forward
# on every refresh.

class ExtraDomainBody(BaseModel):
    domain: str


def _normalize_extra_domain(raw: str) -> str:
    """Mirror of worker.services.trust_sources._normalize_domain — kept local
    to avoid importing the worker module from the API container."""
    import re as _re
    if not raw or not isinstance(raw, str):
        return ""
    nd = _re.sub(r"^https?://", "", raw.strip().lower())
    if nd.startswith("www."):
        nd = nd[4:]
    nd = nd.split("/", 1)[0].strip().rstrip(".")
    if "." not in nd or len(nd) > 253:
        return ""
    # Disallow control chars / spaces — defensive against pasted junk
    if _re.search(r"[\s<>\"']", nd):
        return ""
    return nd


@router.post("/{client_id}/trust-sources/extra-domains")
async def add_trust_source_extra(client_id: str, body: ExtraDomainBody,
                                  user=Depends(get_current_user),
                                  db: Session = Depends(get_db)):
    """Append a user-managed domain to the prefer-hint list."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    domain = _normalize_extra_domain(body.domain)
    if not domain:
        raise HTTPException(400, {
            "error": "invalid_domain",
            "message": "Domain must be a bare hostname (e.g. example.com).",
        })

    apps = dict(client.apps or {})
    trust = dict(apps.get("trust_sources") or {})
    extras = list(trust.get("extra_domains") or [])
    discovered = {(d or "").lower() for d in (trust.get("domains") or [])}

    if domain in discovered:
        raise HTTPException(409, {
            "error": "already_discovered",
            "message": f"{domain} is already in the discovered list — no need to add it manually.",
        })
    if domain in extras:
        raise HTTPException(409, {
            "error": "duplicate",
            "message": f"{domain} is already in your extras.",
        })

    extras.append(domain)
    trust["extra_domains"] = extras
    apps["trust_sources"] = trust
    client.apps = apps
    flag_modified(client, "apps")
    db.commit()
    return {"ok": True, "extra_domains": extras}


# ─── Sitemap-index pages : per-brand crawl + manual URL escape hatch ──────
# Phase D. Lets the user :
#   - See the pages crawled from a brand's sitemap.xml + how many embedded
#   - Trigger a manual refresh (re-runs crawl_brand_sitemap, which chains
#     fetch_brand_pages and — Day 3 onward — embed_brand_pages)
#   - Add a page manually when the sitemap is missing / incomplete. Manual
#     rows enter the same fetch -> embed pipeline but are exempt from the
#     sitemap-diff mark_gone branch (see migration 027).

def _brand_for_client(brand_id: str, client_id: str, db: Session) -> ClientBrand:
    """Resolve a brand belonging to a client, 404 otherwise.

    Centralizes the access pattern : `_check_client_access` has already
    enforced the user can touch this client_id, so a brand mismatch is a
    not-found rather than a permission error.
    """
    brand = (
        db.query(ClientBrand)
        .filter(ClientBrand.id == brand_id, ClientBrand.client_id == client_id)
        .first()
    )
    if not brand:
        raise HTTPException(404, "Brand not found for this client")
    return brand


def _normalize_brand_host(domain: str) -> str:
    """Strip scheme/path/www/trailing slash, lowercase."""
    if not domain:
        return ""
    import re as _re
    d = _re.sub(r"^https?://", "", (domain or "").strip().lower())
    if d.startswith("www."):
        d = d[4:]
    return d.split("/", 1)[0].strip().rstrip(".")


def _validate_manual_page_url(raw_url: str, brand: ClientBrand) -> str:
    """Validate that a user-supplied URL is acceptable for this brand.

    Rules :
      - Must be https:// (we don't crawl http in v1 — sitemap_crawler tries
        https only too, so this keeps the pipeline coherent)
      - Hostname must match the brand's registered domain, with or without
        the www. prefix on either side. This prevents an editor from
        seeding URLs that point at unrelated sites (would pollute the
        matcher corpus + leak the user's brand index across competitors)
      - Must parse as a real URL — bare hostnames rejected
      - Returns the canonical-form URL (no trailing slash on bare host,
        scheme normalized to https, fragment stripped — fragments duplicate
        the parent page in the index)

    Raises HTTPException(400/422) with a structured error payload on
    failure so the UI can render a precise message.
    """
    if not raw_url or not isinstance(raw_url, str):
        raise HTTPException(400, {"error": "invalid_url", "message": "URL is required."})
    raw_url = raw_url.strip()
    if not raw_url:
        raise HTTPException(400, {"error": "invalid_url", "message": "URL is required."})

    from urllib.parse import urlparse, urlunparse
    try:
        parsed = urlparse(raw_url)
    except ValueError:
        raise HTTPException(400, {"error": "invalid_url", "message": "URL is malformed."})

    if parsed.scheme not in ("https",):
        raise HTTPException(
            422,
            {"error": "scheme_unsupported",
             "message": "URL must start with https://. http is not crawled in v1."},
        )
    if not parsed.hostname:
        raise HTTPException(400, {"error": "invalid_url", "message": "URL must include a host."})

    brand_host = _normalize_brand_host(brand.domain or "")
    if not brand_host:
        raise HTTPException(
            422,
            {"error": "brand_domain_missing",
             "message": "This brand has no registered domain. Set the brand domain in Settings → My primary brands first."},
        )
    url_host = parsed.hostname.lower()
    if url_host.startswith("www."):
        url_host = url_host[4:]
    if url_host != brand_host:
        raise HTTPException(
            422,
            {"error": "host_mismatch",
             "message": f"URL host must be {brand_host} (got {parsed.hostname}). Manual pages can only point to the brand's own domain."},
        )

    # Strip fragment + normalize trailing slash on path-empty URLs to keep
    # the dedup tight against the sitemap rows.
    clean = parsed._replace(fragment="")
    if clean.path == "":
        clean = clean._replace(path="/")
    return urlunparse(clean)


@router.get("/{client_id}/brands/{brand_id}/pages")
async def list_brand_pages(
    client_id: str, brand_id: str,
    source: str | None = None,   # 'sitemap' | 'manual' | 'all' (default all)
    status: str | None = None,
    q: str | None = None,         # ILIKE search on url + title
    sort: str = "inlinks_desc",   # inlinks_desc | status | first_seen | last_embedded
    offset: int = 0,
    limit: int = 50,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List indexed pages for a brand + summary stats.

    Default behavior changed in 2026-05-14 (Phase D detail view) :
      - `source=None` (or 'all') returns every page, not just manual ones
      - Pagination via offset + limit (default 50, max 200)
      - Optional ILIKE search on url/title, plus 4-mode sort

    Drives both :
      - the Settings sitemaps card initial render (passes source=manual,
        limit=200 to fetch the manual-pages list)
      - the "View pages" expand panel (passes source=all + sort + page nav)
    """
    _check_client_access(client_id, user, db)
    brand = _brand_for_client(brand_id, client_id, db)

    # Aggregate stats across the whole brand (no filter), one round-trip
    from sqlalchemy import func, or_
    stats_rows = (
        db.query(
            ClientBrandPage.status,
            ClientBrandPage.source,
            func.count(ClientBrandPage.id).label("n"),
            func.max(ClientBrandPage.last_seen_at).label("last_seen"),
            func.max(ClientBrandPage.last_crawled_at).label("last_crawled"),
        )
        .filter(ClientBrandPage.client_brand_id == brand_id)
        .group_by(ClientBrandPage.status, ClientBrandPage.source)
        .all()
    )
    by_status: dict[str, int] = {}
    by_source: dict[str, int] = {}
    total = 0
    last_seen_max = None
    last_crawled_max = None
    for r in stats_rows:
        total += r.n
        by_status[r.status] = by_status.get(r.status, 0) + r.n
        by_source[r.source] = by_source.get(r.source, 0) + r.n
        if r.last_seen and (last_seen_max is None or r.last_seen > last_seen_max):
            last_seen_max = r.last_seen
        if r.last_crawled and (last_crawled_max is None or r.last_crawled > last_crawled_max):
            last_crawled_max = r.last_crawled

    # Active crawl/fetch/embed job — drives the "Refreshing..." pill
    in_flight_crawl = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type.in_(("crawl_brand_sitemap", "fetch_brand_pages", "embed_brand_pages")),
            Job.status.in_(("pending", "running")),
            Job.payload["client_brand_id"].astext == str(brand_id),
        )
        .order_by(Job.created_at.desc())
        .first()
    )

    # Pages query — start from the filter base, then apply paging.
    base_q = (
        db.query(ClientBrandPage)
        .filter(ClientBrandPage.client_brand_id == brand_id)
    )
    if source in ("manual", "sitemap"):
        base_q = base_q.filter(ClientBrandPage.source == source)
    if status:
        base_q = base_q.filter(ClientBrandPage.status == status)
    if q:
        pattern = f"%{q.strip()}%"
        base_q = base_q.filter(
            or_(
                ClientBrandPage.url.ilike(pattern),
                ClientBrandPage.title.ilike(pattern),
            )
        )

    filtered_total = base_q.with_entities(func.count(ClientBrandPage.id)).scalar() or 0

    # Sort
    if sort == "status":
        base_q = base_q.order_by(
            ClientBrandPage.status.asc(),
            ClientBrandPage.internal_inlink_count.desc(),
        )
    elif sort == "first_seen":
        base_q = base_q.order_by(ClientBrandPage.first_seen_at.asc())
    elif sort == "last_embedded":
        base_q = base_q.order_by(ClientBrandPage.last_embedded_at.desc().nullslast())
    else:  # 'inlinks_desc' default
        base_q = base_q.order_by(
            ClientBrandPage.internal_inlink_count.desc(),
            ClientBrandPage.first_seen_at.asc(),
        )

    base_q = base_q.offset(max(0, int(offset))).limit(max(1, min(int(limit), 200)))
    page_rows = base_q.all()

    return {
        "ok": True,
        "brand": {
            "id": str(brand.id), "name": brand.name, "domain": brand.domain,
            "locale_path_prefix": brand.locale_path_prefix,
            "sitemap_urls_override": list(brand.sitemap_urls_override or []),
        },
        "stats": {
            "total": total,
            "by_status": by_status,
            "by_source": by_source,
            "last_seen_at": last_seen_max.isoformat() if last_seen_max else None,
            "last_crawled_at": last_crawled_max.isoformat() if last_crawled_max else None,
        },
        "in_flight_job": {
            "id": str(in_flight_crawl.id),
            "type": in_flight_crawl.job_type,
            "status": in_flight_crawl.status,
        } if in_flight_crawl else None,
        "page_info": {
            "total": int(filtered_total),
            "offset": int(offset),
            "limit": int(limit),
            "returned": len(page_rows),
            "has_more": int(offset) + len(page_rows) < int(filtered_total),
        },
        "pages": [
            {
                "id": str(p.id),
                "url": p.url,
                "title": p.title,
                "status": p.status,
                "source": p.source,
                "fetch_error": p.fetch_error,
                "lang": p.lang,
                "internal_inlink_count": int(p.internal_inlink_count or 0),
                "lastmod": p.lastmod.isoformat() if p.lastmod else None,
                "last_crawled_at": p.last_crawled_at.isoformat() if p.last_crawled_at else None,
                "last_embedded_at": p.last_embedded_at.isoformat() if p.last_embedded_at else None,
            }
            for p in page_rows
        ],
    }


class ManualPageUrlBody(BaseModel):
    url: str


@router.post("/{client_id}/brands/{brand_id}/pages/manual")
@limiter.limit("20/minute")
async def add_manual_page(
    request: Request, client_id: str, brand_id: str, body: ManualPageUrlBody,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a URL to the brand's page index. Goes into the fetch -> embed
    pipeline like any sitemap-discovered row, but with source='manual'
    so it survives sitemap diffs.

    Enqueues a fetch_brand_pages job if one isn't already in flight for
    this brand — same in-flight pattern as trust-sources discovery.
    """
    _check_client_access(client_id, user, db)
    brand = _brand_for_client(brand_id, client_id, db)
    canonical_url = _validate_manual_page_url(body.url, brand)

    existing = (
        db.query(ClientBrandPage)
        .filter(
            ClientBrandPage.client_brand_id == brand_id,
            ClientBrandPage.url == canonical_url,
        )
        .first()
    )
    if existing:
        # If it was previously sitemap-discovered then went gone, restore
        # via the user's manual intent. Otherwise 409 — the row already
        # exists and the user can see it in the list.
        if existing.status == "gone":
            existing.status = "pending_fetch"
            existing.gone_since = None
            existing.fetch_error = None
            existing.fetch_retry_count = 0
            existing.source = "manual"
            db.commit()
            _maybe_enqueue_fetch(client_id, brand_id, db)
            return {
                "ok": True, "url": canonical_url, "status": existing.status,
                "source": "manual", "restored": True,
            }
        raise HTTPException(409, {
            "error": "duplicate",
            "message": f"This URL is already indexed (status={existing.status}, source={existing.source}).",
        })

    row = ClientBrandPage(
        client_brand_id=brand_id,
        url=canonical_url,
        status="pending_fetch",
        source="manual",
    )
    db.add(row)
    db.commit()

    _maybe_enqueue_fetch(client_id, brand_id, db)

    return {
        "ok": True, "url": canonical_url, "status": "pending_fetch",
        "source": "manual", "restored": False,
    }


@router.delete("/{client_id}/brands/{brand_id}/pages/manual")
async def delete_manual_page(
    client_id: str, brand_id: str, body: ManualPageUrlBody,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Hard-delete a manual page. Sitemap-discovered rows cannot be
    removed this way (would be re-inserted by the next crawl) — they
    only leave the index via the natural gone -> purge_stale_pages flow.
    """
    _check_client_access(client_id, user, db)
    _brand_for_client(brand_id, client_id, db)
    target_url = (body.url or "").strip()
    if not target_url:
        raise HTTPException(400, "URL is required.")

    row = (
        db.query(ClientBrandPage)
        .filter(
            ClientBrandPage.client_brand_id == brand_id,
            ClientBrandPage.url == target_url,
        )
        .first()
    )
    if not row:
        raise HTTPException(404, "Page not found in this brand's index.")
    if row.source != "manual":
        raise HTTPException(
            409,
            {"error": "not_manual",
             "message": "Only manually-added pages can be deleted. Sitemap-discovered "
                        "pages must be removed from your sitemap.xml or wait for the "
                        "30-day gone-purge cycle."},
        )

    db.delete(row)
    db.commit()
    return {"ok": True, "deleted": target_url}


def _maybe_enqueue_fetch(client_id: str, brand_id: str, db: Session) -> str | None:
    """Enqueue a fetch_brand_pages job iff none is already pending/running.

    Mirrors the discover_trust_sources in-flight protection — a single
    pending fetch will pick up every pending_fetch row at runtime, so
    queueing more is wasted work."""
    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "fetch_brand_pages",
            Job.status.in_(("pending", "running")),
            Job.payload["client_brand_id"].astext == str(brand_id),
        )
        .first()
    )
    if in_flight:
        return str(in_flight.id)
    job = Job(
        client_id=client_id,
        job_type="fetch_brand_pages",
        status="pending",
        payload={"client_brand_id": str(brand_id)},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    return str(job.id)


@router.post("/{client_id}/brands/{brand_id}/sitemap/refresh")
@limiter.limit("3/minute")
async def refresh_brand_sitemap(
    request: Request, client_id: str, brand_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Manually trigger a sitemap re-crawl for one brand. Enqueues
    crawl_brand_sitemap, which chains fetch_brand_pages on completion
    (Day 2). In-flight protection : if a crawl is already pending /
    running for this brand we return its job id."""
    _check_client_access(client_id, user, db)
    brand = _brand_for_client(brand_id, client_id, db)

    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "crawl_brand_sitemap",
            Job.status.in_(("pending", "running")),
            Job.payload["client_brand_id"].astext == str(brand_id),
        )
        .first()
    )
    if in_flight:
        return {
            "ok": True, "job_id": str(in_flight.id),
            "status": in_flight.status, "message": "Already in flight",
        }

    if not (brand.domain or "").strip():
        raise HTTPException(
            422,
            {"error": "brand_domain_missing",
             "message": "This brand has no registered domain — cannot crawl. "
                        "Set it in Settings → My primary brands first."},
        )

    job = Job(
        client_id=client_id,
        job_type="crawl_brand_sitemap",
        status="pending",
        payload={"client_brand_id": str(brand_id)},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"ok": True, "job_id": str(job.id), "status": "pending"}


# ─── Per-brand sitemap config : override URLs + locale path filter ────────
# Phase D (migration 028). For multi-locale brands (Ducray etc.) the user
# can set a `locale_path_prefix` (e.g. '/fr-fr/') to scope the crawl to one
# locale, OR provide an explicit `sitemap_urls_override` array that bypasses
# auto-discovery entirely. The two compose : override wins for discovery,
# locale_prefix is always applied as a post-filter.

class SitemapConfigBody(BaseModel):
    sitemap_urls_override: list[str] | None = None
    locale_path_prefix: str | None = None


def _validate_sitemap_config(
    body: SitemapConfigBody, brand: ClientBrand,
) -> tuple[list[str], str | None]:
    """Normalize + validate the user-supplied sitemap config.

    Override URLs must :
      - be https (we don't crawl http in v1)
      - point at the brand's own domain (anti-cross-pollution, same rule
        as manual page URLs in /pages/manual)
      - de-duped, capped at 20 (very generous, no realistic site needs
        more)
    Locale path prefix must :
      - start with '/' and be ≤ 32 chars (e.g. '/fr-fr/', '/intl/en/')
      - contain only path-safe chars (letters/digits/dash/slash)
    """
    import re as _re
    from urllib.parse import urlparse

    raw_urls = body.sitemap_urls_override if body.sitemap_urls_override is not None else []
    if not isinstance(raw_urls, list):
        raise HTTPException(400, {"error": "invalid_type", "message": "sitemap_urls_override must be a list"})
    if len(raw_urls) > 20:
        raise HTTPException(422, {"error": "too_many_urls",
                                   "message": "sitemap_urls_override capped at 20 URLs"})

    brand_host = _normalize_brand_host(brand.domain or "")
    cleaned_urls: list[str] = []
    seen: set[str] = set()
    for raw in raw_urls:
        if not isinstance(raw, str):
            continue
        u = raw.strip()
        if not u:
            continue
        try:
            parsed = urlparse(u)
        except ValueError:
            raise HTTPException(400, {"error": "invalid_url", "message": f"Malformed URL: {u}"})
        if parsed.scheme != "https":
            raise HTTPException(422, {"error": "scheme_unsupported",
                                       "message": f"URL must be https: {u}"})
        url_host = (parsed.hostname or "").lower()
        if url_host.startswith("www."):
            url_host = url_host[4:]
        if not brand_host:
            raise HTTPException(422, {"error": "brand_domain_missing",
                                       "message": "Brand has no registered domain; set it first."})
        if url_host != brand_host:
            raise HTTPException(422, {"error": "host_mismatch",
                                       "message": f"URL host must be {brand_host} (got {parsed.hostname})"})
        clean = parsed._replace(fragment="").geturl()
        if clean in seen:
            continue
        seen.add(clean)
        cleaned_urls.append(clean)

    prefix = body.locale_path_prefix
    if prefix is not None:
        prefix = prefix.strip()
        if prefix == "":
            prefix = None
        elif not prefix.startswith("/"):
            raise HTTPException(422, {"error": "invalid_prefix",
                                       "message": "locale_path_prefix must start with '/'"})
        elif len(prefix) > 32:
            raise HTTPException(422, {"error": "invalid_prefix",
                                       "message": "locale_path_prefix capped at 32 chars"})
        elif not _re.match(r"^[A-Za-z0-9/_\-]+$", prefix):
            raise HTTPException(422, {"error": "invalid_prefix",
                                       "message": "locale_path_prefix may only contain letters, digits, '-', '_' and '/'"})

    return cleaned_urls, prefix


@router.get("/{client_id}/brands/{brand_id}/sitemap-config")
async def get_sitemap_config(
    client_id: str, brand_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_client_access(client_id, user, db)
    brand = _brand_for_client(brand_id, client_id, db)
    return {
        "ok": True,
        "brand": {"id": str(brand.id), "name": brand.name, "domain": brand.domain},
        "sitemap_urls_override": list(brand.sitemap_urls_override or []),
        "locale_path_prefix": brand.locale_path_prefix,
    }


@router.put("/{client_id}/brands/{brand_id}/sitemap-config")
async def put_sitemap_config(
    client_id: str, brand_id: str, body: SitemapConfigBody,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the brand's sitemap discovery config.

    Doesn't auto-trigger a refresh — the user clicks Refresh after saving
    if they want to re-crawl with the new config.
    """
    _check_client_access(client_id, user, db)
    brand = _brand_for_client(brand_id, client_id, db)
    cleaned_urls, prefix = _validate_sitemap_config(body, brand)
    brand.sitemap_urls_override = cleaned_urls
    brand.locale_path_prefix = prefix
    db.commit()
    return {
        "ok": True,
        "sitemap_urls_override": cleaned_urls,
        "locale_path_prefix": prefix,
    }


@router.delete("/{client_id}/trust-sources/extra-domains/{domain}")
async def remove_trust_source_extra(client_id: str, domain: str,
                                     user=Depends(get_current_user),
                                     db: Session = Depends(get_db)):
    """Remove a user-managed domain from the prefer-hint list."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    normalized = _normalize_extra_domain(domain)
    if not normalized:
        raise HTTPException(400, "Invalid domain")

    apps = dict(client.apps or {})
    trust = dict(apps.get("trust_sources") or {})
    extras = list(trust.get("extra_domains") or [])
    if normalized not in extras:
        raise HTTPException(404, f"{normalized} not in extras")
    extras = [d for d in extras if d != normalized]
    trust["extra_domains"] = extras
    apps["trust_sources"] = trust
    client.apps = apps
    flag_modified(client, "apps")
    db.commit()
    return {"ok": True, "extra_domains": extras}
