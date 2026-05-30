"""Organizations router — Phase E.C.2.

Surfaces the orgs the current user belongs to + lets them pick an "active"
org via a server-set HttpOnly cookie. The active org id is read back by
`clients.list_clients` (and future cross-client endpoints) to scope the
workspace list when a user is in 2+ orgs.

Why cookie HttpOnly and not localStorage / URL path :
- Consistent with the auth cookie pattern (cf. feedback_auth_cookie_httponly)
  — the server is the source of truth for the session.
- Astro SSR reads it via `Astro.cookies` on every dashboard page render
  without needing a separate API round-trip.
- URL-scoping (`/app/org/{id}/...`) is deferred to C.3 where the cross-client
  dashboard actually benefits from having org in the URL.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import cast, func, Float, Integer
from sqlalchemy.orm import Session

from config import settings
from models import (
    Client, Job, Organization, OrganizationUser, OrgUserClient, Scan, User, get_db,
)
from services.access import list_user_organizations, resolve_active_organization_id
from services.auth_service import get_current_user


router = APIRouter()


ACTIVE_ORG_COOKIE = "active_organization_id"
# 180 days — purely a UI affordance, no security boundary. Re-set silently
# whenever the user switches. Cleared on logout via /api/auth/logout (path=/).
_ACTIVE_ORG_COOKIE_MAX_AGE = 60 * 60 * 24 * 180


class OrgListItem(BaseModel):
    id: str
    name: str
    slug: str | None
    is_personal: bool
    role: str | None         # user's org-level role from organization_users
    member_count: int
    client_count: int
    is_active: bool
    # Sprint S15.4 white-label lite. `branding` JSONB shape :
    #   {"logo_url": "https://...", "display_name": "Agency XYZ", "accent_color": "#FF5733"}
    # All keys optional. Empty dict = no branding override.
    branding: dict = {}


class SetActiveOrgRequest(BaseModel):
    organization_id: str


@router.get("/", response_model=list[OrgListItem])
async def list_organizations(
    active_organization_id: str | None = Cookie(default=None),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the orgs the user can switch between.

    Shape used by the header dropdown :
      - role : owner | admin | member  (org-level, from organization_users)
      - member_count / client_count : computed in a single GROUP BY each, no N+1
      - is_active : true at most once, derived from the active_organization_id
        cookie. False everywhere when the cookie is missing OR points at an
        org the user no longer belongs to (defensive against stale cookies).
    """
    orgs = list_user_organizations(user, db)
    if not orgs:
        return []

    org_ids = [o.id for o in orgs]

    # Org-level role per (user, org). Single round-trip.
    role_rows = (
        db.query(OrganizationUser.organization_id, OrganizationUser.role)
        .filter(
            OrganizationUser.user_id == user.id,
            OrganizationUser.organization_id.in_(org_ids),
        )
        .all()
    )
    role_by_org = {str(r.organization_id): r.role for r in role_rows}

    member_rows = (
        db.query(
            OrganizationUser.organization_id,
            func.count(OrganizationUser.user_id).label("n"),
        )
        .filter(OrganizationUser.organization_id.in_(org_ids))
        .group_by(OrganizationUser.organization_id)
        .all()
    )
    members_by_org = {str(r.organization_id): int(r.n) for r in member_rows}

    client_rows = (
        db.query(
            Client.organization_id,
            func.count(Client.id).label("n"),
        )
        .filter(Client.organization_id.in_(org_ids))
        .group_by(Client.organization_id)
        .all()
    )
    clients_by_org = {str(r.organization_id): int(r.n) for r in client_rows}

    # Match the API-side resolution so the dropdown highlights the same org
    # that `GET /api/clients/` will scope to. Without this, a multi-org user
    # without a cookie would see "no active" in the UI while their clients
    # were silently scoped to the personal org — confusing.
    effective_active_id = resolve_active_organization_id(user, db, active_organization_id)

    return [
        OrgListItem(
            id=str(o.id),
            name=o.name,
            slug=o.slug,
            is_personal=bool(o.is_personal),
            role=role_by_org.get(str(o.id)),
            member_count=members_by_org.get(str(o.id), 0),
            client_count=clients_by_org.get(str(o.id), 0),
            is_active=(effective_active_id is not None and str(o.id) == effective_active_id),
            branding=o.branding or {},
        )
        for o in orgs
    ]


@router.post("/active")
async def set_active_organization(
    req: SetActiveOrgRequest,
    response: Response,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Set the active org cookie after verifying membership.

    Cookie attributes mirror the auth `token` cookie : HttpOnly + Secure +
    SameSite=lax + path=/. HttpOnly because the frontend never needs to read
    the value — it just reloads the page after a successful POST and the
    Astro SSR layer reads the cookie on the next render.
    """
    try:
        org_uuid = uuid.UUID(req.organization_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed organization_id")

    org = db.query(Organization).filter(Organization.id == org_uuid).first()
    if not org:
        raise HTTPException(404, "Organization not found")

    membership = (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org.id,
            OrganizationUser.user_id == user.id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(403, "You are not a member of this organization")

    response.set_cookie(
        ACTIVE_ORG_COOKIE,
        str(org.id),
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=_ACTIVE_ORG_COOKIE_MAX_AGE,
        path="/",
    )
    return {"ok": True, "organization_id": str(org.id)}


@router.delete("/active")
async def clear_active_organization(response: Response, user=Depends(get_current_user)):
    """Clear the active org cookie. Used by the dropdown's "All workspaces"
    affordance (when added in C.3) and by sign-out flows that want a clean slate.
    """
    response.delete_cookie(ACTIVE_ORG_COOKIE, path="/")
    return {"ok": True}


# Phase E.C.3 — cross-client overview for the /app/org page

@router.get("/{org_id}/overview")
async def organization_overview(
    org_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cross-client aggregate of one organization.

    Per-client : last completed scan, scan count, total critical opportunities,
    rough visibility score. Org-level : totals + visibility average across
    own-domain completed scans only (competitor scans skew the score).

    Single-trip SQL — one GROUP BY per metric, no N+1 per client. We use
    Postgres JSONB operators to read `summary->'opportunities'->>'critique'`
    and `summary->>'brand_mention_rate'` directly in aggregates.
    """
    try:
        org_uuid = uuid.UUID(org_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed org_id")

    membership = (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org_uuid,
            OrganizationUser.user_id == user.id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(403, "You are not a member of this organization")

    org = db.query(Organization).filter(Organization.id == org_uuid).first()
    if not org:
        raise HTTPException(404, "Organization not found")

    clients = (
        db.query(Client)
        .filter(Client.organization_id == org_uuid)
        .order_by(Client.name.asc())
        .all()
    )

    client_ids = [c.id for c in clients]
    if not client_ids:
        return {
            "org": {"id": str(org.id), "name": org.name, "is_personal": bool(org.is_personal)},
            "totals": {"clients": 0, "scans_completed": 0, "opportunities_critique": 0, "avg_visibility": None},
            "clients": [],
        }

    # Per-client aggregates : scan counts, last completed, opportunities sum.
    # Visibility avg is computed across completed scans only ; we coerce the
    # JSONB scalar to numeric defensively (legacy summary blobs may be int OR
    # float OR null).
    visibility_expr = func.coalesce(
        cast(Scan.summary["brand_mention_rate"].astext, Float),
        cast(Scan.summary["citation_rate"].astext, Float),
    )
    opportunities_expr = cast(
        Scan.summary["opportunities"]["critique"].astext, Integer,
    )

    rows = (
        db.query(
            Scan.client_id,
            func.count(Scan.id).label("scan_count"),
            func.count(Scan.id).filter(Scan.status == "completed").label("scans_completed"),
            func.max(Scan.completed_at).label("last_scan_at"),
            func.coalesce(
                func.sum(func.coalesce(opportunities_expr, 0))
                .filter(Scan.status == "completed"),
                0,
            ).label("opps_critique"),
            func.avg(visibility_expr).filter(Scan.status == "completed").label("avg_visibility"),
        )
        .filter(Scan.client_id.in_(client_ids))
        .group_by(Scan.client_id)
        .all()
    )
    by_client = {str(r.client_id): r for r in rows}

    client_payload = []
    for c in clients:
        row = by_client.get(str(c.id))
        client_payload.append({
            "id": str(c.id),
            "name": c.name,
            "brand": c.brand,
            "scan_count": int(row.scan_count) if row else 0,
            "scans_completed": int(row.scans_completed) if row else 0,
            "last_scan_at": row.last_scan_at.isoformat() if row and row.last_scan_at else None,
            "opportunities_critique": int(row.opps_critique) if row and row.opps_critique is not None else 0,
            "visibility_rate": (
                round(float(row.avg_visibility), 1)
                if row and row.avg_visibility is not None else None
            ),
        })

    # Org-level totals : aggregate across the per-client payload to keep the
    # response self-consistent (instead of a separate query that could drift).
    totals_scans_completed = sum(c["scans_completed"] for c in client_payload)
    totals_opportunities = sum(c["opportunities_critique"] for c in client_payload)
    vis_values = [c["visibility_rate"] for c in client_payload if c["visibility_rate"] is not None]
    totals_visibility = round(sum(vis_values) / len(vis_values), 1) if vis_values else None

    return {
        "org": {"id": str(org.id), "name": org.name, "is_personal": bool(org.is_personal)},
        "totals": {
            "clients": len(clients),
            "scans_completed": totals_scans_completed,
            "opportunities_critique": totals_opportunities,
            "avg_visibility": totals_visibility,
        },
        "clients": client_payload,
    }


# Phase E.C.5 — Members management : org detail + role mutations.
# The members page renders the matrix (member × client → role) and lets
# owner/admin grant/revoke per-client access without dropping to SQL.

_VALID_ORG_ROLES = {"owner", "admin", "member"}
_VALID_CLIENT_ROLES = {"viewer", "editor", "manager"}
_ORG_MANAGER_ROLES = {"owner", "admin"}


def _require_org_manager(org_id: str, user, db: Session) -> tuple[Organization, OrganizationUser]:
    """Return (org, caller_membership) when user can manage this org.

    Owner + admin can read members and grant client access. Only owner
    can promote/demote owners (enforced at the mutation site).
    """
    try:
        org_uuid = uuid.UUID(org_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed organization_id")
    org = db.query(Organization).filter(Organization.id == org_uuid).first()
    if not org:
        raise HTTPException(404, "Organization not found")
    membership = (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org.id,
            OrganizationUser.user_id == user.id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(403, "You are not a member of this organization")
    if membership.role not in _ORG_MANAGER_ROLES:
        raise HTTPException(
            403,
            f"Only owners and admins can manage members (your role: '{membership.role}')",
        )
    return org, membership


def _count_org_owners(org_id, db: Session, exclude_user_id=None) -> int:
    """Count remaining owners on an org. Used to block last-owner demotion."""
    q = (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org_id,
            OrganizationUser.role == "owner",
        )
    )
    if exclude_user_id is not None:
        q = q.filter(OrganizationUser.user_id != exclude_user_id)
    return q.count()


@router.get("/{org_id}")
async def organization_detail(
    org_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Detailed view of one organization : members + clients + access matrix.

    Drives the members page UI. The members payload includes each member's
    org-level role AND their per-client access rows so the UI can render
    the grant matrix without a second round-trip.

    Members + access endpoint is owner/admin only — letting a `member`
    enumerate other members + their access would be a privacy leak.
    """
    org, caller = _require_org_manager(org_id, user, db)

    clients_in_org = (
        db.query(Client)
        .filter(Client.organization_id == org.id)
        .order_by(Client.name.asc())
        .all()
    )
    client_ids = [c.id for c in clients_in_org]

    members_rows = (
        db.query(OrganizationUser, User)
        .join(User, User.id == OrganizationUser.user_id)
        .filter(OrganizationUser.organization_id == org.id)
        .order_by(User.email.asc())
        .all()
    )

    # Single query for the access matrix : all org_user_clients rows for
    # this org. Cheaper than a per-member loop ; we slot into a dict.
    access_rows = (
        db.query(OrgUserClient)
        .filter(OrgUserClient.organization_id == org.id)
        .all()
    )
    access_by_user: dict[str, dict[str, str]] = {}
    for row in access_rows:
        access_by_user.setdefault(str(row.user_id), {})[str(row.client_id)] = row.role

    members_payload = []
    for ou, u in members_rows:
        client_access = [
            {
                "client_id": str(c.id),
                "client_name": c.name,
                "role": access_by_user.get(str(ou.user_id), {}).get(str(c.id)),
            }
            for c in clients_in_org
        ]
        members_payload.append({
            "user_id": str(ou.user_id),
            "email": u.email,
            "name": u.name,
            "org_role": ou.role,
            "joined_at": ou.joined_at.isoformat() if ou.joined_at else None,
            "is_self": str(ou.user_id) == str(user.id),
            "client_access": client_access,
        })

    return {
        "org": {
            "id": str(org.id),
            "name": org.name,
            "slug": org.slug,
            "is_personal": bool(org.is_personal),
        },
        "your_role": caller.role,
        "clients": [{"id": str(c.id), "name": c.name, "brand": c.brand} for c in clients_in_org],
        "members": members_payload,
    }


class UpdateOrgRoleRequest(BaseModel):
    role: str  # owner | admin | member


@router.patch("/{org_id}/members/{user_id}")
async def update_member_org_role(
    org_id: str, user_id: str, body: UpdateOrgRoleRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change an org member's org-level role.

    Rules :
      - Admin can promote/demote between member and admin only.
      - Only an owner can mint or demote another owner.
      - Cannot demote the last owner of the org (would orphan it).
      - Self-demotion allowed except when you're the last owner.
    """
    org, caller = _require_org_manager(org_id, user, db)

    new_role = (body.role or "").strip().lower()
    if new_role not in _VALID_ORG_ROLES:
        raise HTTPException(400, f"role must be one of {sorted(_VALID_ORG_ROLES)}")

    try:
        target_uuid = uuid.UUID(user_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed user_id")

    target = (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org.id,
            OrganizationUser.user_id == target_uuid,
        )
        .first()
    )
    if not target:
        raise HTTPException(404, "Member not found in this organization")

    if target.role == new_role:
        return {"ok": True, "unchanged": True, "role": new_role}

    # Admin cannot touch owner-tier (promote-to-owner OR demote-an-owner).
    if caller.role != "owner" and (new_role == "owner" or target.role == "owner"):
        raise HTTPException(
            403, "Only an owner can promote or demote owner-level roles",
        )

    # Last-owner guard : prevent locking the org out of ownership.
    if target.role == "owner" and new_role != "owner":
        remaining_owners = _count_org_owners(org.id, db, exclude_user_id=target.user_id)
        if remaining_owners < 1:
            raise HTTPException(
                409,
                "Cannot demote the last owner — promote someone else to owner first.",
            )

    target.role = new_role
    db.commit()
    return {"ok": True, "role": new_role, "user_id": str(target.user_id)}


@router.delete("/{org_id}/members/{user_id}")
async def remove_member(
    org_id: str, user_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a member from the org. Cascades their org_user_clients rows
    via the FK CASCADE on org_user_clients.organization_id+user_id.

    Same guards as role change : admin can't kick an owner, last-owner
    cannot be removed, self-removal allowed except last-owner.
    """
    org, caller = _require_org_manager(org_id, user, db)

    try:
        target_uuid = uuid.UUID(user_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed user_id")

    target = (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org.id,
            OrganizationUser.user_id == target_uuid,
        )
        .first()
    )
    if not target:
        raise HTTPException(404, "Member not found in this organization")

    if caller.role != "owner" and target.role == "owner":
        raise HTTPException(403, "Only an owner can remove another owner")
    if target.role == "owner":
        remaining = _count_org_owners(org.id, db, exclude_user_id=target.user_id)
        if remaining < 1:
            raise HTTPException(
                409,
                "Cannot remove the last owner — promote someone else to owner first.",
            )

    # Wipe their per-client access in this org too. The FK has
    # ON DELETE CASCADE but we mirror it explicitly for clarity + so the
    # response body knows how many rows went.
    deleted_clients = (
        db.query(OrgUserClient)
        .filter(
            OrgUserClient.organization_id == org.id,
            OrgUserClient.user_id == target_uuid,
        )
        .delete(synchronize_session=False)
    )
    db.delete(target)
    db.commit()
    return {"ok": True, "removed_user_id": str(target_uuid), "client_grants_revoked": deleted_clients}


class UpdateClientGrantRequest(BaseModel):
    role: str  # viewer | editor | owner


@router.put("/{org_id}/members/{user_id}/clients/{client_id}")
async def grant_client_access(
    org_id: str, user_id: str, client_id: str, body: UpdateClientGrantRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Grant or update per-client access for an org member.

    PUT semantics : idempotent upsert (creates row if absent, updates
    role if present). The members page sends this on every dropdown
    change without needing to know "is this a new grant or an update".
    """
    org, caller = _require_org_manager(org_id, user, db)

    new_role = (body.role or "").strip().lower()
    if new_role not in _VALID_CLIENT_ROLES:
        raise HTTPException(400, f"role must be one of {sorted(_VALID_CLIENT_ROLES)}")

    try:
        target_user_uuid = uuid.UUID(user_id)
        client_uuid = uuid.UUID(client_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed user_id or client_id")

    # Target must be a member of this org (preserves org isolation).
    if not (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org.id,
            OrganizationUser.user_id == target_user_uuid,
        )
        .first()
    ):
        raise HTTPException(404, "User is not a member of this organization")

    # Client must belong to this org.
    if not (
        db.query(Client)
        .filter(Client.id == client_uuid, Client.organization_id == org.id)
        .first()
    ):
        raise HTTPException(404, "Client not found in this organization")

    existing = (
        db.query(OrgUserClient)
        .filter(
            OrgUserClient.organization_id == org.id,
            OrgUserClient.user_id == target_user_uuid,
            OrgUserClient.client_id == client_uuid,
        )
        .first()
    )
    if existing:
        existing.role = new_role
    else:
        db.add(OrgUserClient(
            organization_id=org.id,
            user_id=target_user_uuid,
            client_id=client_uuid,
            role=new_role,
        ))
    db.commit()
    return {"ok": True, "role": new_role}


@router.delete("/{org_id}/members/{user_id}/clients/{client_id}")
async def revoke_client_access(
    org_id: str, user_id: str, client_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Revoke per-client access. Idempotent (404 returns ok=true rather
    than erroring — the UI calls this on toggling 'no access' regardless
    of prior state)."""
    org, _caller = _require_org_manager(org_id, user, db)

    try:
        target_user_uuid = uuid.UUID(user_id)
        client_uuid = uuid.UUID(client_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed user_id or client_id")

    deleted = (
        db.query(OrgUserClient)
        .filter(
            OrgUserClient.organization_id == org.id,
            OrgUserClient.user_id == target_user_uuid,
            OrgUserClient.client_id == client_uuid,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"ok": True, "revoked": bool(deleted)}


# ── Sprint 15.3 : create a NEW client workspace within an org ────────────
# Distinct from POST /api/clients/ which is idempotent (1 user = 1 personal
# client). This is the agency / multi-tenant flow : owners + admins can spin
# up additional client workspaces inside their org from the header dropdown.
#
# Backend invariants :
#  - caller must be org owner or admin (other org members can be granted
#    access to specific clients individually, but they can't create new ones)
#  - new client is linked to the org via Client.organization_id
#  - caller is auto-granted manager role on the new client via UserClient
#    (legacy) AND OrgUserClient (modern)
#  - apps default to ai_scan enabled, same as POST /clients/ baseline
#
# Frontend wires this from DashboardLayout's org switcher "+ Add new client
# workspace" item.

class NewOrgClientRequest(BaseModel):
    name: str
    brand: str | None = None


class BulkClientCreateRequest(BaseModel):
    # One domain per element. The frontend splits a textarea on newlines and
    # filters empties before submitting, so this list arrives already cleaned
    # but we re-validate server-side. Capped at BULK_CLIENT_MAX_BATCH.
    domains: list[str]


BULK_CLIENT_MAX_BATCH = 50


def _normalize_domain(raw: str) -> str:
    """Same normalisation as scans.create_scan : lowercase, strip scheme,
    drop trailing slash."""
    return (
        (raw or "")
        .strip()
        .lower()
        .removeprefix("https://")
        .removeprefix("http://")
        .rstrip("/")
    )


def _workspace_name_from_domain(domain: str) -> str:
    """Derive a workspace label from a domain. `eau-thermale-avene.fr` ->
    `eau-thermale-avene` ; `www.klorane.com` -> `klorane`. Falls back to
    the raw domain if no usable label is found (truly malformed input)."""
    parts = [p for p in domain.split(".") if p and p != "www"]
    return parts[0] if parts else domain


class RenameOrgRequest(BaseModel):
    name: str


@router.patch("/{org_id}")
async def rename_organization(
    org_id: str,
    req: RenameOrgRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sprint 15.3 - let an org owner/admin rename their agency. The
    auto-generated name from /clients/ (e.g. "Dung Anh LE's agency") is
    fine as a default but agencies want to rebrand to their actual name."""
    from services.sanitize import strip_tags
    org, _caller = _require_org_manager(org_id, user, db)
    new_name = strip_tags((req.name or "").strip())[:120]
    if not new_name:
        raise HTTPException(400, "Organization name is required")
    org.name = new_name
    db.commit()
    db.refresh(org)
    return {"id": str(org.id), "name": org.name}


@router.post("/{org_id}/clients")
async def create_client_in_org(
    org_id: str,
    req: NewOrgClientRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a fresh Client inside the org. Caller becomes manager."""
    from services.sanitize import strip_tags
    org, _caller = _require_org_manager(org_id, user, db)

    raw_name = (req.name or "").strip()
    if not raw_name:
        raise HTTPException(400, "Client name is required")

    client = Client(
        name=strip_tags(raw_name)[:120],
        brand=strip_tags((req.brand or "").strip())[:120] if req.brand else None,
        organization_id=org.id,
    )
    db.add(client)
    db.flush()

    # Legacy user_clients link kept for read paths that haven't migrated to
    # OrgUserClient yet (cleanup tracked in [[project-phase-e-c1-organizations-foundation]]).
    from models import UserClient
    db.add(UserClient(user_id=user.id, client_id=client.id, role="manager"))
    # Modern per-org-per-client role row.
    db.add(OrgUserClient(
        organization_id=org.id,
        user_id=user.id,
        client_id=client.id,
        role="manager",
    ))
    db.commit()
    db.refresh(client)

    return {
        "id": str(client.id),
        "name": client.name,
        "brand": client.brand,
        "organization_id": str(client.organization_id) if client.organization_id else None,
    }


@router.post("/{org_id}/clients/bulk")
async def bulk_create_clients_in_org(
    org_id: str,
    req: BulkClientCreateRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sprint S15.4 - create N workspaces in one call for agencies onboarding
    a portfolio. Each line of the input is treated as a domain ; we derive
    the workspace name from its first non-www label.

    For every successful entry we also create the draft Scan and enqueue
    `fetch_keywords`, so the agency comes back ~10 min later with every
    workspace ready at Gate-2 (Topics / Brands review).

    Idempotent at row level : duplicate `(org_id, name)` is skipped and
    reported under `skipped`, never raised, so a partial retry stays safe.
    """
    from services.sanitize import strip_tags
    from models import UserClient
    org, _caller = _require_org_manager(org_id, user, db)

    raw = req.domains or []
    if len(raw) > BULK_CLIENT_MAX_BATCH:
        raise HTTPException(
            400,
            f"Too many workspaces in one batch (max {BULK_CLIENT_MAX_BATCH}). "
            f"Split into smaller batches.",
        )

    # Seed dedup set with the org's existing client names so we skip
    # duplicates in one DB roundtrip rather than per-loop.
    existing_names = {
        (c.name or "").lower()
        for c in db.query(Client.name).filter(Client.organization_id == org.id).all()
    }

    created = []
    skipped = []
    errors = []
    batch_seen: set[str] = set()  # de-dup within the submitted batch itself

    for raw_domain in raw:
        clean_domain = _normalize_domain(raw_domain)
        if not clean_domain or "." not in clean_domain:
            errors.append({"input": raw_domain, "message": "Invalid domain"})
            continue

        derived = _workspace_name_from_domain(clean_domain)
        ws_name = strip_tags(derived)[:120]
        if not ws_name:
            errors.append({"input": raw_domain, "message": "Could not derive a workspace name"})
            continue

        if ws_name.lower() in existing_names or ws_name.lower() in batch_seen:
            skipped.append({"input": raw_domain, "name": ws_name, "reason": "duplicate"})
            continue
        batch_seen.add(ws_name.lower())

        client = Client(name=ws_name, organization_id=org.id)
        db.add(client)
        db.flush()  # assign client.id without committing - keeps the batch atomic

        db.add(UserClient(user_id=user.id, client_id=client.id, role="manager"))
        db.add(OrgUserClient(
            organization_id=org.id,
            user_id=user.id,
            client_id=client.id,
            role="manager",
        ))

        scan = Scan(
            client_id=client.id,
            name=ws_name,
            domain=clean_domain,
            config={
                "max_position": 50,
                "max_urls": 2000,
                "target_domains": [clean_domain],
                "brand_names": [],
            },
            scan_type="own_brand",
            created_by=user.id,
            run_index=1,
            status="fetching_keywords",
        )
        db.add(scan)
        db.flush()

        db.add(Job(
            scan_id=scan.id,
            job_type="fetch_keywords",
            status="pending",
            payload={"domain": clean_domain, "max_position": 50, "max_urls": 2000},
            attempts=0,
            max_attempts=3,
        ))

        created.append({
            "client_id": str(client.id),
            "scan_id": str(scan.id),
            "name": ws_name,
            "domain": clean_domain,
        })

    db.commit()

    return {
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "summary": {
            "submitted": len(raw),
            "created": len(created),
            "skipped": len(skipped),
            "errors": len(errors),
        },
    }


@router.get("/{org_id}/workspaces/overview")
async def workspaces_overview(
    org_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """S15.4 - cross-workspace dashboard for agencies. Returns one row per
    Client in the org with the latest completed scan summary + auto-rescan
    schedule + focus-brand crisis severity. Any org member (owner / admin /
    member) can read this ; it's the agency's "all-in-one" view that
    replaces visiting each workspace one by one.
    """
    try:
        org_uuid = uuid.UUID(org_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed organization_id")
    org = db.query(Organization).filter(Organization.id == org_uuid).first()
    if not org:
        raise HTTPException(404, "Organization not found")
    membership = (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org.id,
            OrganizationUser.user_id == user.id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(403, "You are not a member of this organization")

    clients = (
        db.query(Client)
        .filter(Client.organization_id == org.id)
        .order_by(Client.created_at.asc())
        .all()
    )
    if not clients:
        return {
            "organization": {"id": str(org.id), "name": org.name},
            "workspaces": [],
            "summary": {"total": 0, "with_scan": 0, "with_crisis": 0},
        }

    cids = [str(c.id) for c in clients]
    from sqlalchemy import text as _text

    # Latest scan per client (any status), so an in-flight rescan still shows up.
    latest_rows = db.execute(_text(
        """
        SELECT DISTINCT ON (s.client_id)
               s.client_id::text   AS cid,
               s.id::text          AS scan_id,
               s.name              AS scan_name,
               s.domain,
               s.status,
               s.schedule,
               s.next_run_at,
               s.completed_at,
               s.summary,
               s.focus_brand_id::text AS focus_brand_id
          FROM scans s
         WHERE s.client_id::text = ANY(:cids)
         ORDER BY s.client_id, s.completed_at DESC NULLS LAST, s.created_at DESC
        """
    ), {"cids": cids}).fetchall()
    latest_by_client = {r.cid: r for r in latest_rows}

    count_rows = db.execute(_text(
        """
        SELECT client_id::text AS cid,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE status='completed') AS completed_count
          FROM scans
         WHERE client_id::text = ANY(:cids)
         GROUP BY client_id
        """
    ), {"cids": cids}).fetchall()
    counts_by_client = {r.cid: (r.total or 0, r.completed_count or 0) for r in count_rows}

    # Focus-brand crisis severity for the latest scan only - keeps the
    # response shape narrow and avoids joining the full lineage.
    latest_scan_ids = [r.scan_id for r in latest_rows if r.scan_id]
    crisis_by_scan: dict[str, tuple] = {}
    if latest_scan_ids:
        crisis_rows = db.execute(_text(
            """
            SELECT scs.scan_id::text AS sid,
                   scs.severity,
                   scs.severity_label
              FROM scan_crisis_signals scs
              JOIN scans s ON s.id = scs.scan_id
             WHERE scs.scan_id::text = ANY(:sids)
               AND s.focus_brand_id = scs.brand_id
            """
        ), {"sids": latest_scan_ids}).fetchall()
        crisis_by_scan = {r.sid: (r.severity, r.severity_label) for r in crisis_rows}

    workspaces = []
    with_scan = 0
    with_crisis = 0
    for c in clients:
        cid = str(c.id)
        total, completed_count = counts_by_client.get(cid, (0, 0))
        latest = latest_by_client.get(cid)
        ws = {
            "client_id": cid,
            "name": c.name,
            "scan_count": total,
            "completed_count": completed_count,
            "latest_scan": None,
        }
        if latest is not None:
            summary = latest.summary or {}
            opp = (summary.get("opportunities") or {}) if isinstance(summary, dict) else {}
            critical_count = (opp.get("critique") if isinstance(opp, dict) else None) or 0
            sev, sev_label = crisis_by_scan.get(latest.scan_id, (None, None))
            if sev_label and sev_label not in ("none", None) and (sev or 0) >= 36:
                with_crisis += 1
            if latest.status == "completed":
                with_scan += 1
            ws["latest_scan"] = {
                "scan_id": latest.scan_id,
                "name": latest.scan_name,
                "domain": latest.domain,
                "status": latest.status,
                "completed_at": latest.completed_at.isoformat() + "Z" if latest.completed_at else None,
                "schedule": latest.schedule,
                "next_run_at": latest.next_run_at.isoformat() + "Z" if latest.next_run_at else None,
                "citation_rate": summary.get("citation_rate") if isinstance(summary, dict) else None,
                "brand_mention_rate": summary.get("brand_mention_rate") if isinstance(summary, dict) else None,
                "target_cited": summary.get("target_cited") if isinstance(summary, dict) else None,
                "total_tests": summary.get("total_tests") if isinstance(summary, dict) else None,
                "critical_count": critical_count,
                "crisis_severity": sev,
                "crisis_severity_label": sev_label,
            }
        workspaces.append(ws)

    return {
        "organization": {"id": str(org.id), "name": org.name, "is_personal": org.is_personal},
        "workspaces": workspaces,
        "summary": {
            "total": len(workspaces),
            "with_scan": with_scan,
            "with_crisis": with_crisis,
        },
    }


class BrandingUpdateRequest(BaseModel):
    # All optional. Pass empty string to clear a field.
    logo_url: str | None = None
    display_name: str | None = None
    accent_color: str | None = None  # CSS hex like "#FF5733"


@router.patch("/{org_id}/branding")
async def update_org_branding(
    org_id: str,
    req: BrandingUpdateRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """S15.4 white-label lite - per-org branding override. Owner / admin only.
    Stored as a flat dict on organizations.branding (JSONB). Empty string on
    any field clears it. Accent color must be a CSS hex like '#FF5733'."""
    org, _caller = _require_org_manager(org_id, user, db)

    current = dict(org.branding or {})
    if req.logo_url is not None:
        v = req.logo_url.strip()
        if v and not (v.startswith("https://") or v.startswith("http://")):
            raise HTTPException(400, "logo_url must be an http(s) URL")
        if v:
            current["logo_url"] = v[:500]
        else:
            current.pop("logo_url", None)
    if req.display_name is not None:
        v = req.display_name.strip()
        if v:
            from services.sanitize import strip_tags
            current["display_name"] = strip_tags(v)[:80]
        else:
            current.pop("display_name", None)
    if req.accent_color is not None:
        v = req.accent_color.strip()
        if v:
            import re as _re
            if not _re.match(r"^#[0-9a-fA-F]{6}$", v):
                raise HTTPException(400, "accent_color must be a CSS hex like #FF5733")
            current["accent_color"] = v.lower()
        else:
            current.pop("accent_color", None)

    org.branding = current
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(org, "branding")
    db.commit()
    db.refresh(org)
    return {"id": str(org.id), "branding": org.branding}


@router.get("/{org_id}/compliance/pdf")
async def org_compliance_pdf(
    org_id: str,
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """S14.1 - server-side rendered PDF of the org-level compliance hub
    (audit log + DPIA template + sub-processors + governance). Same
    Astro-via-internal-HTTP + weasyprint pattern as the per-scan PDF."""
    try:
        org_uuid = uuid.UUID(org_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Malformed organization_id")
    org = db.query(Organization).filter(Organization.id == org_uuid).first()
    if not org:
        raise HTTPException(404, "Organization not found")
    membership = (
        db.query(OrganizationUser)
        .filter(
            OrganizationUser.organization_id == org.id,
            OrganizationUser.user_id == user.id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(403, "You are not a member of this organization")
    from routers.scans import _render_astro_to_pdf
    cookies = {
        k: v for k, v in request.cookies.items()
        if k in ("token", "active_organization_id", "active_client_id")
    }
    return await _render_astro_to_pdf(
        page_path="/app/compliance",
        cookies=cookies,
        filename=f"compliance-org-{str(org.id)[:8]}.pdf",
    )
