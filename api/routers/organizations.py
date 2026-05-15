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

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import settings
from models import (
    Client, Organization, OrganizationUser, OrgUserClient, get_db,
)
from services.access import list_user_organizations
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

    valid_active = active_organization_id in {str(o.id) for o in orgs}

    return [
        OrgListItem(
            id=str(o.id),
            name=o.name,
            slug=o.slug,
            is_personal=bool(o.is_personal),
            role=role_by_org.get(str(o.id)),
            member_count=members_by_org.get(str(o.id), 0),
            client_count=clients_by_org.get(str(o.id), 0),
            is_active=(valid_active and str(o.id) == active_organization_id),
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
