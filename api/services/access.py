"""Centralized access-control helper for client-scoped endpoints.

Phase E.C foundation. Replaces the four `_check_client_access` duplicates
scattered across routers (brands, clients, content_items, oauth) with one
helper that reads the new Organization model AND falls back to the legacy
`user_clients` table during the incremental migration.

The fallback is deliberate : not every router has been refactored yet, and
some integration tests still seed `user_clients` directly. We read both
sources, prefer the new `org_user_clients` row when present, else trust
the legacy row. Phase E.C.next will deprecate `user_clients` once every
endpoint goes through this helper.

Public surface :
    - check_client_access(client_id, user, db, method=None) → str (role)
        Raises HTTPException(403) on no access. Returns the effective role.
    - get_user_client_role(client_id, user, db) → str | None
        Non-raising version. Returns None when no access.
    - require_role(role, minimum) → None
        Raises HTTPException(403) when role rank is below minimum.
    - list_user_organizations(user, db) → list[Organization]
        Used by the future org switcher.
    - list_user_clients(user, db, organization_id=None) → list[Client]
        The 'workspaces you can access' list, optionally scoped to one org.
"""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models import (
    Client, Organization, OrganizationUser, OrgUserClient, UserClient,
)
from services.request_context import current_request_method


# Role rank for write-gate comparisons. Higher = more privileged.
ROLE_RANK = {"viewer": 0, "editor": 1, "owner": 2}
DESTRUCTIVE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def get_user_client_role(client_id: str, user, db: Session) -> str | None:
    """Return the user's effective role on this client, or None when no access.

    Two-layer lookup :
      1. New : `org_user_clients` row for (org_of_client, user, client). The
         org is resolved via `clients.organization_id`.
      2. Legacy fallback : `user_clients` row for (user, client).
    The fallback preserves access for old test fixtures + routers not yet
    refactored to populate org rows on client creation. The new layer wins
    when both exist.
    """
    if not client_id or not user:
        return None

    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return None

    if client.organization_id:
        ouc = (
            db.query(OrgUserClient)
            .filter(
                OrgUserClient.organization_id == client.organization_id,
                OrgUserClient.user_id == user.id,
                OrgUserClient.client_id == client.id,
            )
            .first()
        )
        if ouc and ouc.role:
            return ouc.role

    legacy = (
        db.query(UserClient)
        .filter(UserClient.user_id == user.id, UserClient.client_id == client_id)
        .first()
    )
    if legacy and legacy.role:
        return legacy.role
    return None


def require_role(role: str | None, minimum: str = "viewer") -> None:
    """Raise HTTPException(403) when role rank is below minimum."""
    if role is None:
        raise HTTPException(403, "Access denied")
    actual = ROLE_RANK.get(role, -1)
    floor = ROLE_RANK.get(minimum, 0)
    if actual < floor:
        raise HTTPException(
            403,
            f"Insufficient role: '{role}' (requires '{minimum}' or above)",
        )


def check_client_access(
    client_id: str, user, db: Session, method: str | None = None,
) -> str:
    """Drop-in replacement for the legacy `_check_client_access` duplicates.

    Resolves the user's effective role and raises HTTPException(403) when
    insufficient. Returns the role so callers can branch on it (read-only
    UI hints, audit logs, etc.).

    `method` defaults to the current request method from the contextvar
    set by main.py middleware. Override only for synthetic checks.
    """
    role = get_user_client_role(client_id, user, db)
    if role is None:
        raise HTTPException(403, "Access denied")

    if method is None:
        try:
            method = current_request_method.get()
        except LookupError:
            method = None
    if method and method in DESTRUCTIVE_METHODS:
        require_role(role, minimum="editor")
    return role


def resolve_active_organization_id(user, db: Session, cookie_value: str | None) -> str | None:
    """Resolve the effective active org for this user, in priority order :

    1. Explicit cookie value, IF the user is still a member of that org.
       Stale cookies (org deleted, user removed) are ignored — fall through.
    2. The user's `is_personal` org if any (C.1 backfill creates one per
       legacy client). This is the safe default that matches the "each org
       = isolated workspace" mental model.
    3. None — caller decides what to do (typically : show everything).

    Why this exists : without a default, multi-org users hitting any page
    that does `clients[0]?.id` would get a client from whichever org
    happens to sort first alphabetically — which silently breaks every
    legacy page that assumes clients[0] is "your workspace". Observed
    2026-05-15 with a smoke org "Demo Client B" pushing the real client
    out of position 0 → /welcome onboarding loop.
    """
    if not user:
        return None

    if cookie_value:
        is_member = (
            db.query(OrganizationUser)
            .filter(
                OrganizationUser.user_id == user.id,
                OrganizationUser.organization_id == cookie_value,
            )
            .first()
        )
        if is_member:
            return cookie_value

    personal = (
        db.query(Organization)
        .join(OrganizationUser, OrganizationUser.organization_id == Organization.id)
        .filter(
            OrganizationUser.user_id == user.id,
            Organization.is_personal.is_(True),
        )
        .order_by(Organization.created_at.asc())
        .first()
    )
    if personal:
        return str(personal.id)
    return None


def list_user_organizations(user, db: Session) -> list[Organization]:
    """All orgs the user is a member of, ordered alphabetically by name.

    Drives the future header org switcher.
    """
    if not user:
        return []
    return (
        db.query(Organization)
        .join(OrganizationUser, OrganizationUser.organization_id == Organization.id)
        .filter(OrganizationUser.user_id == user.id)
        .order_by(Organization.name.asc())
        .all()
    )


def list_user_clients(
    user, db: Session, organization_id: str | None = None,
) -> list[Client]:
    """Clients the user can access, optionally scoped to one org.

    New : uses `org_user_clients` as the primary source. Falls back to
    `user_clients` when the user has legacy rows that haven't been
    org-migrated (defensive — backfill should have caught everything).
    Dedupes by client.id ; order = alphabetical by client.name.
    """
    if not user:
        return []

    org_clients = (
        db.query(Client)
        .join(OrgUserClient, OrgUserClient.client_id == Client.id)
        .filter(OrgUserClient.user_id == user.id)
    )
    if organization_id:
        org_clients = org_clients.filter(OrgUserClient.organization_id == organization_id)
    rows = list(org_clients.all())

    if not organization_id:
        legacy = (
            db.query(Client)
            .join(UserClient, UserClient.client_id == Client.id)
            .filter(UserClient.user_id == user.id)
            .all()
        )
        seen = {c.id for c in rows}
        for c in legacy:
            if c.id not in seen:
                rows.append(c)
                seen.add(c.id)

    rows.sort(key=lambda c: (c.name or "").lower())
    return rows
