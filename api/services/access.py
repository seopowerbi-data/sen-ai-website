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
