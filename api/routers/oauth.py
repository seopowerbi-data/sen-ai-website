"""OAuth delegation endpoints — connect external accounts (Phase 0).

Flow:
  1. User clicks "Connect Google Sheets" in /app/settings/connections
  2. Frontend → GET /api/oauth/google/authorize?product=sheets&client_id=X
  3. API signs a state JWT and redirects to Google consent screen
  4. Google redirects to GET /api/oauth/google/callback?code=X&state=Y
  5. API exchanges code → tokens, encrypts, stores in oauth_connections
  6. User is redirected to /app/settings/connections?status=connected
"""

import logging
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from config import settings
from models import OAuthConnection, UserClient, get_db
from services.auth_service import get_current_user
from services.oauth_providers import (
    PRODUCT_PROVIDER,
    PRODUCT_SCOPES,
    PROVIDER_PRODUCTS,
    get_provider,
)
from services.rate_limit import limiter
from services.token_manager import decrypt_token, encrypt_token

logger = logging.getLogger(__name__)
router = APIRouter()

STATE_TTL_MINUTES = 10


# ── Helpers ──────────────────────────────────────────────────────────

def _sign_state(payload: dict) -> str:
    """Create a signed, short-lived state JWT for the OAuth redirect."""
    payload["exp"] = datetime.utcnow() + timedelta(minutes=STATE_TTL_MINUTES)
    payload["jti"] = uuid.uuid4().hex
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _verify_state(token: str) -> dict:
    """Decode and validate the state JWT. Raises HTTPException on failure."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise HTTPException(400, "Invalid or expired OAuth state")


def _check_client_access(user, client_id: str, db: Session, require_role: str = "editor"):
    """Phase E.C : delegate to services.access. Different signature kept
    for call-site compatibility (oauth callers pass require_role)."""
    from services.access import get_user_client_role, require_role as _req
    role = get_user_client_role(client_id, user, db)
    if role is None:
        raise HTTPException(403, "Access denied to this client")
    _req(role, minimum=require_role)
    return role


# ── Authorize ────────────────────────────────────────────────────────

@router.get("/{provider}/authorize")
async def oauth_authorize(
    provider: str,
    product: str = Query(..., description="Product to connect, e.g. sheets"),
    client_id: str = Query(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Start the OAuth flow: validate inputs, sign state, redirect to provider."""

    # Validate provider + product
    if provider not in PROVIDER_PRODUCTS:
        raise HTTPException(400, f"Unknown provider: {provider}")
    if product not in PROVIDER_PRODUCTS.get(provider, []):
        raise HTTPException(400, f"Product '{product}' is not valid for provider '{provider}'")

    # Check client access (editor or owner can connect)
    _check_client_access(user, client_id, db, require_role="editor")

    scopes = PRODUCT_SCOPES.get(product, [])
    state = _sign_state({
        "user_id": str(user.id),
        "client_id": client_id,
        "provider": provider,
        "product": product,
    })

    oauth_provider = get_provider(provider)
    url = oauth_provider.authorize_url(state, scopes)
    return RedirectResponse(url, status_code=302)


# ── Callback ─────────────────────────────────────────────────────────

@router.get("/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """Handle the OAuth redirect: exchange code, encrypt tokens, store connection."""

    # Validate state
    payload = _verify_state(state)
    if payload.get("provider") != provider:
        raise HTTPException(400, "Provider mismatch in state")

    user_id = payload["user_id"]
    client_id = payload["client_id"]
    product = payload["product"]

    oauth_provider = get_provider(provider)

    # Exchange code for tokens
    try:
        tokens = await oauth_provider.exchange_code(code)
    except Exception as e:
        logger.exception(f"OAuth code exchange failed for {provider}")
        return RedirectResponse(
            f"{settings.frontend_url}/app/settings/connections?status=error&message=code_exchange_failed",
            status_code=302,
        )

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in")

    if not access_token:
        return RedirectResponse(
            f"{settings.frontend_url}/app/settings/connections?status=error&message=no_access_token",
            status_code=302,
        )

    # Fetch identity of the user who consented
    try:
        user_info = await oauth_provider.fetch_user_info(access_token)
    except Exception:
        logger.exception(f"Failed to fetch user info from {provider}")
        user_info = {}

    # Encrypt tokens
    access_encrypted = encrypt_token(access_token)
    refresh_encrypted = encrypt_token(refresh_token) if refresh_token else None

    # Check for existing active connection (same client + provider + product + account)
    account_id = user_info.get("account_id")
    existing = (
        db.query(OAuthConnection)
        .filter(
            OAuthConnection.client_id == client_id,
            OAuthConnection.provider == provider,
            OAuthConnection.product == product,
            OAuthConnection.account_id == account_id,
            OAuthConnection.status == "active",
        )
        .first()
    )

    granted_scopes = tokens.get("scope", "").split() if tokens.get("scope") else PRODUCT_SCOPES.get(product, [])

    if existing:
        # Update tokens on existing connection (re-consent)
        existing.access_token_encrypted = access_encrypted
        if refresh_encrypted:
            existing.refresh_token_encrypted = refresh_encrypted
        existing.token_expires_at = (
            datetime.utcnow() + timedelta(seconds=expires_in) if expires_in else None
        )
        existing.scopes = granted_scopes
        existing.account_email = user_info.get("account_email") or existing.account_email
        existing.account_name = user_info.get("account_name") or existing.account_name
        existing.authorized_by_user_id = user_id
        existing.authorized_at = datetime.utcnow()
        existing.status = "active"
        existing.updated_at = datetime.utcnow()
        db.commit()
        logger.info(f"Updated OAuth connection {existing.id} ({provider}/{product}) for client {client_id}")
    else:
        # Create new connection
        conn = OAuthConnection(
            client_id=client_id,
            provider=provider,
            product=product,
            account_id=account_id,
            account_email=user_info.get("account_email"),
            account_name=user_info.get("account_name"),
            access_token_encrypted=access_encrypted,
            refresh_token_encrypted=refresh_encrypted,
            token_expires_at=(
                datetime.utcnow() + timedelta(seconds=expires_in) if expires_in else None
            ),
            scopes=granted_scopes,
            status="active",
            authorized_by_user_id=user_id,
            authorized_at=datetime.utcnow(),
        )
        db.add(conn)
        db.commit()
        logger.info(f"Created OAuth connection ({provider}/{product}) for client {client_id}")

    return RedirectResponse(
        f"{settings.frontend_url}/app/settings/connections?status=connected&product={product}",
        status_code=302,
    )


# ── List connections ─────────────────────────────────────────────────

@router.get("/connections")
async def list_connections(
    client_id: str = Query(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all OAuth connections for a client (no tokens returned)."""
    _check_client_access(user, client_id, db, require_role="viewer")

    rows = (
        db.query(OAuthConnection)
        .filter(OAuthConnection.client_id == client_id)
        .order_by(OAuthConnection.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(c.id),
            "provider": c.provider,
            "product": c.product,
            "account_id": c.account_id,
            "account_email": c.account_email,
            "account_name": c.account_name,
            "status": c.status,
            "scopes": c.scopes,
            "token_expires_at": c.token_expires_at.isoformat() if c.token_expires_at else None,
            "authorized_at": c.authorized_at.isoformat() if c.authorized_at else None,
            "last_used_at": c.last_used_at.isoformat() if c.last_used_at else None,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in rows
    ]


# ── Delete (revoke) ─────────────────────────────────────────────────

@router.delete("/connections/{connection_id}")
async def delete_connection(
    connection_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Revoke and delete an OAuth connection."""
    conn = db.query(OAuthConnection).filter(OAuthConnection.id == connection_id).first()
    if not conn:
        raise HTTPException(404, "Connection not found")

    _check_client_access(user, str(conn.client_id), db, require_role="editor")

    # Best-effort revocation at the provider
    if conn.access_token_encrypted:
        try:
            provider = get_provider(conn.provider)
            token = decrypt_token(conn.access_token_encrypted)
            await provider.revoke(token)
        except Exception:
            logger.warning(f"Provider revocation failed for connection {conn.id}, proceeding with deletion")

    db.delete(conn)
    db.commit()
    logger.info(f"Deleted OAuth connection {conn.id} ({conn.provider}/{conn.product})")

    return {"deleted": True}


# ── Force refresh ────────────────────────────────────────────────────

@router.post("/connections/{connection_id}/refresh")
async def refresh_connection(
    connection_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Force-refresh the access token for a connection."""
    conn = db.query(OAuthConnection).filter(OAuthConnection.id == connection_id).first()
    if not conn:
        raise HTTPException(404, "Connection not found")

    _check_client_access(user, str(conn.client_id), db, require_role="editor")

    if conn.status != "active":
        raise HTTPException(400, f"Connection is {conn.status}, cannot refresh")
    if not conn.refresh_token_encrypted:
        raise HTTPException(400, "No refresh token available")

    provider = get_provider(conn.provider)
    refresh_token = decrypt_token(conn.refresh_token_encrypted)

    try:
        new_tokens = await provider.refresh_access_token(refresh_token)
    except Exception as e:
        logger.exception(f"Refresh failed for connection {conn.id}")
        conn.status = "expired"
        db.commit()
        raise HTTPException(502, f"Provider refresh failed: {e}")

    conn.access_token_encrypted = encrypt_token(new_tokens["access_token"])
    conn.token_expires_at = datetime.utcnow() + timedelta(
        seconds=new_tokens.get("expires_in", 3600)
    )
    if new_tokens.get("refresh_token"):
        conn.refresh_token_encrypted = encrypt_token(new_tokens["refresh_token"])
    conn.last_used_at = datetime.utcnow()
    conn.updated_at = datetime.utcnow()
    db.commit()

    return {
        "id": str(conn.id),
        "status": conn.status,
        "token_expires_at": conn.token_expires_at.isoformat() if conn.token_expires_at else None,
    }


# ── Admin: list all connections (superadmin) ─────────────────────────

@router.get("/admin/connections")
async def admin_list_connections(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Superadmin: list all OAuth connections across all clients."""
    if not user.is_superadmin:
        raise HTTPException(403, "Superadmin access required")

    rows = (
        db.query(OAuthConnection)
        .order_by(OAuthConnection.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(c.id),
            "client_id": str(c.client_id),
            "provider": c.provider,
            "product": c.product,
            "account_id": c.account_id,
            "account_email": c.account_email,
            "account_name": c.account_name,
            "status": c.status,
            "scopes": c.scopes,
            "token_expires_at": c.token_expires_at.isoformat() if c.token_expires_at else None,
            "authorized_at": c.authorized_at.isoformat() if c.authorized_at else None,
            "last_used_at": c.last_used_at.isoformat() if c.last_used_at else None,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in rows
    ]
