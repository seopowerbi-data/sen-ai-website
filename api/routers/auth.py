import logging
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import func
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError
import httpx

from config import settings
from models import Client, ClientCredit, User, UserClient, get_db
from services.auth_service import get_current_user
from services.audit import audit_log
from services.rate_limit import limiter
from services.sanitize import strip_tags

logger = logging.getLogger(__name__)

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _validate_password(password: str):
    """Enforce password complexity: min 8 chars, 1 uppercase, 1 lowercase, 1 digit."""
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if not any(c.isupper() for c in password):
        raise HTTPException(400, "Password must contain at least 1 uppercase letter")
    if not any(c.islower() for c in password):
        raise HTTPException(400, "Password must contain at least 1 lowercase letter")
    if not any(c.isdigit() for c in password):
        raise HTTPException(400, "Password must contain at least 1 digit")


def _create_verification_token(user_id: str, email: str) -> str:
    """Short-lived JWT for email verification (24h)."""
    payload = {
        "sub": user_id,
        "email": email,
        "purpose": "email_verification",
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _send_verification_email(email: str, verify_url: str) -> bool:
    """Send verification email via Resend. Falls back to log if not configured."""
    if not settings.resend_api_key:
        logger.warning(f"RESEND_API_KEY not set — verification URL for {email}: {verify_url}")
        return False
    try:
        import resend
        resend.api_key = settings.resend_api_key
        resend.Emails.send({
            "from": settings.resend_from_email,
            "to": [email],
            "subject": "Verify your sen-ai.fr email",
            "html": (
                f"<p>Welcome to sen-ai.fr! Please verify your email to activate your account "
                f"and receive <strong>50 free scan credits</strong>.</p>"
                f'<p><a href="{verify_url}" style="display:inline-block;padding:12px 24px;'
                f'background:#E8604C;color:white;border-radius:8px;text-decoration:none;'
                f'font-weight:bold;">Verify my email</a></p>'
                f"<p>This link expires in 24 hours.</p>"
                f"<p>— sen-ai.fr</p>"
            ),
        })
        return True
    except Exception:
        logger.exception(f"Failed to send verification email to {email}")
        return False


@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
async def register(request: Request, req: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    # Registration kill-switch (config flag). The audit-gratuit form stays open
    # since it does not create an account.
    if not settings.registration_open:
        raise HTTPException(503, "Registrations are temporarily closed. Request a free audit instead at https://sen-ai.fr/#engagement")
    _validate_password(req.password)
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(400, "Registration failed. If you already have an account, try logging in.")

    user = User(
        email=req.email,
        name=strip_tags(req.name),
        password_hash=pwd_context.hash(req.password),
        is_email_verified=False,
    )
    db.add(user)
    audit_log(db, action="auth.register", user_id=str(user.id) if user.id else None,
              target_type="user", ip=request.client.host if request.client else None,
              details={"email": req.email})
    db.commit()
    db.refresh(user)

    # Send verification email (welcome bonus granted on verify, not here)
    verify_token = _create_verification_token(str(user.id), user.email)
    verify_url = f"{settings.frontend_url}/verify-email?token={verify_token}"
    _send_verification_email(user.email, verify_url)

    token = create_token(str(user.id), user.email)
    # Same HttpOnly cookie logic as /login — overwrites any stale session cookie.
    response.set_cookie(
        "token", token,
        httponly=True, secure=True, samesite="lax",
        max_age=settings.jwt_expire_minutes * 60,
        path="/",
    )
    return TokenResponse(access_token=token)


@router.post("/verify-email")
@limiter.limit("10/minute")
async def verify_email(request: Request, response: Response, db: Session = Depends(get_db)):
    """Verify email from token (query param or JSON body)."""
    # Accept token from query param or JSON body
    verify_token = request.query_params.get("token", "")
    if not verify_token:
        try:
            body = await request.json()
            verify_token = body.get("token", "")
        except Exception:
            pass
    if not verify_token:
        raise HTTPException(400, "Verification token required")

    try:
        payload = jwt.decode(verify_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("purpose") != "email_verification":
            raise HTTPException(400, "Invalid verification token")
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(400, "Invalid or expired verification token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(400, "Invalid verification token")

    if user.is_email_verified:
        return {"ok": True, "message": "Email already verified.", "already_verified": True}

    user.is_email_verified = True
    audit_log(db, action="auth.verify_email", user_id=str(user.id),
              target_type="user", ip=request.client.host if request.client else None)
    db.commit()

    # Grant welcome bonus now that email is verified
    _grant_welcome_bonus(user, db)

    # Set auth cookie so user is logged in after clicking the link
    auth_token = create_token(str(user.id), user.email)
    response.set_cookie(
        "token", auth_token,
        httponly=True, secure=True, samesite="lax",
        max_age=settings.jwt_expire_minutes * 60,
        path="/",
    )
    return {"ok": True, "message": "Email verified. Welcome bonus credited!"}


@router.post("/resend-verification")
@limiter.limit("3/minute")
async def resend_verification(request: Request, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Resend verification email for the current user."""
    if user.is_email_verified:
        return {"ok": True, "message": "Email already verified."}

    verify_token = _create_verification_token(str(user.id), user.email)
    verify_url = f"{settings.frontend_url}/verify-email?token={verify_token}"
    _send_verification_email(user.email, verify_url)
    return {"ok": True, "message": "Verification email sent."}


def _grant_welcome_bonus(user: User, db: Session):
    """Grant 50 scan credits to the user's client (called after email verification)."""
    link = db.query(UserClient).filter(UserClient.user_id == user.id).first()
    if not link:
        return
    # Check if bonus was already granted (idempotent)
    existing = db.query(ClientCredit).filter(
        ClientCredit.client_id == link.client_id,
        ClientCredit.description == "Welcome bonus — 50 free scan credits",
    ).first()
    if existing:
        return
    db.add(ClientCredit(
        client_id=link.client_id,
        credit_type="scan",
        amount=50,
        balance_after=50,
        description="Welcome bonus — 50 free scan credits",
    ))
    db.commit()


@router.post("/logout")
async def logout(response: Response):
    """Clear the HttpOnly token cookie server-side.

    A client-side `document.cookie = 'token=; max-age=0'` CANNOT delete an
    HttpOnly cookie, so without this endpoint users remain stuck on a stale
    session even after clicking "logout" or submitting a different login form.
    """
    response.delete_cookie("token", path="/")
    # Phase E.C.2/3 — also clear the active org + client cookies so a
    # different user logging in on the same browser doesn't inherit the
    # previous session's workspace selection.
    response.delete_cookie("active_organization_id", path="/")
    response.delete_cookie("active_client_id", path="/")
    return {"ok": True}


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    """Lightweight user profile — used by frontend for verification status check."""
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "is_email_verified": user.is_email_verified,
        "is_superadmin": user.is_superadmin,
    }


@router.delete("/me")
async def delete_account(
    response: Response,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """H7 / GDPR Art.17 — self-service account deletion.

    Hard-deletes the user. For each client the user has access to:
      * If the user is the SOLE member of that client → delete the client
        too. Migration 008 cascades the deletion through scans, brands,
        credits, sub-tables and jobs, so a single DELETE wipes everything.
      * If the client has other members → only the user_clients link is
        removed (cascade from user delete). Client and its data are
        preserved for the remaining members.

    Scans authored by this user on multi-member clients keep their rows but
    `created_by` becomes NULL (audit trail anonymized — see migration 008).

    The auth cookie is cleared in the response so the browser session ends
    immediately. The action is logged at WARNING level for compliance audit.
    """
    user_id = user.id
    user_email = user.email

    # Identify clients where this user is the sole member.
    # `having count(...) = 1` ensures only solo-owned clients are picked.
    sole_client_ids = (
        db.query(UserClient.client_id)
        .group_by(UserClient.client_id)
        .having(func.count(UserClient.user_id) == 1)
        .filter(
            UserClient.client_id.in_(
                db.query(UserClient.client_id).filter(UserClient.user_id == user_id)
            )
        )
        .all()
    )
    sole_client_ids = [row[0] for row in sole_client_ids]

    # Delete sole-owned clients first. Migration 008's CASCADE chain wipes
    # user_clients, scans (and all scan_* children), brands, credits,
    # api_keys, modules, subscriptions for each one.
    if sole_client_ids:
        db.query(Client).filter(Client.id.in_(sole_client_ids)).delete(
            synchronize_session=False
        )

    # Audit before delete (user_id will be SET NULL after cascade)
    audit_log(db, action="auth.delete_account", user_id=str(user_id),
              target_type="user", details={"email": user_email, "sole_clients_dropped": len(sole_client_ids)})

    # Delete the user. user_clients rows referencing the user (on
    # multi-member clients) cascade away; scans.created_by becomes NULL.
    db.query(User).filter(User.id == user_id).delete(synchronize_session=False)
    db.commit()

    logger.warning(
        f"GDPR account deletion: user={user_id} email={user_email} "
        f"sole_clients_dropped={len(sole_client_ids)}"
    )

    response.delete_cookie("token", path="/")
    return {
        "deleted": True,
        "sole_clients_dropped": len(sole_client_ids),
    }


@router.get("/me/export")
async def export_account_data(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """H8 / GDPR Art.15 (right of access) + Art.20 (data portability).

    Returns a structured JSON dump of every record tied to the requesting
    user. Scope per client they have access to: client metadata, the user's
    role on that client, scans + their full content (topics, personas,
    questions, LLM results, opportunities, brand classifications, content
    items), brands seen for that client, and the credit ledger.

    The returned shape is stable so a user can reliably re-import their
    data elsewhere (Art.20). For multi-tenant clients we expose the same
    payload regardless of which member is asking — every member of a
    client has equal access to that client's data already.
    """
    # Local imports keep cold-start cheap and avoid pulling these models
    # into other auth code paths.
    from models import (
        ClientCredit,
        ClientBrand,
        Scan,
        ScanBrandClassification,
        ScanBrandTopic,
        ScanContentItem,
        ScanKeyword,
        ScanLLMResult,
        ScanOpportunity,
        ScanPersona,
        ScanQuestion,
        ScanTopic,
    )

    def serialize_row(row, fields: list[str]) -> dict:
        out = {}
        for f in fields:
            v = getattr(row, f, None)
            if isinstance(v, datetime):
                out[f] = v.isoformat()
            elif hasattr(v, "hex"):  # UUID
                out[f] = str(v)
            else:
                out[f] = v
        return out

    # User profile (no password hash, no Google ID — those are credentials,
    # not portable data the user is entitled to under Art.20)
    user_payload = {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "auth_methods": {
            "password": user.password_hash is not None,
            "google_oauth": user.google_id is not None,
        },
    }

    # Clients the user has access to
    links = (
        db.query(UserClient).filter(UserClient.user_id == user.id).all()
    )

    clients_payload = []
    for link in links:
        client = db.query(Client).filter(Client.id == link.client_id).first()
        if not client:
            continue

        # Brands seen for this client
        brands = db.query(ClientBrand).filter(ClientBrand.client_id == client.id).all()

        # Credit ledger for this client
        credits = (
            db.query(ClientCredit)
            .filter(ClientCredit.client_id == client.id)
            .order_by(ClientCredit.created_at)
            .all()
        )

        # Scans for this client (with all nested children)
        scans_payload = []
        scans = db.query(Scan).filter(Scan.client_id == client.id).order_by(Scan.created_at).all()
        for scan in scans:
            scan_id = scan.id
            scans_payload.append({
                "id": str(scan_id),
                "name": scan.name,
                "domain": scan.domain,
                "status": scan.status,
                "run_index": scan.run_index,
                "parent_scan_id": str(scan.parent_scan_id) if scan.parent_scan_id else None,
                "config": scan.config,
                "summary": scan.summary,
                "created_at": scan.created_at.isoformat() if scan.created_at else None,
                "completed_at": scan.completed_at.isoformat() if scan.completed_at else None,
                "topics": [
                    serialize_row(t, ["id", "name", "description", "keyword_count"])
                    for t in db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).all()
                ],
                "personas": [
                    serialize_row(p, ["id", "name", "data", "is_active"])
                    for p in db.query(ScanPersona).filter(ScanPersona.scan_id == scan_id).all()
                ],
                "questions": [
                    serialize_row(q, ["id", "persona_id", "question", "type_question", "is_active"])
                    for q in db.query(ScanQuestion).filter(ScanQuestion.scan_id == scan_id).all()
                ],
                "keywords": [
                    serialize_row(k, ["id", "url", "keyword", "position", "traffic", "search_volume"])
                    for k in db.query(ScanKeyword).filter(ScanKeyword.scan_id == scan_id).all()
                ],
                "llm_results": [
                    serialize_row(r, [
                        "id", "question_id", "provider", "model", "response_text",
                        "citations", "target_cited", "target_position",
                        "brand_mentions", "brand_analysis", "created_at",
                    ])
                    for r in db.query(ScanLLMResult).filter(ScanLLMResult.scan_id == scan_id).all()
                ],
                "opportunities": [
                    serialize_row(o, [
                        "id", "question_id", "topic_name", "persona_name",
                        "brand_cited", "brand_position", "best_competitor_name",
                        "priority", "opportunity_score", "recommended_action", "target_url",
                    ])
                    for o in db.query(ScanOpportunity).filter(ScanOpportunity.scan_id == scan_id).all()
                ],
                "brand_classifications": [
                    serialize_row(b, ["id", "brand_id", "classification", "is_focus", "classified_by"])
                    for b in db.query(ScanBrandClassification).filter(ScanBrandClassification.scan_id == scan_id).all()
                ],
                "brand_topics": [
                    serialize_row(bt, ["id", "brand_id", "topic_id"])
                    for bt in db.query(ScanBrandTopic).filter(ScanBrandTopic.scan_id == scan_id).all()
                ],
                "content_items": [
                    serialize_row(ci, [
                        "id", "content_type", "topic_name", "persona_name",
                        "target_url", "target_question", "content_html",
                        "status", "created_at",
                    ])
                    for ci in db.query(ScanContentItem).filter(ScanContentItem.scan_id == scan_id).all()
                ],
            })

        clients_payload.append({
            "id": str(client.id),
            "name": client.name,
            "brand": client.brand,
            "user_role": link.role,
            "created_at": client.created_at.isoformat() if client.created_at else None,
            "brands": [
                serialize_row(b, [
                    "id", "name", "canonical_name", "category", "domain",
                    "first_detected_at", "detection_source", "validated_by_user",
                ])
                for b in brands
            ],
            "credit_ledger": [
                serialize_row(c, [
                    "id", "credit_type", "amount", "balance_after",
                    "description", "stripe_session_id", "scan_id", "created_at",
                ])
                for c in credits
            ],
            "scans": scans_payload,
        })

    return {
        "export_format": "sen-ai.fr GDPR data export v1",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "user": user_payload,
        "clients": clients_payload,
    }


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    password: str


def _create_reset_token(user_id: str, email: str) -> str:
    """Short-lived JWT for password reset (15 min)."""
    payload = {
        "sub": user_id,
        "email": email,
        "purpose": "password_reset",
        "exp": datetime.utcnow() + timedelta(minutes=15),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _send_reset_email(email: str, reset_url: str) -> bool:
    """Send reset email via Resend. Returns False if Resend is not configured."""
    if not settings.resend_api_key:
        logger.warning(f"RESEND_API_KEY not set — reset URL for {email}: {reset_url}")
        return False
    try:
        import resend
        resend.api_key = settings.resend_api_key
        resend.Emails.send({
            "from": settings.resend_from_email,
            "to": [email],
            "subject": "Reset your sen-ai.fr password",
            "html": (
                f"<p>You requested a password reset for your sen-ai.fr account.</p>"
                f'<p><a href="{reset_url}" style="display:inline-block;padding:12px 24px;'
                f'background:#E8604C;color:white;border-radius:8px;text-decoration:none;'
                f'font-weight:bold;">Reset Password</a></p>'
                f"<p>This link expires in 15 minutes. If you didn't request this, ignore this email.</p>"
                f"<p>— sen-ai.fr</p>"
            ),
        })
        return True
    except Exception:
        logger.exception(f"Failed to send reset email to {email}")
        return False


@router.post("/forgot-password")
@limiter.limit("5/minute")
async def forgot_password(request: Request, req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Request a password reset link. Always returns 200 (no account enumeration)."""
    user = db.query(User).filter(User.email == req.email).first()
    if user and user.password_hash:
        token = _create_reset_token(str(user.id), user.email)
        reset_url = f"{settings.frontend_url}/reset-password?token={token}"
        _send_reset_email(user.email, reset_url)
    # Always return success to prevent account enumeration
    return {"ok": True, "message": "If this email exists, a reset link has been sent."}


@router.post("/reset-password")
@limiter.limit("5/minute")
async def reset_password(request: Request, req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Verify reset token and set new password."""
    try:
        payload = jwt.decode(req.token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("purpose") != "password_reset":
            raise HTTPException(400, "Invalid reset token")
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(400, "Invalid or expired reset token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(400, "Invalid reset token")

    _validate_password(req.password)
    user.password_hash = pwd_context.hash(req.password)
    db.commit()

    return {"ok": True, "message": "Password updated. You can now log in."}


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.password_hash:
        raise HTTPException(401, "Invalid credentials")
    if not pwd_context.verify(req.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")

    audit_log(db, action="auth.login", user_id=str(user.id),
              target_type="user", ip=request.client.host if request.client else None)
    db.commit()

    token = create_token(str(user.id), user.email)

    # Set cookie server-side (HttpOnly) so it OVERWRITES any existing HttpOnly
    # token from a prior session (e.g. Google OAuth). Without this, the browser
    # keeps the old HttpOnly cookie and JS cannot overwrite it → user sees the
    # previous account. Matches the /google/callback cookie attributes.
    response.set_cookie(
        "token", token,
        httponly=True, secure=True, samesite="lax",
        max_age=settings.jwt_expire_minutes * 60,
        path="/",
    )
    return TokenResponse(access_token=token)


_ALLOWED_INTENTS = {"agency"}


def _sign_oauth_state(intent: str = "") -> str:
    """Create a signed, short-lived state JWT for CSRF protection on Google login.
    Sprint 15 : carries the signup `intent` (currently 'agency' only) so the
    callback can forward the right onboarding flag to /welcome after auth.
    Invalid intents are silently dropped - the state stays a CSRF token first."""
    payload = {
        "purpose": "google_login",
        "jti": uuid.uuid4().hex,
        "exp": datetime.utcnow() + timedelta(minutes=10),
    }
    if intent and intent in _ALLOWED_INTENTS:
        payload["intent"] = intent
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _verify_oauth_state(token: str) -> dict:
    """Decode and validate the Google login state JWT."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("purpose") != "google_login":
            raise HTTPException(400, "Invalid OAuth state")
        return payload
    except JWTError:
        raise HTTPException(400, "Invalid or expired OAuth state")


@router.get("/google")
async def google_login(intent: str = ""):
    """Sprint 15 : optional `?intent=agency` carries through the OAuth flow
    via the signed `state` JWT. The callback decodes it and forwards to
    /welcome?intent=agency so the wizard can surface the agency banner."""
    state = _sign_oauth_state(intent=intent)
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/auth?{query}")


@router.get("/google/callback")
async def google_callback(code: str, state: str = "", response: Response = None, db: Session = Depends(get_db)):
    state_payload = _verify_oauth_state(state)
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(400, "Google OAuth failed")
        tokens = token_resp.json()

        userinfo_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        userinfo = userinfo_resp.json()

    user = db.query(User).filter(User.google_id == userinfo["id"]).first()
    is_new_user = False
    if not user:
        user = db.query(User).filter(User.email == userinfo["email"]).first()
        if user:
            user.google_id = userinfo["id"]
            # Auto-verify existing user linking Google account
            if not user.is_email_verified:
                user.is_email_verified = True
        else:
            # Registration kill-switch: refuse to auto-create new accounts via Google.
            # Existing accounts (already in DB) continue to work and can link Google.
            if not settings.registration_open:
                return RedirectResponse(f"{settings.frontend_url}/register?error=closed")
            is_new_user = True
            user = User(
                email=userinfo["email"],
                name=userinfo.get("name", ""),
                google_id=userinfo["id"],
                is_email_verified=True,  # Google already verified the email
            )
            db.add(user)
        db.commit()
        db.refresh(user)

    # Grant welcome bonus for new Google OAuth users (email pre-verified)
    if is_new_user:
        _grant_welcome_bonus(user, db)

    token = create_token(str(user.id), user.email)
    # Forward the agency intent (carried via the OAuth state JWT) so the
    # /welcome wizard surfaces the agency banner and the deep onboarding
    # flow (S15.1) can branch on it. Default redirect = /app/dashboard ;
    # /app/dashboard itself redirects to /welcome when the user has no
    # workspace yet, so the intent flag has to ride along.
    intent = (state_payload or {}).get("intent") or ""
    dest = "/app/dashboard"
    if intent in _ALLOWED_INTENTS:
        dest = f"/app/dashboard?intent={intent}"
    resp = RedirectResponse(dest)
    resp.set_cookie(
        "token", token,
        httponly=True, secure=True, samesite="lax",
        max_age=settings.jwt_expire_minutes * 60,
    )
    return resp
