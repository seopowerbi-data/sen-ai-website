"""018: Public audit-gratuit submissions.

Lifecycle:
  POST /api/audit-requests          → creates pending record + sends magic-link
  POST /api/audit-requests/confirm  → validates JWT, marks confirmed, notifies admin

Spam defense layers:
  1. Rate limit 3/hour per IP (slowapi)
  2. Honeypot field (hidden CSS input bots fill but humans don't)
  3. Magic-link confirmation (kills 99% of bot spam — they don't click)
  4. Input sanitization (strip_tags) on all text fields
  5. Audit log every action for forensics

Account creation deliberately delayed: a User/Client is created later when
results are delivered (Phase 1.5+). That keeps the form completely friction-
less for the first-time visitor — no "create account please" wall.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from config import settings
from models import AuditRequest, get_db
from services.audit import audit_log
from services.rate_limit import limiter
from services.sanitize import strip_tags

logger = logging.getLogger(__name__)

router = APIRouter()


class AuditRequestCreate(BaseModel):
    website: str = Field(min_length=3, max_length=500)
    email: EmailStr
    topic_focus: str = Field(min_length=2, max_length=500)
    first_name: Optional[str] = Field(default=None, max_length=100)
    message: Optional[str] = Field(default=None, max_length=2000)
    # Honeypot field — bots fill it, humans (with hidden CSS) don't.
    # Frontend renders it as an off-screen input the user never sees.
    honeypot: Optional[str] = Field(default=None, max_length=500)


class AuditRequestConfirm(BaseModel):
    token: str


def _create_confirmation_token(audit_request_id: str, email: str, jti: str) -> str:
    """JWT for magic-link confirmation. 24h expiry, jti for replay protection."""
    payload = {
        "sub": audit_request_id,
        "email": email,
        "purpose": "audit_request_confirm",
        "jti": jti,
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _send_confirmation_email(email: str, first_name: str | None, website: str, topic_focus: str, confirm_url: str) -> bool:
    """Send magic-link to prospect. Returns False if Resend not configured."""
    if not settings.resend_api_key:
        logger.warning(f"RESEND_API_KEY not set - audit confirm URL for {email}: {confirm_url}")
        return False
    greeting = f"Bonjour {first_name}," if first_name else "Bonjour,"
    try:
        import resend
        resend.api_key = settings.resend_api_key
        resend.Emails.send({
            "from": settings.resend_from_email,
            "to": [email],
            "subject": "Confirmez votre demande d'audit gratuit sen-ai.fr",
            "html": (
                f"<p>{greeting}</p>"
                f"<p>Vous avez demandé un audit gratuit de la visibilité IA de "
                f"<strong>{website}</strong> sur le sujet <strong>{topic_focus}</strong>.</p>"
                f"<p>Cliquez sur le bouton ci-dessous pour confirmer votre demande. "
                f"Une fois confirmée, on revient vers vous sous 24h avec :</p>"
                f"<ul>"
                f"<li>3 personas réalistes de vos clients sur ce sujet</li>"
                f"<li>6 questions clés posées à ChatGPT et Gemini</li>"
                f"<li>Un mini-rapport actionable avec ce que dit l'IA de votre marque</li>"
                f"</ul>"
                f'<p><a href="{confirm_url}" style="display:inline-block;padding:12px 24px;'
                f'background:#E8707A;color:white;border-radius:8px;text-decoration:none;'
                f'font-weight:bold;">Confirmer ma demande</a></p>'
                f"<p style=\"font-size:13px;color:#666;\">Ce lien expire dans 24 heures. "
                f"Si vous n'êtes pas à l'origine de cette demande, ignorez ce mail.</p>"
                f"<p>- L'équipe sen-ai.fr</p>"
            ),
        })
        return True
    except Exception:
        logger.exception(f"Failed to send audit confirmation email to {email}")
        return False


def _send_admin_notification(audit_req: AuditRequest) -> bool:
    """Notify admin of a confirmed audit request."""
    if not settings.resend_api_key:
        logger.info(f"Audit request confirmed (Resend disabled) - id={audit_req.id} email={audit_req.email}")
        return False
    try:
        import resend
        resend.api_key = settings.resend_api_key
        name_line = f"<strong>Prénom :</strong> {audit_req.first_name}<br>" if audit_req.first_name else ""
        message_line = f"<p><strong>Message :</strong><br>{audit_req.message}</p>" if audit_req.message else ""
        resend.Emails.send({
            "from": settings.resend_from_email,
            "to": [settings.audit_notification_email],
            "subject": f"[sen-ai.fr] Nouvelle demande d'audit confirmee : {audit_req.website}",
            "html": (
                f"<h2>Nouvelle demande d'audit gratuit confirmee</h2>"
                f"<p>"
                f"<strong>Site :</strong> {audit_req.website}<br>"
                f"<strong>Sujet :</strong> {audit_req.topic_focus}<br>"
                f"<strong>Email :</strong> {audit_req.email}<br>"
                f"{name_line}"
                f"</p>"
                f"{message_line}"
                f"<p style=\"font-size:13px;color:#666;\">"
                f"ID : {audit_req.id}<br>"
                f"Confirme : {audit_req.confirmed_at.isoformat() if audit_req.confirmed_at else 'now'}<br>"
                f"IP : {audit_req.source_ip or 'n/a'}"
                f"</p>"
                f'<p><a href="{settings.frontend_url}/app/admin/audit-requests" '
                f'style="display:inline-block;padding:10px 20px;background:#1A202C;color:white;'
                f'border-radius:8px;text-decoration:none;">Voir dans l\'admin</a></p>'
            ),
        })
        return True
    except Exception:
        logger.exception(f"Failed to send admin notification for audit_request {audit_req.id}")
        return False


@router.post("")
@limiter.limit("3/hour")
async def create_audit_request(
    request: Request,
    req: AuditRequestCreate,
    db: Session = Depends(get_db),
):
    """Public endpoint: anonymous visitor submits audit-gratuit form.

    On success, returns 200 even if Resend fails (so the visitor sees a
    success state). Real failures are logged and surfaced via admin
    monitoring.
    """
    # Honeypot trip — return 200 so bot doesn't learn it was caught,
    # but log and skip everything else.
    if req.honeypot:
        logger.warning(f"Audit request honeypot triggered from {request.client.host if request.client else 'unknown'}")
        return {"ok": True, "message": "Demande envoyee."}

    # Sanitize all user-controlled text (defense-in-depth XSS prevention)
    website = strip_tags(req.website) or ""
    topic_focus = strip_tags(req.topic_focus) or ""
    first_name = strip_tags(req.first_name)
    message = strip_tags(req.message)
    email = req.email.lower().strip()

    if len(website) < 3 or len(topic_focus) < 2:
        raise HTTPException(400, "Invalid input")

    # Capture context for forensics + admin display
    xff = request.headers.get("x-forwarded-for", "")
    source_ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else None)
    user_agent = request.headers.get("user-agent", "")[:1000]  # cap to avoid abuse

    # Generate JWT jti up-front — stored for replay protection on confirm
    jti = uuid.uuid4().hex

    audit_req = AuditRequest(
        website=website,
        email=email,
        topic_focus=topic_focus,
        first_name=first_name,
        message=message,
        status="pending",
        confirmation_jti=jti,
        source_ip=source_ip,
        user_agent=user_agent,
    )
    db.add(audit_req)
    db.flush()  # get the id without committing yet

    audit_log(
        db,
        action="audit_request.create",
        target_type="audit_request",
        target_id=str(audit_req.id),
        ip=source_ip,
        details={"email": email, "website": website, "topic_focus": topic_focus},
    )
    db.commit()
    db.refresh(audit_req)

    # Send magic-link confirmation email
    confirm_token = _create_confirmation_token(str(audit_req.id), email, jti)
    confirm_url = f"{settings.frontend_url}/audit/confirm?token={confirm_token}"
    _send_confirmation_email(email, first_name, website, topic_focus, confirm_url)

    return {
        "ok": True,
        "message": "Demande envoyee. Verifiez votre email pour confirmer.",
        "id": str(audit_req.id),
    }


@router.post("/confirm")
@limiter.limit("20/hour")
async def confirm_audit_request(
    request: Request,
    req: AuditRequestConfirm,
    db: Session = Depends(get_db),
):
    """Magic-link target. Validates JWT + jti, marks confirmed, notifies admin."""
    try:
        payload = jwt.decode(req.token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("purpose") != "audit_request_confirm":
            raise HTTPException(400, "Invalid confirmation token")
        audit_request_id = payload.get("sub")
        token_jti = payload.get("jti")
    except JWTError:
        raise HTTPException(400, "Invalid or expired confirmation token")

    audit_req = db.query(AuditRequest).filter(AuditRequest.id == audit_request_id).first()
    if not audit_req:
        raise HTTPException(400, "Invalid confirmation token")

    # Replay protection: jti in DB must match token jti and only be valid once
    if not audit_req.confirmation_jti or audit_req.confirmation_jti != token_jti:
        raise HTTPException(400, "This confirmation link has already been used or expired.")

    if audit_req.status != "pending":
        # Already confirmed (or further along) — return success but don't re-notify admin
        return {
            "ok": True,
            "already_confirmed": True,
            "website": audit_req.website,
            "topic_focus": audit_req.topic_focus,
        }

    # Mark confirmed + invalidate jti so the same link can't be reused
    audit_req.status = "confirmed"
    audit_req.confirmed_at = datetime.utcnow()
    audit_req.confirmation_jti = None  # one-shot

    audit_log(
        db,
        action="audit_request.confirm",
        target_type="audit_request",
        target_id=str(audit_req.id),
        ip=request.client.host if request.client else None,
        details={"email": audit_req.email},
    )
    db.commit()
    db.refresh(audit_req)

    _send_admin_notification(audit_req)

    return {
        "ok": True,
        "website": audit_req.website,
        "topic_focus": audit_req.topic_focus,
        "first_name": audit_req.first_name,
    }
