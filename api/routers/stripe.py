"""Stripe integration — one-time credit pack purchases.

Flow: Frontend → POST /checkout (pack_id) → Stripe Checkout → webhook → credit ledger.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import desc, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import stripe as stripe_lib

from config import settings
from models import Client, ClientCredit, Scan, UserClient, get_db
from services.auth_service import get_current_user
from services.audit import audit_log
from services.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter()
stripe_lib.api_key = settings.stripe_api_key

# ── Credit pack catalog ──────────────────────────────────────────────
# price_id filled from Stripe Dashboard (Products → Price ID).
# Set via env: STRIPE_PRICE_SCAN_5, etc. For now, hardcode after creation.
CREDIT_PACKS = {
    "scan_5":      {"credit_type": "scan",    "amount": 5,   "price_eur": 40,  "price_id": "price_1TJKlzAWSRE7HR2Kcvdt8UfR"},
    "scan_15":     {"credit_type": "scan",    "amount": 15,  "price_eur": 100, "price_id": "price_1TJKnyAWSRE7HR2KLKai7OsG"},
    "scan_50":     {"credit_type": "scan",    "amount": 50,  "price_eur": 250, "price_id": "price_1TJKoXAWSRE7HR2K5FbkE64A"},
    "content_10":  {"credit_type": "content", "amount": 10,  "price_eur": 40,  "price_id": "price_1TJKp4AWSRE7HR2KtGXhcBEo"},
    "content_30":  {"credit_type": "content", "amount": 30,  "price_eur": 100, "price_id": "price_1TJKpJAWSRE7HR2KFPdHTdvl"},
    "content_100": {"credit_type": "content", "amount": 100, "price_eur": 250, "price_id": "price_1TJKpXAWSRE7HR2KQjsJZdf5"},
}


# ── Helpers ───────────────────────────────────────────────────────────

def lock_client_credits(client_id: str, db: Session) -> None:
    """Acquire a row-lock on the clients row to serialize credit operations.

    The lock is held until the current transaction commits or rolls back.
    This prevents the credit double-spend race where two concurrent debits
    both read the same balance and each succeed (e.g. user clicks Launch
    twice or two API requests overlap). Callers that need to atomically
    read-then-mutate the balance must take this lock first.
    """
    db.execute(
        text("SELECT 1 FROM clients WHERE id = :id FOR UPDATE"),
        {"id": str(client_id)},
    )


def get_credit_balance(client_id: str, credit_type: str, db: Session) -> int:
    """Read current balance from the latest ledger entry."""
    latest = (
        db.query(ClientCredit)
        .filter(ClientCredit.client_id == client_id, ClientCredit.credit_type == credit_type)
        .order_by(desc(ClientCredit.created_at))
        .first()
    )
    return latest.balance_after if latest else 0


def add_credits(
    client_id: str,
    credit_type: str,
    amount: int,
    description: str,
    db: Session,
    stripe_session_id: str | None = None,
    scan_id: str | None = None,
) -> ClientCredit:
    """Insert a ledger row and return it. amount can be positive (purchase) or negative (consumption).

    Acquires the per-client credit lock so the read-modify-write of the
    balance is serialized across concurrent transactions. Re-entrant within
    the same transaction (cheap no-op if the caller already locked).
    """
    lock_client_credits(client_id, db)
    current = get_credit_balance(client_id, credit_type, db)
    new_balance = current + amount
    if new_balance < 0:
        raise ValueError(f"Insufficient {credit_type} credits: have {current}, need {abs(amount)}")
    entry = ClientCredit(
        client_id=client_id,
        credit_type=credit_type,
        amount=amount,
        balance_after=new_balance,
        description=description,
        stripe_session_id=stripe_session_id,
        scan_id=scan_id,
    )
    db.add(entry)
    return entry


# ── Endpoints ─────────────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    pack_id: str      # e.g. "scan_5", "content_30"
    client_id: str


@router.post("/checkout")
@limiter.limit("10/minute")
async def create_checkout(request: Request, req: CheckoutRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a Stripe Checkout session for a credit pack (one-time payment)."""
    pack = CREDIT_PACKS.get(req.pack_id)
    if not pack:
        raise HTTPException(400, f"Unknown pack: {req.pack_id}")
    if not pack["price_id"]:
        raise HTTPException(503, "Stripe price not configured for this pack yet")

    client = db.query(Client).filter(Client.id == req.client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")
    # Phase E.C.2 — delegate to services/access so org_user_clients counts too.
    from services.access import get_user_client_role
    role = get_user_client_role(str(client.id), user, db)
    if role is None:
        raise HTTPException(403, "Access denied")
    # H6: only owners can spend money. Editors and viewers can browse pricing
    # but not initiate a Stripe Checkout that creates a real charge.
    if role != "owner":
        raise HTTPException(
            403,
            f"Only an owner can purchase credits (your role: '{role}')",
        )

    # Create or reuse Stripe customer
    if not client.stripe_customer_id:
        customer = stripe_lib.Customer.create(email=user.email, name=client.name)
        client.stripe_customer_id = customer.id
        db.commit()

    session = stripe_lib.checkout.Session.create(
        customer=client.stripe_customer_id,
        line_items=[{"price": pack["price_id"], "quantity": 1}],
        mode="payment",  # one-time, not subscription
        success_url=f"{settings.frontend_url}/app/settings?payment=success",
        cancel_url=f"{settings.frontend_url}/app/settings?payment=canceled",
        metadata={
            "client_id": str(client.id),
            "pack_id": req.pack_id,
            "credit_type": pack["credit_type"],
            "credit_amount": str(pack["amount"]),
        },
    )
    return {"checkout_url": session.url}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events — credits are added on successful payment."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe_lib.Webhook.construct_event(
            payload, sig, settings.stripe_webhook_secret
        )
    except (ValueError, stripe_lib.error.SignatureVerificationError):
        raise HTTPException(400, "Invalid webhook")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        meta = getattr(session, "metadata", None) or {}
        client_id = meta.get("client_id") if isinstance(meta, dict) else getattr(meta, "client_id", None)
        credit_type = meta.get("credit_type") if isinstance(meta, dict) else getattr(meta, "credit_type", None)
        credit_amount = meta.get("credit_amount") if isinstance(meta, dict) else getattr(meta, "credit_amount", None)
        pack_id = meta.get("pack_id", "") if isinstance(meta, dict) else getattr(meta, "pack_id", "")
        stripe_session_id = getattr(session, "id", None)

        if client_id and credit_type and credit_amount:
            # Serialize concurrent credit operations on this client (incl.
            # the idempotency check below) — without the lock, two webhook
            # deliveries could both see existing=None and double-credit.
            lock_client_credits(client_id, db)
            # Idempotency: check if this session was already processed
            existing = db.query(ClientCredit).filter(
                ClientCredit.stripe_session_id == stripe_session_id
            ).first()
            if existing:
                logger.info(f"Webhook duplicate — session {stripe_session_id} already processed")
                return {"received": True}

            amount = int(credit_amount)
            try:
                add_credits(
                    client_id=client_id,
                    credit_type=credit_type,
                    amount=amount,
                    description=f"Purchased pack {pack_id} ({amount} {credit_type} credits)",
                    db=db,
                    stripe_session_id=stripe_session_id,
                )
                audit_log(db, action="credit.purchase", target_type="client", target_id=client_id,
                          details={"pack_id": pack_id, "credit_type": credit_type, "amount": amount,
                                   "stripe_session_id": stripe_session_id})
                db.commit()
            except IntegrityError:
                # Race-safe fallback: a concurrent webhook delivery beat us to
                # the insert and the partial UNIQUE index on stripe_session_id
                # rejected ours. Treat as already-processed (idempotent).
                db.rollback()
                logger.info(
                    f"Webhook race — session {stripe_session_id} already credited "
                    f"by concurrent delivery (UNIQUE constraint hit)"
                )
                return {"received": True}
            logger.info(f"Credits added: {amount} {credit_type} for client {client_id} (session {stripe_session_id})")

    return {"received": True}


@router.get("/credits/{client_id}")
async def get_credits(client_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Get current credit balances for a client."""
    from services.access import check_client_access
    check_client_access(client_id, user, db)
    return {
        "scan": get_credit_balance(client_id, "scan", db),
        "content": get_credit_balance(client_id, "content", db),
    }


@router.get("/credits/{client_id}/history")
async def get_credit_history(client_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Get credit transaction history for a client."""
    from services.access import check_client_access
    check_client_access(client_id, user, db)
    entries = (
        db.query(ClientCredit)
        .filter(ClientCredit.client_id == client_id)
        .order_by(desc(ClientCredit.created_at))
        .limit(50)
        .all()
    )
    return [
        {
            "id": str(e.id),
            "credit_type": e.credit_type,
            "amount": e.amount,
            "balance_after": e.balance_after,
            "description": e.description,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in entries
    ]


@router.get("/packs")
async def get_packs():
    """Return available credit packs (public, no auth needed)."""
    return [
        {
            "id": pack_id,
            "credit_type": pack["credit_type"],
            "amount": pack["amount"],
            "price_eur": pack["price_eur"],
            "available": bool(pack["price_id"]),
        }
        for pack_id, pack in CREDIT_PACKS.items()
    ]
