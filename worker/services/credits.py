"""Credit refund helpers used by handlers (in-flight refunds for partial delivery).

`_refund_scan_credits` in worker/main.py handles the FULL refund when a scan
permanently fails. This module covers the partial case: a scan completes but
delivers fewer results than the user paid for (e.g., circuit breaker skipped
N tests, or all providers errored on K questions). C.3 / Phase C.

Pattern follows _refund_scan_credits (same locking, same ledger row shape) so
the audit trail stays consistent — every refund row carries scan_id and a
human-readable description.
"""

from __future__ import annotations

import logging

from sqlalchemy import desc, text
from sqlalchemy.orm import Session

from models import ClientCredit

logger = logging.getLogger(__name__)


def partial_refund_scan_credits(
    db: Session,
    client_id,
    scan_id,
    amount: int,
    description: str,
) -> None:
    """Credit `amount` scan_credits back to client.

    Caller is responsible for ensuring `amount > 0` and that the refund is
    legitimate (e.g., questions never delivered). No idempotency check here:
    the caller decides when to issue. Net-aware refund logic still lives in
    `_refund_scan_credits` (full refund on failure) — partial refunds layer
    cleanly on top because both decrement the absolute net the same way.

    Locks the client row via SELECT ... FOR UPDATE to serialize against any
    concurrent credit op (Stripe webhook, another scan launch, etc.).
    """
    if amount <= 0:
        return

    db.execute(
        text("SELECT 1 FROM clients WHERE id = :id FOR UPDATE"),
        {"id": str(client_id)},
    )
    latest = (
        db.query(ClientCredit)
        .filter(
            ClientCredit.client_id == client_id,
            ClientCredit.credit_type == "scan",
        )
        .order_by(desc(ClientCredit.created_at))
        .first()
    )
    new_balance = (latest.balance_after if latest else 0) + amount

    db.add(ClientCredit(
        client_id=client_id,
        credit_type="scan",
        amount=amount,
        balance_after=new_balance,
        description=description,
        scan_id=scan_id,
    ))
    db.commit()
    logger.info(
        f"Partial refund: {amount} scan credits to client {client_id} "
        f"for scan {scan_id} ({description})"
    )
