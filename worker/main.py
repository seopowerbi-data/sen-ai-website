"""Worker main loop — polls PostgreSQL for pending jobs and executes handlers."""

import logging
import time
from datetime import datetime, timedelta

import httpx
from sqlalchemy import desc, text
from sqlalchemy.orm import Session

from config import settings
from models import ClientCredit, Job, SessionLocal


def _format_user_error(exc: Exception) -> str:
    """Convert a raw exception into a user-facing scan.error_message.

    httpx.HTTPStatusError stringifies as 'Client error 'XYZ' for url ...' which
    is meaningless to end users. Most provider errors carry a JSON body with a
    human-readable message (`{"error": {"type": ..., "message": ...}}`) — we
    extract it and prepend the provider name so the user knows where to act.

    Special-cases the common billing/quota error so it reads as a clear billing
    issue rather than a vague rate-limit-y message.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.json()
            err = body.get("error", {}) if isinstance(body, dict) else {}
            provider_msg = (err.get("message") or "").strip()
            err_type = err.get("type", "")
            url = str(exc.request.url).lower()
            if "anthropic.com" in url:
                provider = "Anthropic (Claude)"
            elif "openai.com" in url:
                provider = "OpenAI"
            elif "googleapis.com" in url or "generativelanguage" in url:
                provider = "Gemini"
            else:
                provider = "AI provider"
            msg_lower = provider_msg.lower()
            if any(kw in msg_lower for kw in ("credit balance", "billing", "quota", "insufficient_quota")):
                return (
                    f"{provider} billing/quota issue: {provider_msg}\n"
                    f"Recharge your {provider} account, then click Retry."
                )
            if exc.response.status_code == 429:
                return f"{provider} rate-limited: {provider_msg or 'too many requests'} — try again in a few minutes."
            return f"{provider} error ({err_type or exc.response.status_code}): {provider_msg[:300]}"
        except Exception:
            pass
    return str(exc)[:500]

# H4: stuck-job sweep config. The longest legitimate handler is run_llm_tests
# (LLM calls per question × providers — can run 30-60 min for big scans).
# 2h is a comfortable cap; anything past that is definitely worker-killed.
STUCK_JOB_TIMEOUT_HOURS = 2
CLEANUP_INTERVAL_SECONDS = 300  # run sweep at most every 5 min
_LAST_CLEANUP_TS = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("worker")

# Handler registry
HANDLERS = {}


def register_handler(job_type: str):
    def decorator(func):
        HANDLERS[job_type] = func
        return func
    return decorator


def load_handlers():
    from handlers import (fetch_keywords, classify_topics, assign_keywords,
                          generate_personas, generate_persona_questions,
                          run_llm_tests, generate_editorial,
                          detect_competitors, generate_opportunities, cleanup_brands,
                          generate_domain_brief, generate_client_brief,
                          generate_faq)  # noqa: F401
    HANDLERS["fetch_keywords"] = fetch_keywords.execute
    HANDLERS["classify_topics"] = classify_topics.execute
    HANDLERS["assign_keywords"] = assign_keywords.execute
    HANDLERS["detect_competitors"] = detect_competitors.execute
    HANDLERS["generate_personas"] = generate_personas.execute
    HANDLERS["generate_persona_questions"] = generate_persona_questions.execute
    HANDLERS["run_llm_tests"] = run_llm_tests.execute
    HANDLERS["generate_opportunities"] = generate_opportunities.execute
    HANDLERS["generate_editorial"] = generate_editorial.execute
    HANDLERS["cleanup_brands"] = cleanup_brands.execute
    HANDLERS["generate_domain_brief"] = generate_domain_brief.execute
    HANDLERS["generate_client_brief"] = generate_client_brief.execute
    HANDLERS["generate_faq"] = generate_faq.execute


def _refund_scan_credits(scan_id, db: Session) -> None:
    """Refund any credits that were debited for this scan.

    Called when a scan permanently fails (attempts >= max_attempts).
    Idempotent: skips if a refund row already exists for this scan.
    Net-aware: refunds the absolute net of debits minus prior refunds, so a
    partial refund history can't double-refund.
    """
    if not scan_id:
        return

    # All ledger rows tied to this scan
    rows = db.query(ClientCredit).filter(ClientCredit.scan_id == scan_id).all()
    if not rows:
        return

    # Net per credit_type (negative = still owed back to user)
    net_by_type: dict[str, int] = {}
    client_id = None
    for r in rows:
        net_by_type[r.credit_type] = net_by_type.get(r.credit_type, 0) + r.amount
        client_id = r.client_id

    for credit_type, net in net_by_type.items():
        if net >= 0:
            continue  # nothing owed (already refunded or never debited)
        refund_amount = -net  # positive

        # Lock the client row to serialize against any concurrent credit op
        db.execute(
            text("SELECT 1 FROM clients WHERE id = :id FOR UPDATE"),
            {"id": str(client_id)},
        )

        # Read latest balance for this (client, type) AFTER lock
        latest = (
            db.query(ClientCredit)
            .filter(
                ClientCredit.client_id == client_id,
                ClientCredit.credit_type == credit_type,
            )
            .order_by(desc(ClientCredit.created_at))
            .first()
        )
        new_balance = (latest.balance_after if latest else 0) + refund_amount

        db.add(ClientCredit(
            client_id=client_id,
            credit_type=credit_type,
            amount=refund_amount,
            balance_after=new_balance,
            description="Refund: scan failed",
            scan_id=scan_id,
        ))
        logger.info(
            f"Refunded {refund_amount} {credit_type} credits to client {client_id} "
            f"for failed scan {scan_id}"
        )


def cleanup_stuck_jobs() -> None:
    """Sweep for jobs stuck in 'running' for too long and reclaim them.

    H4: a worker that crashes (OOM, kill -9, container restart, host reboot)
    leaves its job in status='running' forever — the existing retry logic
    only fires when the handler raises an exception in the same process,
    so a hard-killed worker bypasses C2 entirely. This sweep is the safety
    net: any job that started > STUCK_JOB_TIMEOUT_HOURS ago gets marked
    failed, its scan marked failed, and credits refunded via the C2 helper.

    Cheap no-op most of the time: only runs every CLEANUP_INTERVAL_SECONDS,
    and uses FOR UPDATE SKIP LOCKED so multiple workers don't fight over
    the same row.
    """
    global _LAST_CLEANUP_TS
    now = time.time()
    if now - _LAST_CLEANUP_TS < CLEANUP_INTERVAL_SECONDS:
        return
    _LAST_CLEANUP_TS = now

    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(hours=STUCK_JOB_TIMEOUT_HOURS)
        stuck_rows = db.execute(
            text("""
                SELECT id FROM jobs
                WHERE status = 'running' AND started_at < :cutoff
                FOR UPDATE SKIP LOCKED
            """),
            {"cutoff": cutoff},
        ).fetchall()

        if not stuck_rows:
            return

        logger.warning(
            f"Stuck-job sweep: found {len(stuck_rows)} job(s) running > "
            f"{STUCK_JOB_TIMEOUT_HOURS}h — reclaiming"
        )

        from models import Scan  # local import: only loaded if there's work

        for (job_id,) in stuck_rows:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                continue

            elapsed_min = 0
            if job.started_at:
                elapsed_min = int((datetime.utcnow() - job.started_at).total_seconds() / 60)
            error_msg = (
                f"Job stuck — no progress for {elapsed_min} min "
                f"(worker likely killed mid-execution)"
            )

            job.status = "failed"
            job.result = {"error": error_msg, "stuck_cleanup": True}
            job.completed_at = datetime.utcnow()

            scan = db.query(Scan).filter(Scan.id == job.scan_id).first()
            if scan:
                scan.status = "failed"
                scan.error_message = error_msg
                scan.updated_at = datetime.utcnow()

            try:
                _refund_scan_credits(job.scan_id, db)
            except Exception:
                logger.exception(
                    f"Failed to refund credits for stuck scan {job.scan_id}"
                )

            logger.warning(
                f"Reclaimed stuck job {job_id} (scan={job.scan_id}, "
                f"type={job.job_type}, elapsed={elapsed_min}min)"
            )

        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Stuck-job sweep failed (will retry next interval)")
    finally:
        db.close()


def poll_and_execute():
    """Pick one pending job and execute it."""
    db = SessionLocal()
    try:
        # FOR UPDATE SKIP LOCKED: safe concurrent polling
        job = db.execute(
            text("""
                SELECT id FROM jobs
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            """)
        ).fetchone()

        if not job:
            return False

        job_id = job[0]
        job_obj = db.query(Job).filter(Job.id == job_id).first()
        if not job_obj:
            return False

        job_obj.status = "running"
        job_obj.started_at = datetime.utcnow()
        job_obj.attempts = (job_obj.attempts or 0) + 1
        db.commit()

        handler = HANDLERS.get(job_obj.job_type)
        if not handler:
            job_obj.status = "failed"
            job_obj.result = {"error": f"Unknown job type: {job_obj.job_type}"}
            job_obj.completed_at = datetime.utcnow()
            db.commit()
            logger.error(f"Unknown job type: {job_obj.job_type}")
            return True

        logger.info(f"Executing job {job_obj.id} type={job_obj.job_type} scan={job_obj.scan_id} client={job_obj.client_id}")

        try:
            result = handler(
                job_payload=job_obj.payload or {},
                scan_id=str(job_obj.scan_id) if job_obj.scan_id else None,
                db=db,
            )
            job_obj.status = "completed"
            job_obj.result = result or {}
            job_obj.completed_at = datetime.utcnow()
            db.commit()
            logger.info(f"Job {job_obj.id} completed: {result}")

        except Exception as e:
            db.rollback()
            logger.exception(f"Job {job_obj.id} failed: {e}")

            # PermanentScanError signals "retrying won't help" — typically a
            # data-availability issue on user input (e.g., HaloScan has no
            # ranking data for the domain). Skip the retry loop, fail fast,
            # and tell the UI to hide the retry button.
            from exceptions import PermanentScanError
            is_permanent = isinstance(e, PermanentScanError)

            # Re-fetch job after rollback
            job_obj = db.query(Job).filter(Job.id == job_id).first()
            if job_obj:
                if is_permanent or (job_obj.attempts or 0) >= (job_obj.max_attempts or 3):
                    user_msg = str(e) if is_permanent else _format_user_error(e)
                    job_obj.status = "failed"
                    job_obj.attempts = job_obj.max_attempts  # block any further retry
                    job_obj.result = {
                        "error": str(e),
                        "user_message": user_msg,
                        "permanent": is_permanent,
                    }

                    # Also mark scan as failed + flag retryable for the UI
                    from models import Scan
                    from sqlalchemy.orm.attributes import flag_modified
                    scan = db.query(Scan).filter(Scan.id == job_obj.scan_id).first()
                    if scan:
                        scan.status = "failed"
                        scan.error_message = user_msg
                        scan.updated_at = datetime.utcnow()
                        if is_permanent:
                            summary = dict(scan.summary or {})
                            summary["retryable"] = False
                            scan.summary = summary
                            flag_modified(scan, "summary")

                    # Auto-refund any credits debited for this scan so the
                    # user is not charged for a job that never delivered.
                    try:
                        _refund_scan_credits(job_obj.scan_id, db)
                    except Exception:
                        logger.exception(
                            f"Failed to refund credits for scan {job_obj.scan_id}"
                        )
                else:
                    job_obj.status = "pending"  # Retry

                job_obj.completed_at = datetime.utcnow()
                db.commit()

        return True

    finally:
        db.close()


def wait_for_db():
    """Wait for PostgreSQL to be ready and tables to exist."""
    from sqlalchemy import text
    for attempt in range(30):
        try:
            db = SessionLocal()
            db.execute(text("SELECT 1 FROM jobs LIMIT 0"))
            db.close()
            return
        except Exception:
            logger.info(f"Waiting for database... (attempt {attempt + 1})")
            time.sleep(2)
    raise RuntimeError("Database not ready after 60s")


def main():
    logger.info(f"Worker {settings.worker_id} starting, poll interval={settings.poll_interval}s")
    wait_for_db()
    load_handlers()
    logger.info(f"Registered handlers: {list(HANDLERS.keys())}")

    while True:
        try:
            cleanup_stuck_jobs()  # cheap no-op except every CLEANUP_INTERVAL_SECONDS
            had_job = poll_and_execute()
            if not had_job:
                time.sleep(settings.poll_interval)
        except KeyboardInterrupt:
            logger.info("Worker shutting down")
            break
        except Exception:
            logger.exception("Unexpected error in poll loop")
            time.sleep(5)


if __name__ == "__main__":
    main()
