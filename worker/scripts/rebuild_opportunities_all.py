"""Phase B Tier A one-shot : nuke + rebuild all opportunities + content items.

Test-phase helper. NOT for production with real user data.

Three steps :
  1. Backfill intent_category on every ScanQuestion (Haiku batch via
     classify_question_intent handler, idempotent on NULL filter).
  2. DELETE all ScanContentItem rows (regardless of status — test data
     only). ScanOpportunity rows are wiped by generate_opportunities
     itself, no manual DELETE needed.
  3. Enqueue generate_opportunities for every completed scan. The worker
     picks them up FIFO and chains materialize_content_items inline.

Usage :
  docker exec -e PYTHONPATH=/app -w /app senai-worker \\
      python scripts/rebuild_opportunities_all.py
  docker exec -e PYTHONPATH=/app -w /app senai-worker \\
      python scripts/rebuild_opportunities_all.py --skip-backfill
  docker exec -e PYTHONPATH=/app -w /app senai-worker \\
      python scripts/rebuild_opportunities_all.py --dry-run

Pass --skip-backfill when the intent_category column is already
populated (saves the Haiku cost on re-runs). Pass --dry-run to see
counts without mutating.
"""

import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("rebuild_opportunities_all")


def main():
    from models import (
        Scan,
        ScanContentItem,
        ScanOpportunity,
        ScanQuestion,
        Job,
        SessionLocal,
    )
    from handlers.classify_question_intent import execute as classify
    from sqlalchemy import distinct

    argv = sys.argv[1:]
    skip_backfill = "--skip-backfill" in argv
    dry_run = "--dry-run" in argv

    db = SessionLocal()
    try:
        # ── Counts before ──────────────────────────────────────────
        n_questions_null = (
            db.query(ScanQuestion)
            .filter(ScanQuestion.intent_category.is_(None))
            .count()
        )
        n_items = db.query(ScanContentItem).count()
        n_opps = db.query(ScanOpportunity).count()
        n_completed_scans = db.query(Scan).filter(Scan.status == "completed").count()

        print("== Current state ==", flush=True)
        print(f"  ScanQuestion with intent_category=NULL : {n_questions_null}", flush=True)
        print(f"  ScanContentItem rows                    : {n_items}", flush=True)
        print(f"  ScanOpportunity rows                    : {n_opps}", flush=True)
        print(f"  Completed scans                          : {n_completed_scans}", flush=True)

        if dry_run:
            print("\n--dry-run set, nothing mutated.", flush=True)
            return

        # ── Step 1 : backfill intent_category ──────────────────────
        if skip_backfill:
            print("\n[1/3] Backfill SKIPPED (--skip-backfill)", flush=True)
        else:
            print("\n[1/3] Backfill intent_category on unclassified questions", flush=True)
            scan_ids = [
                str(row[0]) for row in
                db.query(distinct(ScanQuestion.scan_id))
                .filter(ScanQuestion.intent_category.is_(None))
                .all()
                if row[0] is not None
            ]
            print(f"      Scans needing backfill : {len(scan_ids)}", flush=True)
            ok = failed = total = 0
            for idx, scan_id in enumerate(scan_ids, 1):
                try:
                    res = classify({}, scan_id, db)
                    total += int(res.get("classified", 0))
                    ok += 1
                    if idx % 5 == 0 or idx == len(scan_ids):
                        print(f"      progress {idx}/{len(scan_ids)} "
                              f"(scans_ok={ok} classified={total})", flush=True)
                except Exception:
                    logger.exception(f"classify failed for scan {scan_id}")
                    failed += 1
                    db.rollback()
            print(f"      Backfill done : scans_ok={ok} failed={failed} "
                  f"questions_classified={total}", flush=True)

        # ── Step 2 : nuke ScanContentItem ──────────────────────────
        print("\n[2/3] DELETE all ScanContentItem rows (test phase)", flush=True)
        deleted_items = db.query(ScanContentItem).delete(synchronize_session=False)
        db.commit()
        print(f"      Deleted {deleted_items} ScanContentItem rows", flush=True)

        # ── Step 3 : enqueue generate_opportunities ───────────────
        print("\n[3/3] Enqueue generate_opportunities for completed scans", flush=True)
        completed = (
            db.query(Scan.id)
            .filter(Scan.status == "completed")
            .all()
        )
        enqueued = 0
        for (sid,) in completed:
            db.add(Job(
                scan_id=sid,
                job_type="generate_opportunities",
                status="pending",
                created_at=datetime.utcnow(),
            ))
            enqueued += 1
        db.commit()
        print(f"      Enqueued {enqueued} generate_opportunities jobs", flush=True)

        print("\nDone. Watch worker logs to follow the recompute.", flush=True)
        print("Expected chain per scan : generate_opportunities → "
              "materialize_content_items → (media_picker + auto_suggest_leads inline)",
              flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
