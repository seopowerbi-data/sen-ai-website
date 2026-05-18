"""One-shot backfill : classify intent_category on existing ScanQuestion
rows that pre-date migration 035.

Iterates over distinct scan_ids that have at least one unclassified
ScanQuestion, then delegates to `classify_question_intent.execute` per
scan. Per-scan delegation keeps the batching + budget cap + LLM logging
identical to the live pipeline path — no risk of divergence.

Idempotent : `classify_question_intent` filters on
`intent_category IS NULL`, so re-running this script is safe.

Usage :
  docker exec -e PYTHONPATH=/app -w /app senai-worker python scripts/backfill_question_intent.py
  docker exec -e PYTHONPATH=/app -w /app senai-worker python scripts/backfill_question_intent.py --limit 5
  docker exec -e PYTHONPATH=/app -w /app senai-worker python scripts/backfill_question_intent.py --scan <scan_uuid>

The --limit flag caps the number of scans processed (useful for a smoke
run before a full sweep). --scan targets a single scan.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("backfill_question_intent")


def _parse_arg(flag: str, argv: list[str]) -> str | None:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def main():
    from models import ScanQuestion, SessionLocal
    from handlers.classify_question_intent import execute as classify
    from sqlalchemy import distinct

    argv = sys.argv[1:]
    limit_str = _parse_arg("--limit", argv)
    limit = int(limit_str) if limit_str else None
    only_scan = _parse_arg("--scan", argv)

    db = SessionLocal()
    try:
        q = (
            db.query(distinct(ScanQuestion.scan_id))
            .filter(ScanQuestion.intent_category.is_(None))
        )
        if only_scan:
            q = q.filter(ScanQuestion.scan_id == only_scan)
        scan_ids = [str(row[0]) for row in q.all() if row[0] is not None]

        if not scan_ids:
            print("No scans with unclassified questions. Nothing to do.", flush=True)
            return

        if limit:
            scan_ids = scan_ids[:limit]

        print(f"Scans to backfill : {len(scan_ids)}", flush=True)

        ok = 0
        failed = 0
        total_classified = 0
        for idx, scan_id in enumerate(scan_ids, 1):
            print(f"  [{idx}/{len(scan_ids)}] scan {scan_id}", flush=True)
            try:
                result = classify({}, scan_id, db)
                total_classified += int(result.get("classified", 0))
                ok += 1
            except Exception:
                logger.exception(f"classify failed for scan {scan_id}")
                failed += 1
                db.rollback()

        print(
            f"Done. scans_ok={ok} scans_failed={failed} "
            f"questions_classified={total_classified}",
            flush=True,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
