"""One-shot backfill : populate target_url_candidates for existing
netlinking_article items that were materialized before C.1.3 ship.

Idempotent : skips items that already have candidates UNLESS --force is
passed. Doesn't overwrite user-set target_url (target_url_source='user_input')
— just populates the candidates list for the picker UI.

Run via :
  docker exec -e PYTHONPATH=/app -w /app senai-worker python scripts/backfill_media_candidates.py
  docker exec -e PYTHONPATH=/app -w /app senai-worker python scripts/backfill_media_candidates.py --force
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


def main():
    from models import ScanContentItem, SessionLocal
    from services.media_picker import pick_media_candidates
    from sqlalchemy.orm.attributes import flag_modified

    force = "--force" in sys.argv

    db = SessionLocal()
    try:
        items = (
            db.query(ScanContentItem)
            .filter(
                ScanContentItem.content_type == "netlinking_article",
                ScanContentItem.status.in_(("identified", "draft")),
            )
            .all()
        )
        if force:
            items_to_backfill = items
            print(f"Total netlinking_article items: {len(items)}", flush=True)
            print(f"FORCE mode : re-picking all {len(items)} items "
                  f"(skipping user_input). Existing candidates will be overwritten.",
                  flush=True)
        else:
            items_to_backfill = [i for i in items if not (i.target_url_candidates or [])]
            print(f"Total netlinking_article items: {len(items)}", flush=True)
            print(f"Without candidates (to backfill): {len(items_to_backfill)}", flush=True)

        matched = 0
        empty = 0
        enriched = 0
        for idx, item in enumerate(items_to_backfill, 1):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(items_to_backfill)} "
                      f"(matched={matched}, empty={empty}, enriched={enriched})",
                      flush=True)
                db.commit()  # incremental commits to bound crash blast radius

            try:
                candidates = pick_media_candidates(
                    scan_id=str(item.scan_id),
                    db=db,
                    target_question=item.target_question,
                    top_k=3,
                )
            except Exception as e:
                print(f"  ERROR item {item.id}: {e}", flush=True)
                continue

            if not candidates:
                empty += 1
                continue

            top1 = candidates[0]
            # Don't overwrite user-set target_url (target_url_source='user_input')
            if item.target_url and item.target_url_source == "user_input":
                item.target_url_candidates = candidates
                flag_modified(item, "target_url_candidates")
            else:
                item.target_url = top1["url"]
                item.target_url_source = "media_picker"
                item.target_url_score = float(top1.get("relevance_score") or 0.0)
                item.target_page_title = top1.get("name") or top1.get("domain")
                item.target_url_candidates = candidates
                flag_modified(item, "target_url_candidates")
            matched += 1
            if top1.get("price_eur") is not None or top1.get("da") is not None:
                enriched += 1

        db.commit()
        print(
            f"Backfill done: matched={matched}, empty={empty}, "
            f"linkfinder_enriched={enriched}",
            flush=True,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
