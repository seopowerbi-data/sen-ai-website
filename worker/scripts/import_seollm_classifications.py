"""One-shot import : seo-llm `dim_domain.csv` → `domain_classifications` table.

The source CSV is PF's SharePoint-synced dim_domain (3503 rows as of
2026-05-17) containing 12 months of Gemini-classified domains the seo-llm
CLI has accumulated. Importing this seeds the global SaaS cache with PF
parity from day 1 — zero Gemini calls needed for citations on PF scans
where the cited domain is already in this dataset.

Cross-client benefit : a Brand site is a Brand site regardless of which
client surfaced it. doctissimo.fr being tagged `Health & Beauty Media`
helps any future automotive / finance / tech client whose first scan
happens to also cite a health URL on doctissimo.

Idempotent : ON CONFLICT (domain) DO NOTHING preserves any pre-existing
classification (e.g. user manual override added later, or a more recent
Gemini run on a domain that was in this CSV with an older tag).

Run via :
  docker cp dim_domain.csv senai-worker:/tmp/dim_domain.csv
  docker exec -e PYTHONPATH=/app -w /app senai-worker \\
      python scripts/import_seollm_classifications.py /tmp/dim_domain.csv
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


def main(csv_path: str):
    import pandas as pd
    from sqlalchemy import text

    from models import SessionLocal
    from services.domain_classifier import SITE_CATEGORIES, _normalize_domain

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}", flush=True)
    print(f"Columns: {list(df.columns)}", flush=True)

    expected_cols = {"domain_name", "site_type"}
    if not expected_cols.issubset(set(df.columns)):
        print(f"ERROR: CSV missing required columns. Need {expected_cols}.", flush=True)
        sys.exit(1)

    # Normalize domain + filter invalid categories
    df["domain_norm"] = df["domain_name"].astype(str).apply(_normalize_domain)
    df = df[df["domain_norm"] != ""]

    valid_cats = set(SITE_CATEGORIES)
    invalid_categories = set(df["site_type"]) - valid_cats
    if invalid_categories:
        print(
            f"WARNING: CSV has {len(invalid_categories)} unknown category/ies "
            f"(will be skipped): {sorted(invalid_categories)[:5]}",
            flush=True,
        )
    df = df[df["site_type"].isin(valid_cats)]

    # Dedupe by normalized domain (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["domain_norm"], keep="first")
    after = len(df)
    if before != after:
        print(f"Deduped {before - after} duplicate domains", flush=True)

    print(f"Ready to insert {len(df)} rows", flush=True)

    # Parse created_date column once (vectorized) so we can pass real
    # datetime objects to psycopg2 rather than wrestling with PG cast
    # syntax inside the SQL string. NaT → None (we coalesce to NOW() server-side).
    if "created_date" in df.columns:
        df["created_ts"] = pd.to_datetime(df["created_date"], errors="coerce")
    else:
        df["created_ts"] = pd.NaT

    db = SessionLocal()
    try:
        inserted = 0
        skipped_existing = 0
        for _, row in df.iterrows():
            ts = row["created_ts"]
            ts_param = ts.to_pydatetime() if pd.notna(ts) else None
            r = db.execute(
                text("""
                    INSERT INTO domain_classifications (domain, site_type, classified_at, model, source)
                    VALUES (:d, :st, COALESCE(:ts, NOW()), 'gemini-import', 'import_seollm')
                    ON CONFLICT (domain) DO NOTHING
                """),
                {
                    "d": row["domain_norm"],
                    "st": row["site_type"],
                    "ts": ts_param,
                },
            )
            if r.rowcount > 0:
                inserted += 1
            else:
                skipped_existing += 1
        db.commit()

        # Verify post-import distribution
        result = db.execute(text("""
            SELECT site_type, COUNT(*) AS n
            FROM domain_classifications
            GROUP BY site_type
            ORDER BY n DESC
        """)).fetchall()

        print(f"Inserted: {inserted}", flush=True)
        print(f"Skipped (already in DB): {skipped_existing}", flush=True)
        print("\nPost-import distribution:", flush=True)
        for r in result:
            print(f"  {r.site_type:<25} {r.n}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_seollm_classifications.py <path/to/dim_domain.csv>", flush=True)
        sys.exit(1)
    main(sys.argv[1])
