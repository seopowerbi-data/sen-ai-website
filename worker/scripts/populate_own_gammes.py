"""Populate the focus brand's product_lines (from the brief) as my_brand children.

Mirrors the generate_domain_brief own-gamme pre-population, but standalone so it
can be applied to an already-generated scan WITHOUT a full brief regeneration
(which would overwrite curated competitors). Idempotent.

Parent = the scan's focus brand if set, else the first brief.brands entry.
product_lines arrive as "Name (purpose/category)" → only the name is kept.

Run:
    SCAN_ID=<uuid> docker exec senai-worker python /tmp/populate_own_gammes.py
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime

sys.path.insert(0, '/app')

from sqlalchemy import or_
from models import SessionLocal, Scan, ClientBrand, ScanBrandClassification
from services.brand_name_norm import normalize_brand_name

SCAN_ID = os.environ.get("SCAN_ID", "")


def main():
    if not SCAN_ID:
        print("ERROR: SCAN_ID env var required")
        return 1
    db = SessionLocal()
    scan = db.query(Scan).filter(Scan.id == SCAN_ID).first()
    if not scan:
        print(f"ERROR: scan {SCAN_ID} not found")
        return 1
    brief = (scan.config or {}).get("domain_brief") or {}
    product_lines = brief.get("product_lines") or []
    print(f"Scan: {scan.id} ({scan.domain}) — {len(product_lines)} product_lines in brief")

    # Resolve the parent (focus brand, else first own brand).
    parent = None
    if scan.focus_brand_id:
        parent = db.query(ClientBrand).filter(ClientBrand.id == scan.focus_brand_id).first()
    if parent is None:
        for nm in brief.get("brands", []):
            parent = db.query(ClientBrand).filter(
                ClientBrand.client_id == scan.client_id,
                ClientBrand.canonical_name == normalize_brand_name(nm or ""),
            ).first()
            if parent:
                break
    if parent is None:
        print("ERROR: could not resolve focus/own brand to attach gammes to")
        return 1
    print(f"Parent (focus): {parent.name} ({parent.id})")

    created = reused = classified = 0
    seen: set[str] = set()
    for pl in product_lines:
        name = re.split(r"\s*\(", (pl or "").strip(), maxsplit=1)[0].strip()
        norm = normalize_brand_name(name)
        if not norm or norm in seen or norm == parent.canonical_name:
            continue
        seen.add(norm)

        g = db.query(ClientBrand).filter(
            ClientBrand.client_id == scan.client_id,
            or_(ClientBrand.name == name, ClientBrand.canonical_name == norm),
        ).first()
        if not g:
            g = ClientBrand(
                client_id=scan.client_id, name=name, canonical_name=norm,
                parent_id=parent.id, detected_in_scan_id=SCAN_ID,
                auto_detected=True, validated_by_user=False,
                detection_source="brief", last_seen_at=datetime.utcnow(),
            )
            db.add(g); db.flush()
            created += 1
        else:
            g.last_seen_at = datetime.utcnow()
            if g.parent_id is None:
                g.parent_id = parent.id
            reused += 1

        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == SCAN_ID,
            ScanBrandClassification.brand_id == g.id,
        ).first()
        if sbc is None:
            db.add(ScanBrandClassification(
                scan_id=SCAN_ID, brand_id=g.id, classification="my_brand",
                is_focus=False, classified_by="brief", source="brief"))
            classified += 1
        elif sbc.classification == "unclassified":
            sbc.classification = "my_brand"
            sbc.classified_by = "brief"
            sbc.source = "brief"
            sbc.updated_at = datetime.utcnow()
            classified += 1
        print(f"  {'+' if g else ' '} {name}")

    scan.updated_at = datetime.utcnow()
    db.commit()

    n_my = db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == SCAN_ID,
        ScanBrandClassification.classification == "my_brand").count()
    print(f"\nSUMMARY: created={created} reused={reused} my_brand-classified={classified}")
    print(f"Total my_brand rows now: {n_my}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
