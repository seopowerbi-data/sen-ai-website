"""One-shot: fix the René Furterer scan brief + brand classifications.

The Web Brief LLM tends to list all Pierre Fabre sister brands as "Own brands".
This script repins the brief to ["René Furterer"] and reclassifies sister brands
+ external hair-care competitors as competitors.

René Furterer = premium hair/scalp vertical. Competitors below are hair-care only.

Run:
    SCAN_ID=<furterer-scan-uuid> \
      docker exec senai-worker python /tmp/fix_furterer_brief.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, '/app')

from sqlalchemy import or_
from sqlalchemy.orm.attributes import flag_modified
from models import SessionLocal, Scan, ClientBrand, ScanBrandClassification
from services.brand_name_norm import normalize_brand_name

SCAN_ID = os.environ.get("SCAN_ID", "")

COMPETITORS_OVERRIDE = [
    # --- Pierre Fabre sister brands (user-validated → competitor) ---
    {"name": "Klorane", "domain": "klorane.com", "products": [
        "Quinine", "Ortie", "Galanga", "Mangue",
    ]},
    {"name": "Ducray", "domain": "ducray.com", "products": [
        "Anaphase", "Neoptide", "Squanorm", "Kelual DS",
    ]},
    # NOTE: Avène / A-Derma deliberately excluded — René Furterer is pure
    # hair/scalp and those sisters have no competing hair range (relevance filter).
    # --- Hair-care competitors (premium + pharmacy) ---
    {"name": "Kérastase", "domain": "kerastase.fr", "products": [
        "Nutritive", "Spécifique", "Genesis", "Résistance",
    ]},
    {"name": "Phyto", "domain": "phyto.com", "products": [
        "Phytocyane", "Phytonovathrix", "Phytophanère", "Phytodéfrisant",
    ]},
    {"name": "Luxéol", "domain": "luxeol.fr", "products": [
        "Anti-chute", "Cheveux et ongles", "Pousse",
    ]},
    {"name": "Vichy", "domain": "vichy.fr", "products": [
        "Dercos Aminexil", "Dercos Densi-Solutions", "Dercos Anti-pelliculaire",
    ]},
    {"name": "L'Oréal Professionnel", "domain": "lorealprofessionnel.fr", "products": [
        "Serie Expert", "Metal Detox",
    ]},
    {"name": "Forté Pharma", "domain": "fortepharma.com", "products": [
        "Forcapil",
    ]},
    {"name": "Nioxin", "domain": "nioxin.com", "products": [
        "System Kit",
    ]},
]

SISTER_BRANDS_TO_PROMOTE = [
    # Only sisters with a competing hair range. Avène / A-Derma omitted (no hair line).
    "klorane",
    "ducray",
]


def main():
    if not SCAN_ID:
        print("ERROR: SCAN_ID env var required")
        return 1
    db = SessionLocal()
    scan = db.query(Scan).filter(Scan.id == SCAN_ID).first()
    if not scan:
        print(f"ERROR: scan {SCAN_ID} not found")
        return 1
    print(f"Scan: {scan.id} ({scan.domain})")

    cfg = dict(scan.config or {})
    brief = dict(cfg.get("domain_brief") or {})
    brief["brands"] = ["René Furterer"]  # the scanned brand ONLY
    brief["competitors"] = COMPETITORS_OVERRIDE
    cfg["domain_brief"] = brief
    cfg["domain_brief_manual_edit"] = datetime.utcnow().isoformat()
    scan.config = cfg
    flag_modified(scan, "config")
    print(f"→ Brief updated: brands=['René Furterer'], competitors={len(COMPETITORS_OVERRIDE)} entries")

    promoted = 0
    for sb_name_low in SISTER_BRANDS_TO_PROMOTE:
        brand = db.query(ClientBrand).filter(
            ClientBrand.client_id == scan.client_id,
            ClientBrand.canonical_name == normalize_brand_name(sb_name_low),
            ClientBrand.parent_id.is_(None),
        ).first()
        if not brand:
            continue
        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == SCAN_ID,
            ScanBrandClassification.brand_id == brand.id,
        ).first()
        if sbc:
            if sbc.classification != "competitor" and not sbc.is_focus:
                sbc.classification = "competitor"
                sbc.classified_by = "user_bulk"
                sbc.source = "user_promoted_sister"
                sbc.updated_at = datetime.utcnow()
                promoted += 1
        else:
            db.add(ScanBrandClassification(
                scan_id=SCAN_ID, brand_id=brand.id,
                classification="competitor", is_focus=False,
                classified_by="user_bulk", source="user_promoted_sister",
            ))
            promoted += 1
    print(f"→ Promoted {promoted} sister brands → competitor")

    created_brands = created_gammes = classified = skipped_my_brand = 0

    def _classify_as_competitor(brand_id):
        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == SCAN_ID,
            ScanBrandClassification.brand_id == brand_id,
        ).first()
        if sbc is None:
            db.add(ScanBrandClassification(
                scan_id=SCAN_ID, brand_id=brand_id,
                classification="competitor", is_focus=False,
                classified_by="brief", source="brief_manual_edit",
            ))
            return "classified"
        if sbc.classification == "my_brand" or sbc.is_focus:
            return "skipped_my_brand"
        if sbc.classification != "competitor":
            sbc.classification = "competitor"
            sbc.classified_by = "brief"
            sbc.source = "brief_manual_edit"
            sbc.updated_at = datetime.utcnow()
            return "classified"
        return "already_competitor"

    def _get_or_create_brand(name, domain=None, parent_id=None):
        """Idempotent upsert keyed on the (client_id, name) UNIQUE constraint.
        Matches on name OR canonical_name so a brand is reused whether it sits as
        a root or as a child — avoids UniqueViolation on (client_id, name) when a
        compound name already exists as someone's gamme. Returns (brand, created).
        """
        name = (name or "").strip()
        if not name:
            return None, False
        name_norm = normalize_brand_name(name)
        existing = db.query(ClientBrand).filter(
            ClientBrand.client_id == scan.client_id,
            or_(ClientBrand.name == name, ClientBrand.canonical_name == name_norm),
        ).first()
        if existing:
            existing.last_seen_at = datetime.utcnow()
            if domain and not existing.domain:
                existing.domain = domain
            if parent_id and existing.parent_id is None:
                existing.parent_id = parent_id
            return existing, False
        b = ClientBrand(
            client_id=scan.client_id,
            name=name, canonical_name=name_norm,
            domain=domain, parent_id=parent_id,
            detected_in_scan_id=SCAN_ID,
            auto_detected=True, validated_by_user=False,
            detection_source="brief_manual_edit",
            last_seen_at=datetime.utcnow(),
        )
        db.add(b); db.flush()
        return b, True

    for comp in COMPETITORS_OVERRIDE:
        name = (comp.get("name") or "").strip()
        domain = (comp.get("domain") or "").strip().lower() or None
        if not name:
            continue

        root, was_created = _get_or_create_brand(name, domain=domain)
        if root is None:
            continue
        if was_created:
            created_brands += 1

        action = _classify_as_competitor(root.id)
        if action == "classified":
            classified += 1
        elif action == "skipped_my_brand":
            skipped_my_brand += 1
            continue

        for prod_name in comp.get("products", []):
            prod = (prod_name or "").strip()
            if not prod or prod.lower() == name.lower():
                continue
            gamme, g_created = _get_or_create_brand(prod, parent_id=root.id)
            if gamme is None:
                continue
            if g_created:
                created_gammes += 1
            _classify_as_competitor(gamme.id)

    # Prune stale off-category competitor gammes from a previous run (scan-scoped,
    # children only, never roots/focus). Keeps the competitor set aligned to the
    # current trimmed override on re-runs.
    desired_norms = set()
    for comp in COMPETITORS_OVERRIDE:
        desired_norms.add(normalize_brand_name(comp.get("name") or ""))
        for p in comp.get("products", []):
            desired_norms.add(normalize_brand_name(p))

    pruned = 0
    for sbc in db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == SCAN_ID,
        ScanBrandClassification.classification == "competitor",
        ScanBrandClassification.source == "brief_manual_edit",
    ).all():
        if sbc.is_focus:
            continue
        b = db.query(ClientBrand).filter(ClientBrand.id == sbc.brand_id).first()
        if not b or b.parent_id is None:
            continue
        if normalize_brand_name(b.name) not in desired_norms:
            db.delete(sbc)
            pruned += 1
    print(f"→ Pruned {pruned} stale competitor gamme classifications (off-category)")

    scan.updated_at = datetime.utcnow()
    db.commit()

    print(f"\n=========================")
    print(f"SUMMARY (fix_furterer_brief)")
    print(f"  sister brands promoted → competitor : {promoted}")
    print(f"  new competitor root brands created  : {created_brands}")
    print(f"  new gamme child brands created      : {created_gammes}")
    print(f"  SBC rows classified (new+reclassified): {classified}")
    print(f"  skipped (my_brand or focus)         : {skipped_my_brand}")
    print(f"  stale off-category gammes pruned    : {pruned}")

    counts = {}
    for cls, in db.query(ScanBrandClassification.classification).filter(
        ScanBrandClassification.scan_id == SCAN_ID
    ).all():
        counts[cls] = counts.get(cls, 0) + 1
    print(f"\nFinal classification counts:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:15s} → {n}")
    print(f"=========================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
