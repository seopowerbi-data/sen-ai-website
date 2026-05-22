"""One-shot: fix the combined Pierre Fabre Oral Care scan brief + classifications.

The combined scan tracks 4 PF oral-care brands on one domain. This script:
  1. Sets scan.config.domain_brief.brands → the 4 oral-care brands (own brands)
  2. Sets scan.config.domain_brief.competitors → external oral-care brands + gammes
  3. Ensures the 4 oral-care brands are classified my_brand (never demotes focus)
  4. Demotes any cross-vertical PF derm sisters (Avène, Klorane, Ducray…) that
     leaked in as own/ignored → competitor (they are off-vertical noise here)
  5. Upserts client_brands + scan_brand_classifications for competitors

NOTE: this is a portfolio scan — there is no single focus brand. If the pipeline
set a focus, it is preserved; otherwise the dashboard aggregates the 4 my_brands.

Run:
    SCAN_ID=<oralcare-scan-uuid> \
      docker exec senai-worker python /tmp/fix_oralcare_brief.py
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

# The 4 PF oral-care brands tracked by this scan → my_brand.
MY_ORAL_BRANDS = [
    {"name": "Elgydium", "products": [
        "Elgydium Whitening", "Elgydium Anti-plaque", "Elgydium Sensibilité",
        "Elgydium Protection Caries", "Elgydium Baby", "Elgydium Kids",
        "Elgydium Junior", "Elgydium Ortho",
    ]},
    {"name": "Inava", "products": [
        "Inava Hybrid", "Inava Système", "Inava Brossettes", "Inava Fluid",
    ]},
    {"name": "Arthrodont", "products": [
        "Arthrodont Protect", "Arthrodont Classic",
    ]},
    {"name": "Eluane", "products": []},
]

# External oral-care competitors with their key gammes.
COMPETITORS_OVERRIDE = [
    {"name": "Sensodyne", "domain": "sensodyne.fr", "products": [
        "Sensodyne Répare & Protège", "Sensodyne Pro-Émail", "Sensodyne Sensibilité & Gencives",
    ]},
    {"name": "Parodontax", "domain": "parodontax.fr", "products": [
        "Parodontax Soin Gencives", "Parodontax Complete Protection",
    ]},
    {"name": "Colgate", "domain": "colgate.fr", "products": [
        "Colgate Total", "Colgate Elmex", "Colgate Sensitive Pro-Relief", "Colgate Max White",
    ]},
    {"name": "Elmex", "domain": "elmex.fr", "products": [
        "Elmex Protection Caries", "Elmex Sensitive", "Elmex Anti-Caries Junior",
    ]},
    {"name": "Meridol", "domain": "meridol.fr", "products": [
        "Meridol Protection Gencives", "Meridol Halitosis",
    ]},
    {"name": "Oral-B", "domain": "oralb.fr", "products": [
        "Oral-B Pro", "Oral-B iO", "Oral-B 3D White", "Oral-B Interdental",
    ]},
    {"name": "Signal", "domain": "signal.fr", "products": [
        "Signal Integral 8", "Signal White Now",
    ]},
    {"name": "Listerine", "domain": "listerine.fr", "products": [
        "Listerine Total Care", "Listerine Cool Mint",
    ]},
    {"name": "GUM", "domain": "gumshop.fr", "products": [
        "GUM Soft-Picks", "GUM Trav-Ler", "GUM Paroex",
    ]},
    {"name": "TePe", "domain": "tepe.com", "products": [
        "TePe Original", "TePe Angle", "TePe Interdental",
    ]},
    {"name": "Curaprox", "domain": "curaprox.com", "products": [
        "Curaprox CS 5460", "Curaprox CPS Prime",
    ]},
    {"name": "Email Diamant", "domain": "email-diamant.fr", "products": [
        "Email Diamant Le Blancheur",
    ]},
    {"name": "Sunstar", "domain": "sunstargum.com", "products": []},
]

# Cross-vertical PF derm sisters that should NOT be own brands in an oral-care
# scan. If they leaked in, demote → competitor (or you may prefer ignored).
DERM_SISTERS_TO_DEMOTE = [
    "eau thermale avène", "eau thermale avene", "avène", "avene",
    "klorane", "ducray", "rené furterer", "rene furterer", "a-derma", "aderma",
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

    # 1. Update the brief in place.
    cfg = dict(scan.config or {})
    brief = dict(cfg.get("domain_brief") or {})
    brief["brands"] = [b["name"] for b in MY_ORAL_BRANDS]
    brief["competitors"] = COMPETITORS_OVERRIDE
    cfg["domain_brief"] = brief
    cfg["domain_brief_manual_edit"] = datetime.utcnow().isoformat()
    scan.config = cfg
    flag_modified(scan, "config")
    print(f"→ Brief updated: brands={brief['brands']}, competitors={len(COMPETITORS_OVERRIDE)} entries")

    created_brands = created_gammes = 0

    def _upsert_brand(name, domain=None, parent_id=None):
        nonlocal created_brands, created_gammes
        name = (name or "").strip()
        if not name:
            return None
        name_norm = normalize_brand_name(name)
        # Match on name OR canonical_name regardless of parent: the (client_id,
        # name) UNIQUE constraint means a brand can already exist as someone's
        # child, and a parent_id-IS-NULL-only lookup would miss it then collide
        # on INSERT. Reuse whatever exists.
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
            return existing
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
        if parent_id is None:
            created_brands += 1
        else:
            created_gammes += 1
        return b

    def _set_classification(brand_id, classification, source):
        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == SCAN_ID,
            ScanBrandClassification.brand_id == brand_id,
        ).first()
        if sbc is None:
            db.add(ScanBrandClassification(
                scan_id=SCAN_ID, brand_id=brand_id,
                classification=classification, is_focus=False,
                classified_by="user_bulk", source=source,
            ))
            return True
        # Never silently flip a user-set focus.
        if sbc.is_focus:
            return False
        if sbc.classification != classification:
            sbc.classification = classification
            sbc.classified_by = "user_bulk"
            sbc.source = source
            sbc.updated_at = datetime.utcnow()
            return True
        return False

    # 2. Own brands → my_brand (+ gammes).
    my_classified = 0
    for ob in MY_ORAL_BRANDS:
        root = _upsert_brand(ob["name"])
        if _set_classification(root.id, "my_brand", "oralcare_my_brand"):
            my_classified += 1
        for prod in ob.get("products", []):
            g = _upsert_brand(prod, parent_id=root.id)
            if g:
                _set_classification(g.id, "my_brand", "oralcare_my_brand")
    print(f"→ my_brand classifications set/updated: {my_classified} roots (+gammes)")

    # 3. Competitors → competitor (+ gammes).
    comp_classified = 0
    for comp in COMPETITORS_OVERRIDE:
        root = _upsert_brand(comp["name"], domain=(comp.get("domain") or "").lower() or None)
        if _set_classification(root.id, "competitor", "oralcare_competitor"):
            comp_classified += 1
        for prod in comp.get("products", []):
            g = _upsert_brand(prod, parent_id=root.id)
            if g:
                _set_classification(g.id, "competitor", "oralcare_competitor")
    print(f"→ competitor classifications set/updated: {comp_classified} roots (+gammes)")

    # 4. Demote leaked cross-vertical derm sisters → competitor.
    demoted = 0
    for sname in DERM_SISTERS_TO_DEMOTE:
        brand = db.query(ClientBrand).filter(
            ClientBrand.client_id == scan.client_id,
            ClientBrand.canonical_name == normalize_brand_name(sname),
            ClientBrand.parent_id.is_(None),
        ).first()
        if not brand:
            continue
        if _set_classification(brand.id, "competitor", "oralcare_derm_sister_demote"):
            demoted += 1
    print(f"→ cross-vertical derm sisters demoted → competitor: {demoted}")

    scan.updated_at = datetime.utcnow()
    db.commit()

    print(f"\n=========================")
    print(f"SUMMARY (fix_oralcare_brief)")
    print(f"  new root brands created   : {created_brands}")
    print(f"  new gamme child created   : {created_gammes}")
    print(f"  my_brand roots set        : {my_classified}")
    print(f"  competitor roots set      : {comp_classified}")
    print(f"  derm sisters demoted      : {demoted}")

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
