"""One-shot: restore the Klorane scan to the curated, relevance-clean brief and
RESET its competitor classifications to exactly that set.

Needed after a `Regenerate from web` run polluted the scan: the Gemini fallback
added off-category gammes (Dexyane, Melascreen…) and extra roots (Nuxe, Weleda…),
and the Gate-2 pre-population only ever ADDS, so the competitor set ballooned
(69 → 143) on top of original auto-brief leftovers.

This script:
  1. Sets brief.brands=['Klorane'] + brief.competitors=CURATED (trimmed, named
     gammes, relevance-filtered) and marks edited_by_user=True (protect it).
  2. Upserts the curated competitors + gammes and classifies them competitor.
  3. RESETS: deletes every scan competitor SBC row whose brand is NOT in the
     curated desired set (roots + gammes), excluding focus. my_brand/ignored/
     unclassified rows are untouched. ClientBrand rows are kept (scan-scoped).

Run:
    SCAN_ID=13b8dcfc-f6f8-42c6-9707-06ebbb9350a4 \
      docker exec senai-worker python /tmp/restore_klorane_brief.py
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

# Curated, relevance-clean competitor set (named gammes overlapping Klorane's
# hair + baby categories only). Mirrors fix_klorane_brief.COMPETITORS_OVERRIDE.
CURATED = [
    {"name": "René Furterer", "domain": "renefurterer.com", "products": [
        "Triphasic", "Forticea", "Naturia", "Astera", "Complexe 5"]},
    {"name": "Ducray", "domain": "ducray.com", "products": [
        "Anaphase", "Neoptide", "Squanorm", "Kelual DS", "Kertyol"]},
    {"name": "Avène", "domain": "eau-thermale-avene.fr", "products": ["Pédiatril"]},
    {"name": "A-Derma", "domain": "aderma.fr", "products": ["Primalba"]},
    {"name": "Phyto", "domain": "phyto.com", "products": [
        "Phytocyane", "Phytonovathrix", "Phytodéfrisant", "Phytophanère"]},
    {"name": "Luxéol", "domain": "luxeol.fr", "products": [
        "Anti-chute", "Cheveux et ongles", "Pousse"]},
    {"name": "Vichy", "domain": "vichy.fr", "products": [
        "Dercos Aminexil", "Dercos Anti-pelliculaire", "Dercos Densi-Solutions"]},
    {"name": "Kérastase", "domain": "kerastase.fr", "products": [
        "Nutritive", "Spécifique", "Genesis"]},
    {"name": "Forté Pharma", "domain": "fortepharma.com", "products": ["Forcapil"]},
    {"name": "La Roche-Posay", "domain": "laroche-posay.fr", "products": [
        "Kerium", "Kerium DS", "Kerium Anti-chute"]},
    {"name": "L'Oréal Paris", "domain": "loreal-paris.fr", "products": ["Elseve", "Elvive"]},
    {"name": "Garnier", "domain": "garnier.fr", "products": ["Ultra Doux", "Fructis"]},
    {"name": "Mustela", "domain": "mustela.fr", "products": [
        "Stelatopia", "Hydra Bébé", "Liniment", "Gel lavant"]},
    {"name": "Bioderma", "domain": "bioderma.fr", "products": ["ABCDerm", "Nodé"]},
    {"name": "Uriage", "domain": "uriage.fr", "products": ["Bébé 1er", "DS Hair"]},
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

    desired_norms = set()
    for comp in CURATED:
        desired_norms.add(normalize_brand_name(comp["name"]))
        for p in comp.get("products", []):
            desired_norms.add(normalize_brand_name(p))

    # 1. Restore the curated brief, protect it from auto-regen.
    cfg = dict(scan.config or {})
    brief = dict(cfg.get("domain_brief") or {})
    brief["brands"] = ["Klorane"]
    brief["competitors"] = CURATED
    brief["edited_by_user"] = True
    cfg["domain_brief"] = brief
    cfg["domain_brief_provider"] = "manual_curated"
    cfg["domain_brief_manual_edit"] = datetime.utcnow().isoformat()
    scan.config = cfg
    flag_modified(scan, "config")
    print(f"→ Brief restored: brands=['Klorane'], competitors={len(CURATED)}, edited_by_user=True")

    def _get_or_create_brand(name, domain=None, parent_id=None):
        name = (name or "").strip()
        if not name:
            return None
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
            return existing
        b = ClientBrand(
            client_id=scan.client_id, name=name, canonical_name=name_norm,
            domain=domain, parent_id=parent_id, detected_in_scan_id=SCAN_ID,
            auto_detected=True, validated_by_user=False,
            detection_source="manual_curated", last_seen_at=datetime.utcnow(),
        )
        db.add(b); db.flush()
        return b

    def _ensure_competitor(brand_id):
        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == SCAN_ID,
            ScanBrandClassification.brand_id == brand_id,
        ).first()
        if sbc is None:
            db.add(ScanBrandClassification(
                scan_id=SCAN_ID, brand_id=brand_id, classification="competitor",
                is_focus=False, classified_by="user_bulk", source="manual_curated"))
            return
        if sbc.is_focus or sbc.classification == "my_brand":
            return
        if sbc.classification != "competitor":
            sbc.classification = "competitor"
            sbc.classified_by = "user_bulk"
            sbc.source = "manual_curated"
            sbc.updated_at = datetime.utcnow()

    # 2. Upsert + classify the curated set.
    for comp in CURATED:
        root = _get_or_create_brand(comp["name"], domain=(comp.get("domain") or "").lower() or None)
        _ensure_competitor(root.id)
        for prod in comp.get("products", []):
            if not prod or prod.lower() == comp["name"].lower():
                continue
            g = _get_or_create_brand(prod, parent_id=root.id)
            if g:
                _ensure_competitor(g.id)

    # 3. RESET — delete competitor SBC rows whose brand is not in the curated set.
    deleted = 0
    kept = 0
    for sbc in db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == SCAN_ID,
        ScanBrandClassification.classification == "competitor",
    ).all():
        if sbc.is_focus:
            kept += 1
            continue
        b = db.query(ClientBrand).filter(ClientBrand.id == sbc.brand_id).first()
        if b and normalize_brand_name(b.name) in desired_norms:
            kept += 1
            continue
        db.delete(sbc)
        deleted += 1
    print(f"→ Reset competitors: kept {kept} (curated), deleted {deleted} (regen + auto-brief leftovers)")

    scan.updated_at = datetime.utcnow()
    db.commit()

    counts = {}
    for cls, in db.query(ScanBrandClassification.classification).filter(
        ScanBrandClassification.scan_id == SCAN_ID).all():
        counts[cls] = counts.get(cls, 0) + 1
    print(f"\nFinal classification counts: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
