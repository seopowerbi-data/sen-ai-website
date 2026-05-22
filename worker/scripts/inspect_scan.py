"""Read-only: dump a scan's status, topics (+kw counts), persona/question counts,
and brief brands/competitors. Run via:
    SCAN_ID=<uuid> docker exec senai-worker python /tmp/inspect_scan.py
"""
import os, sys
sys.path.insert(0, '/app')
from models import (SessionLocal, Scan, ScanTopic, ScanKeyword, ScanPersona,
                    ScanQuestion, ScanBrandClassification, ClientBrand)

SCAN_ID = os.environ.get("SCAN_ID", "")
db = SessionLocal()
scan = db.query(Scan).filter(Scan.id == SCAN_ID).first()
if not scan:
    print(f"NOT FOUND: {SCAN_ID}"); sys.exit(1)
print(f"Scan {scan.id}")
print(f"  domain={scan.domain}  status={scan.status}  type={scan.scan_type}")
print(f"  run_index={scan.run_index}  focus_brand_id={scan.focus_brand_id}")

topics = db.query(ScanTopic).filter(ScanTopic.scan_id == SCAN_ID).order_by(ScanTopic.display_order).all()
print(f"\nTopics ({len(topics)}):")
for t in topics:
    kw = db.query(ScanKeyword).filter(ScanKeyword.topic_id == t.id).count()
    pc = db.query(ScanPersona).filter(ScanPersona.topic_id == t.id).count()
    print(f"  [{'A' if t.is_active else '.'}] {t.name!r}  kw={kw} personas={pc}")

null_kw = db.query(ScanKeyword).filter(ScanKeyword.scan_id == SCAN_ID, ScanKeyword.topic_id.is_(None)).count()
total_kw = db.query(ScanKeyword).filter(ScanKeyword.scan_id == SCAN_ID).count()
tp = db.query(ScanPersona).filter(ScanPersona.scan_id == SCAN_ID).count()
tq = db.query(ScanQuestion).filter(ScanQuestion.scan_id == SCAN_ID).count()
print(f"\nKeywords: total={total_kw}  null_topic={null_kw}")
print(f"Personas total={tp}  Questions total={tq}")

cfg = scan.config or {}
brief = cfg.get("domain_brief") or {}
print(f"\nBrief brands: {brief.get('brands')}")
comps = brief.get('competitors') or []
print(f"Brief competitors ({len(comps)}): {[c.get('name') if isinstance(c, dict) else c for c in comps][:20]}")
print(f"import_origin={cfg.get('import_origin')}")

sbc = db.query(ScanBrandClassification).filter(ScanBrandClassification.scan_id == SCAN_ID).all()
counts = {}
for r in sbc:
    counts[r.classification] = counts.get(r.classification, 0) + 1
print(f"\nBrand classifications: {counts}")
foc = [r for r in sbc if r.is_focus]
for r in foc:
    b = db.query(ClientBrand).filter(ClientBrand.id == r.brand_id).first()
    print(f"  FOCUS: {b.name if b else r.brand_id}")
