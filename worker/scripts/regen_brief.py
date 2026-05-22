"""Re-trigger domain-brief generation for a scan.

Backs up the current brief (reversible), clears the `edited_by_user` guard so the
handler won't skip, and enqueues a generate_domain_brief job (picked up by
senai-worker). Run via:
    SCAN_ID=<uuid> docker exec senai-worker python /tmp/regen_brief.py
"""
import os, sys
sys.path.insert(0, '/app')
from datetime import datetime
from sqlalchemy.orm.attributes import flag_modified
from models import SessionLocal, Scan, Job

SCAN_ID = os.environ.get("SCAN_ID", "")
db = SessionLocal()
s = db.query(Scan).filter(Scan.id == SCAN_ID).first()
if not s:
    print(f"NOT FOUND: {SCAN_ID}"); sys.exit(1)

cfg = dict(s.config or {})
brief = dict(cfg.get("domain_brief") or {})
# Reversible backup of the current (curated) brief.
cfg["domain_brief_backup"] = brief
# Clear the guard so generate_domain_brief.execute() won't early-return.
brief = dict(brief)
brief["edited_by_user"] = False
cfg["domain_brief"] = brief
s.config = cfg
flag_modified(s, "config")

job = Job(
    scan_id=SCAN_ID,
    client_id=s.client_id,
    job_type="generate_domain_brief",
    status="pending",
    payload={},
)
db.add(job)
db.commit()
print(f"Scan {SCAN_ID} ({s.domain})")
print(f"  backup saved to config['domain_brief_backup'] ({len(brief.get('competitors') or [])} competitors)")
print(f"  edited_by_user cleared")
print(f"  enqueued generate_domain_brief job {job.id} (status=pending)")
