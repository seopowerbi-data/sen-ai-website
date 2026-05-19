"""One-shot import of seo-llm personas + questions into a sen-ai scan.

Reads the 7 Avène cache files from seo-llm (personas_*.json + questions_*.json)
and replaces the auto-generated personas of the targeted sen-ai scan with the
seo-llm equivalents — preserving the rich persona data (intentions_recherche,
points_douleur, mots_cles_associes, opportunites, metriques) and per-question
signals (intention_cachee, signal_positif, signal_negatif) into
`scan_personas.data` JSONB.

The 7 sen-ai topics covered by seo-llm get their auto-gen personas DELETED and
replaced. The other ~7 topics keep their auto-gen personas (no equivalent in
seo-llm). Newly inserted questions get `intent_category = NULL` so the
classify_question_intent job (Phase B Tier A) will populate them.

Run via: docker exec senai-api python /tmp/import_seollm_avene.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/app')

from models import SessionLocal, Scan, ScanTopic, ScanPersona, ScanQuestion, Job

# --- Config ---------------------------------------------------------------

SCAN_ID = os.environ.get("SCAN_ID", "90604b64-021c-441a-85df-cc0623de95fd")
SEOLLM_CACHE = Path(os.environ.get("SEOLLM_CACHE", "/tmp/seollm_avene"))

# Topic mapping: seo-llm CSV slug → sen-ai topic name (case-sensitive prefix
# match — keeps things explicit, no silent fuzzy drift).
TOPIC_MAPPING = {
    "anti-âge-ave":     "Anti-âge",
    "antirougeur-ave":  "Rosacée et rougeurs",
    "cicatrisation-ave":"Soins Cicalfate",
    "cleanance-ave":    "Acné et peaux grasses",
    "eczéma-ave":       "Eczéma et dermatite",
    "hydrance-ave":     "Hydratation et peaux sèches",
    "solaire-ave":      "Protection solaire",
}


def find_topic(topics, prefix):
    """Match a sen-ai topic by its name starting with `prefix` (case-sensitive).
    Returns the ScanTopic instance or None."""
    for t in topics:
        if t.name.startswith(prefix):
            return t
    return None


def load_cache(slug):
    """Read personas_{slug}_03122025_3.json + questions_{slug}_03122025_3_5.json.
    Returns (personas_list, questions_list) or raises FileNotFoundError."""
    p = SEOLLM_CACHE / f"personas_{slug}_03122025_3.json"
    q = SEOLLM_CACHE / f"questions_{slug}_03122025_3_5.json"
    personas = json.loads(p.read_text(encoding="utf-8"))["personas"]
    questions = json.loads(q.read_text(encoding="utf-8"))["questions"]
    return personas, questions


def main():
    db = SessionLocal()
    scan = db.query(Scan).filter(Scan.id == SCAN_ID).first()
    if not scan:
        print(f"ERROR: scan {SCAN_ID} not found")
        return 1
    print(f"Scan: {scan.id} ({scan.domain}, status={scan.status})")

    topics = db.query(ScanTopic).filter(ScanTopic.scan_id == SCAN_ID).all()
    print(f"Scan has {len(topics)} topics in sen-ai")

    # Resolve seo-llm slug → sen-ai topic
    resolved = {}
    for slug, prefix in TOPIC_MAPPING.items():
        t = find_topic(topics, prefix)
        if not t:
            print(f"  ⚠ NO MATCH for '{slug}' (prefix='{prefix}') — will skip")
            continue
        resolved[slug] = t
        print(f"  ✓ {slug} → {t.name}")

    if len(resolved) != len(TOPIC_MAPPING):
        print(f"\nABORT: only {len(resolved)}/{len(TOPIC_MAPPING)} topics resolved.")
        print("Fix TOPIC_MAPPING prefixes and re-run.")
        return 1

    # For each resolved seo-llm CSV: load + import.
    total_personas = 0
    total_questions = 0
    skipped_questions = 0

    for slug, topic in resolved.items():
        try:
            personas, questions = load_cache(slug)
        except FileNotFoundError as e:
            print(f"  ✗ cache missing for '{slug}': {e}")
            return 1

        # DELETE existing personas of this topic (cascades to questions via FK)
        deleted_p = db.query(ScanPersona).filter(
            ScanPersona.scan_id == SCAN_ID,
            ScanPersona.topic_id == topic.id,
        ).delete(synchronize_session=False)
        db.flush()
        print(f"\n[{slug}] → '{topic.name}'")
        print(f"  deleted {deleted_p} auto-gen personas (questions cascaded)")

        # INSERT seo-llm personas, then their questions.
        # Index questions by persona_nom for the lookup.
        q_by_persona = {}
        for q in questions:
            q_by_persona.setdefault(q.get("persona_nom"), []).append(q)

        for p in personas:
            nom = p.get("nom") or "Unknown"
            # Embed the persona's questions in the JSONB blob so the rich
            # extras (intention_cachee, signal_positif, signal_negatif)
            # survive — matches the structure generate_personas writes.
            p_questions = q_by_persona.get(nom, [])
            data = dict(p)
            data["questions"] = [
                {
                    "type_question": q.get("type_question"),
                    "question": q.get("question"),
                    "intention_cachee": q.get("intention_cachee"),
                    "signal_positif": q.get("signal_positif"),
                    "signal_negatif": q.get("signal_negatif"),
                }
                for q in p_questions
            ]
            data["_source"] = {"origin": "seo-llm", "csv_slug": slug}

            new_persona = ScanPersona(
                id=uuid.uuid4(),
                scan_id=SCAN_ID,
                topic_id=topic.id,
                name=nom,
                data=data,
                is_active=True,
            )
            db.add(new_persona)
            db.flush()
            total_personas += 1

            # INSERT scan_questions
            for q in p_questions:
                text = (q.get("question") or "").strip()
                if len(text) < 10:
                    skipped_questions += 1
                    continue
                db.add(ScanQuestion(
                    id=uuid.uuid4(),
                    scan_id=SCAN_ID,
                    persona_id=new_persona.id,
                    question=text,
                    type_question=q.get("type_question") or "basique",
                    is_active=True,
                    intent_category=None,  # left for classify_question_intent
                ))
                total_questions += 1

        print(f"  inserted {len(personas)} personas, {sum(len(qs) for qs in q_by_persona.values())} questions")

    # Mark scan as imported (for traceability + downstream guards).
    cfg = scan.config or {}
    cfg["import_origin"] = "seo-llm"
    cfg["import_source_ids"] = sorted({3, 4, 11, 12, 15, 16, 25})
    cfg["import_timestamp"] = datetime.utcnow().isoformat()
    cfg["credits_already_debited"] = True
    scan.config = cfg
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(scan, "config")
    scan.updated_at = datetime.utcnow()

    # Enqueue classify_question_intent so the new NULL intent_categories get
    # populated before generate_opportunities can run (PR-1 guard would block
    # opportunities otherwise).
    job = Job(
        scan_id=SCAN_ID,
        client_id=scan.client_id,
        job_type="classify_question_intent",
        status="queued",
        payload={},
    )
    db.add(job)

    # Phase BB sync : enqueue generate_brand_brief for each primary brand
    # in this client that hasn't been briefed yet. Pre-seeds the per-brand
    # briefs at import time so the first generated article inherits them
    # without the user having to click Generate on each brand row.
    from models import Client as _ClientModel, ClientBrand as _ClientBrand
    _client = db.query(_ClientModel).filter(_ClientModel.id == scan.client_id).first()
    bb_enqueued = 0
    if _client and _client.primary_brand_ids:
        for bid in _client.primary_brand_ids:
            brand = db.query(_ClientBrand).filter(_ClientBrand.id == bid).first()
            if brand and brand.brief is None and \
                    int(brand.brief_generations_count or 0) < 3:
                db.add(Job(
                    client_id=scan.client_id,
                    job_type="generate_brand_brief",
                    status="queued",
                    payload={"brand_id": str(brand.id)},
                    max_attempts=2,
                ))
                bb_enqueued += 1

    db.commit()

    print(f"\n=========================")
    print(f"SUMMARY")
    print(f"  topics replaced  : {len(resolved)} / {len(topics)} total")
    print(f"  personas inserted: {total_personas}")
    print(f"  questions inserted: {total_questions}")
    print(f"  questions skipped (<10 chars): {skipped_questions}")
    print(f"  classify_question_intent job enqueued: {job.id}")
    print(f"  generate_brand_brief jobs enqueued: {bb_enqueued}")
    print(f"=========================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
