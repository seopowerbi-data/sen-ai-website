"""One-shot import of the combined Pierre Fabre Oral Care seo-llm personas +
questions into a single sen-ai scan (domain pierrefabre-oralcare.com).

Covers 4 brands that share one domain:
  - Elgydium ×8  (bebe, blancheur, caries, enfant, gencives, ortho, plaque, sensibilite)
  - Inava ×3     (brossage, electriques, interdentaire)
  - Arthrodont ×1 (gencive)
  - Eluane/Eluday ×1 (hygienebouche)
  → 13 sources, 78 personas, 1170 questions per local seo-llm cache snapshot.

Twin of import_seollm_ducray.py, extended to per-slug brand attribution.

Run via:
    SCAN_ID=<the-oralcare-scan-uuid> \
      SEOLLM_CACHE=/tmp/seollm_oralcare \
      docker exec senai-worker python /tmp/import_seollm_oralcare.py
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

SCAN_ID = os.environ.get("SCAN_ID", "")
SEOLLM_CACHE = Path(os.environ.get("SEOLLM_CACHE", "/tmp/seollm_oralcare"))

# seo-llm slug → sen-ai topic name prefix. Prefixes MUST match the unique-prefix
# names created by align_topics_oralcare.py (case-sensitive startswith).
TOPIC_MAPPING = {
    # Elgydium
    "bebe-elg":          "Hygiène dentaire bébé",
    "enfant-elg":        "Hygiène dentaire enfant",
    "blancheur-elg":     "Blancheur",
    "caries-elg":        "Caries",
    "gencives-elg":      "Gencives et parodontie",
    "ortho-elg":         "Orthodontie",
    "plaque-elg":        "Plaque",
    "sensibilite-elg":   "Dents sensibles",
    # Inava
    "brossage-ina":      "Brossage",
    "electriques-ina":   "Brosses à dents électriques",
    "interdentaire-ina": "Hygiène interdentaire",
    # Arthrodont
    "gencive-art":       "Gel gingival",
    # Eluane / Eluday
    "hygienebouche-elu": "Bouche sèche",
}

# slug → brand label recorded in persona data._source.brand
SLUG_BRAND = {
    "bebe-elg": "elgydium", "enfant-elg": "elgydium", "blancheur-elg": "elgydium",
    "caries-elg": "elgydium", "gencives-elg": "elgydium", "ortho-elg": "elgydium",
    "plaque-elg": "elgydium", "sensibilite-elg": "elgydium",
    "brossage-ina": "inava", "electriques-ina": "inava", "interdentaire-ina": "inava",
    "gencive-art": "arthrodont",
    "hygienebouche-elu": "eluane",
}

PERSONAS_PER_SOURCE = {slug: 3 for slug in TOPIC_MAPPING}

ORALCARE_SOURCE_IDS = [26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40]


def _candidate_question_paths(slug: str, nb_personas: int) -> list[Path]:
    return [
        SEOLLM_CACHE / f"questions_{slug}_03122025_{nb_personas}_5.json",
        SEOLLM_CACHE / f"questions_{slug}_03122025_{nb_personas}_15.json",
    ]


def find_topic(topics, prefix):
    # Filter is_active=True so a deactivated auto-detected topic can't steal the
    # match from the freshly-created target.
    for t in topics:
        if t.is_active and t.name.startswith(prefix):
            return t
    return None


def load_cache(slug):
    n = PERSONAS_PER_SOURCE.get(slug, 3)
    p = SEOLLM_CACHE / f"personas_{slug}_03122025_{n}.json"
    personas = json.loads(p.read_text(encoding="utf-8"))["personas"]
    questions = []
    for qpath in _candidate_question_paths(slug, n):
        if qpath.exists():
            questions = json.loads(qpath.read_text(encoding="utf-8"))["questions"]
            break
    if not questions:
        raise FileNotFoundError(
            f"No questions file for slug '{slug}'. Tried: "
            f"{[str(p) for p in _candidate_question_paths(slug, n)]}"
        )
    return personas, questions


def main():
    if not SCAN_ID:
        print("ERROR: SCAN_ID env var required")
        return 1
    db = SessionLocal()
    scan = db.query(Scan).filter(Scan.id == SCAN_ID).first()
    if not scan:
        print(f"ERROR: scan {SCAN_ID} not found")
        return 1
    print(f"Scan: {scan.id} ({scan.domain}, status={scan.status})")

    topics = db.query(ScanTopic).filter(ScanTopic.scan_id == SCAN_ID).all()
    print(f"Scan has {len(topics)} topics in sen-ai")

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
        print("Fix TOPIC_MAPPING prefixes (top of script) and re-run.")
        print("Available active topics:")
        for t in topics:
            if t.is_active:
                print(f"  - {t.name!r}")
        return 1

    total_personas = total_questions = skipped_questions = 0

    for slug, topic in resolved.items():
        try:
            personas, questions = load_cache(slug)
        except FileNotFoundError as e:
            print(f"  ✗ cache missing for '{slug}': {e}")
            return 1

        deleted_p = db.query(ScanPersona).filter(
            ScanPersona.scan_id == SCAN_ID,
            ScanPersona.topic_id == topic.id,
        ).delete(synchronize_session=False)
        db.flush()
        print(f"\n[{slug}] → '{topic.name}' (brand={SLUG_BRAND.get(slug)})")
        print(f"  deleted {deleted_p} auto-gen personas (questions cascaded)")

        q_by_persona = {}
        for q in questions:
            q_by_persona.setdefault(q.get("persona_nom"), []).append(q)

        for p in personas:
            nom = p.get("nom") or "Unknown"
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
            data["_source"] = {
                "origin": "seo-llm", "csv_slug": slug,
                "brand": SLUG_BRAND.get(slug, "oralcare"),
            }

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
                    intent_category=None,
                ))
                total_questions += 1

        print(f"  inserted {len(personas)} personas, "
              f"{sum(len(qs) for qs in q_by_persona.values())} questions")

    cfg = scan.config or {}
    cfg["import_origin"] = "seo-llm"
    cfg["import_source_ids"] = sorted(ORALCARE_SOURCE_IDS)
    cfg["import_brand"] = "pierre-fabre-oral-care"
    cfg["import_brands"] = ["elgydium", "inava", "arthrodont", "eluane"]
    cfg["import_timestamp"] = datetime.utcnow().isoformat()
    cfg["credits_already_debited"] = True
    scan.config = cfg
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(scan, "config")
    scan.updated_at = datetime.utcnow()

    job = Job(
        scan_id=SCAN_ID,
        client_id=scan.client_id,
        job_type="classify_question_intent",
        status="pending",
        payload={},
    )
    db.add(job)

    db.commit()

    print(f"\n=========================")
    print(f"SUMMARY (Pierre Fabre Oral Care import)")
    print(f"  topics replaced  : {len(resolved)} / {len(topics)} total")
    print(f"  personas inserted: {total_personas}")
    print(f"  questions inserted: {total_questions}")
    print(f"  questions skipped (<10 chars): {skipped_questions}")
    print(f"  classify_question_intent job enqueued: {job.id}")
    print(f"=========================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
