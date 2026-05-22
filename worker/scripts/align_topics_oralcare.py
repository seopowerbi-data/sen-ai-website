"""One-shot: align the combined Pierre Fabre Oral Care scan to the 13 seo-llm
oral-care sources (Elgydium ×8 + Inava ×3 + Arthrodont ×1 + Eluane ×1), all on
pierrefabre-oralcare.com.

Unlike the single-brand align_topics_* scripts, this uses a FRESH-CREATE strategy:
the 1:1 "match an existing auto-detected topic" heuristic breaks down with 13
overlapping oral-care concepts (e.g. Elgydium "gencives" vs Arthrodont gum gel
would both match a single detected "gencives" topic and fight over it). Instead we
deactivate every auto-detected topic, create all 13 targets fresh, pool every
keyword to NULL, and let Haiku redistribute across the 13 clean targets.

Each target name has a UNIQUE leading prefix so import_seollm_oralcare.py can
resolve slug → topic by startswith() without collisions.

Run via :
    SCAN_ID=<oralcare-scan-uuid> docker exec senai-worker python /tmp/align_topics_oralcare.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime

import httpx

sys.path.insert(0, '/app')

from models import SessionLocal, Scan, ScanTopic, ScanKeyword
from config import settings
from adapters.json_utils import extract_json_object

SCAN_ID = os.environ.get("SCAN_ID", "")
_BATCH = 60
_HAIKU = "claude-haiku-4-5-20251001"

# 13 oral-care targets. The `name` leading prefix must stay unique (the importer
# matches slug → topic by startswith) — keep the first 2-3 words distinct.
TARGET_TOPICS = [
    {"key": "bebe", "name": "Hygiène dentaire bébé (0-2 ans)",
     "description": "Première dent, poussée dentaire du nourrisson, hygiène bucco-dentaire des tout-petits, dentifrice et brosse adaptés 0-2 ans — gamme Elgydium Baby."},
    {"key": "enfant", "name": "Hygiène dentaire enfant (3-12 ans)",
     "description": "Brossage et dentifrice fluoré pour enfants, apprentissage de l'hygiène, dents de lait et dents définitives, goûts adaptés — gamme Elgydium Kids/Junior."},
    {"key": "blancheur", "name": "Blancheur et dents blanches",
     "description": "Dents jaunes, taches, dentifrices et soins blancheur, éclat de l'émail, blanchiment doux — gamme Elgydium Whitening."},
    {"key": "caries", "name": "Caries et prévention carie",
     "description": "Prévention des caries, déminéralisation de l'émail, fluor, dentifrices anti-caries, protection quotidienne — gamme Elgydium Protection Caries."},
    {"key": "gencives", "name": "Gencives et parodontie",
     "description": "Gencives qui saignent, gingivite, parodontite, soins protecteurs des gencives au quotidien, chlorhexidine — gamme Elgydium Anti-plaque/Gencives."},
    {"key": "ortho", "name": "Orthodontie et appareil dentaire",
     "description": "Soins bucco-dentaires avec bagues et appareil orthodontique, brossage spécifique, prévention caries sous appareil — gamme Elgydium Ortho."},
    {"key": "plaque", "name": "Plaque dentaire et tartre",
     "description": "Plaque bactérienne, tartre, dépôts dentaires, dentifrices anti-plaque, hygiène renforcée — gamme Elgydium Anti-plaque."},
    {"key": "sensibilite", "name": "Dents sensibles",
     "description": "Hypersensibilité dentinaire, douleur au chaud/froid, dentifrices désensibilisants, protection de l'émail fragilisé — gamme Elgydium Sensibilité."},
    {"key": "brossage", "name": "Brossage et brosses à dents manuelles",
     "description": "Technique de brossage, choix de la brosse à dents manuelle, souplesse des poils, fréquence et durée — gamme Inava brosses manuelles."},
    {"key": "electriques", "name": "Brosses à dents électriques",
     "description": "Brosses à dents électriques, têtes de rechange, modes de brossage, efficacité vs manuelle — gamme Inava électrique."},
    {"key": "interdentaire", "name": "Hygiène interdentaire (brossettes, fil)",
     "description": "Nettoyage entre les dents, brossettes interdentaires, fil dentaire, espaces interdentaires, gencives saines — gamme Inava interdentaire."},
    {"key": "gel_gencive", "name": "Gel gingival apaisant (gencives irritées)",
     "description": "Gel gingival pour gencives douloureuses, irritées ou enflammées, poussées dentaires douloureuses, aphtes et irritations buccales — gamme Arthrodont."},
    {"key": "bouche_seche", "name": "Bouche sèche et xérostomie",
     "description": "Sécheresse buccale, xérostomie, manque de salive lié à l'âge ou aux médicaments, hydratation et confort de la bouche — gamme Eluane/Eluday."},
]


def _build_prompt(topics: list[ScanTopic], keywords: list[str]) -> str:
    topics_block = "\n".join(
        f'- "{t.id}" — {t.name}: {t.description or "(no description)"}'
        for t in topics
    )
    kws_block = "\n".join(f'{i+1}. {k}' for i, k in enumerate(keywords))
    return f"""Assign each keyword to ONE topic by its UUID, or "null" if it doesn't fit any.
All topics belong to the oral-care / dental-hygiene vertical.

# Active topics
{topics_block}
- "null" — keyword is out of scope (not oral-care, generic, ambiguous).

# Keywords
{kws_block}

# Output (JSON only, same order)
{{
  "assignments": [
    {{"i": 1, "topic": "<topic-uuid or 'null'>"}}
  ]
}}

Conservative: when in doubt → "null"."""


async def _call(api_key: str, prompt: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": _HAIKU,
                "max_tokens": 4096,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return resp.json()


def main():
    if not SCAN_ID:
        print("ERROR: SCAN_ID env var required")
        return 1
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY missing")
        return 1

    db = SessionLocal()
    scan = db.query(Scan).filter(Scan.id == SCAN_ID).first()
    if not scan:
        print(f"ERROR: scan {SCAN_ID} not found")
        return 1
    print(f"Scan: {scan.id} ({scan.domain}, status={scan.status})")

    # 1. Snapshot + deactivate ALL current topics (fresh-create strategy).
    current = (
        db.query(ScanTopic)
        .filter(ScanTopic.scan_id == SCAN_ID)
        .order_by(ScanTopic.display_order)
        .all()
    )
    print(f"\nCurrent topics: {len(current)} (all will be deactivated)")
    deactivated = 0
    for t in current:
        kw = db.query(ScanKeyword).filter(ScanKeyword.topic_id == t.id).count()
        print(f"  {t.name!r} → {kw} kw, active={t.is_active}")
        if t.is_active:
            t.is_active = False
            deactivated += 1
    db.flush()
    print(f"\n→ Deactivated {deactivated} auto-detected topics")

    # 2. Create all 13 targets fresh.
    max_order = (
        db.query(ScanTopic.display_order)
        .filter(ScanTopic.scan_id == SCAN_ID)
        .order_by(ScanTopic.display_order.desc())
        .first()
    )
    next_order = (max_order[0] or 0) + 1 if max_order else 1
    key_to_topic_id: dict[str, str] = {}
    for cfg in TARGET_TOPICS:
        new_t = ScanTopic(
            id=uuid.uuid4(),
            scan_id=SCAN_ID,
            name=cfg["name"],
            description=cfg["description"],
            keyword_count=0,
            is_active=True,
            display_order=next_order,
        )
        next_order += 1
        db.add(new_t)
        db.flush()
        key_to_topic_id[cfg["key"]] = str(new_t.id)
        print(f"  + Created topic '{cfg['key']}': {cfg['name']!r}")
    db.flush()

    # 3. Pool every keyword to NULL (orphans from now-inactive topics + pre-NULL).
    all_kws = (
        db.query(ScanKeyword)
        .filter(ScanKeyword.scan_id == SCAN_ID)
        .all()
    )
    for kw in all_kws:
        kw.topic_id = None
    db.flush()
    print(f"\nKeywords pooled to NULL for redistribution: {len(all_kws)}")

    # 4. Haiku classifies every keyword across the 13 fresh targets.
    valid_topic_ids = set(key_to_topic_id.values())
    active_topics = [t for t in db.query(ScanTopic).filter(
        ScanTopic.scan_id == SCAN_ID, ScanTopic.is_active == True
    ).order_by(ScanTopic.display_order).all() if str(t.id) in valid_topic_ids]
    assigned = nulled = 0
    n_batches = (len(all_kws) + _BATCH - 1) // _BATCH
    print(f"Classifying {len(all_kws)} keywords via Haiku (batches of {_BATCH}, {n_batches} total)…")
    for bi in range(n_batches):
        batch = all_kws[bi * _BATCH:(bi + 1) * _BATCH]
        prompt = _build_prompt(active_topics, [k.keyword for k in batch])
        try:
            resp = asyncio.run(_call(settings.anthropic_api_key, prompt))
            parsed = extract_json_object(resp["content"][0]["text"])
            by_idx = {e.get("i"): e.get("topic") for e in parsed.get("assignments", [])}
        except Exception as e:
            print(f"  ✗ batch {bi+1}/{n_batches} failed: {e}")
            continue
        la = ln = 0
        for i, kw in enumerate(batch, 1):
            tid = by_idx.get(i)
            if tid and tid in valid_topic_ids:
                kw.topic_id = tid
                assigned += 1
                la += 1
            else:
                nulled += 1
                ln += 1
        db.commit()
        print(f"  ✓ batch {bi+1}/{n_batches}: {la} assigned, {ln} → null")

    # 5. Recompute keyword_count
    for t in active_topics:
        t.keyword_count = db.query(ScanKeyword).filter(ScanKeyword.topic_id == t.id).count()
    scan.updated_at = datetime.utcnow()
    db.commit()

    print(f"\n=========================")
    print(f"SUMMARY (align_topics Oral Care)")
    print(f"  topics deactivated : {deactivated}")
    print(f"  targets created    : {len(TARGET_TOPICS)}")
    print(f"  keywords processed : {len(all_kws)}")
    print(f"  assigned           : {assigned}")
    print(f"  kept null          : {nulled}")
    print(f"\nFinal active topics:")
    for t in active_topics:
        print(f"  {t.name!r} → {t.keyword_count} keywords")
    print(f"=========================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
