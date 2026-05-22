"""One-shot: align René Furterer scan topic taxonomy to the 3 seo-llm sources.

Twin of align_topics_ducray.py — adapted for René Furterer's 3 source CSVs
(cheveux-sec, chute, pousse) on renefurterer.com. Pure hair/scalp vertical.

Run via :
    SCAN_ID=<furterer-scan-uuid> docker exec senai-worker python /tmp/align_topics_furterer.py
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

# Target 3 topics matching seo-llm René Furterer sources.
TARGET_TOPICS = [
    {
        "key": "cheveux_secs",
        "name": "Cheveux secs et abîmés",
        "description": "Cheveux secs, déshydratés, abîmés, sensibilisés, nutrition et réparation de la fibre, masques et soins nourrissants — gammes Karité Nutri / Absolue Kératine.",
    },
    {
        "key": "chute",
        "name": "Chute de cheveux et alopécie",
        "description": "Chute de cheveux réactionnelle et chronique, cheveux affaiblis, fortifiants et stimulants du cuir chevelu, traitements anti-chute — gammes Triphasic / Forticea.",
    },
    {
        "key": "pousse",
        "name": "Pousse et densité capillaire",
        "description": "Croissance et densité des cheveux, cheveux clairsemés, stimulation de la pousse, soins énergisants et fortifiants — gammes RF80 / Forticea.",
    },
]


def _build_prompt(topics: list[ScanTopic], keywords: list[str]) -> str:
    topics_block = "\n".join(
        f'- "{t.id}" — {t.name}: {t.description or "(no description)"}'
        for t in topics
    )
    kws_block = "\n".join(f'{i+1}. {k}' for i, k in enumerate(keywords))
    return f"""Assign each keyword to ONE topic by its UUID, or "null" if it doesn't fit any.

# Active topics
{topics_block}
- "null" — keyword is out of scope for this brand (cosmetics generic, off-vertical, ambiguous).

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

    # 1. Snapshot current topics
    current = (
        db.query(ScanTopic)
        .filter(ScanTopic.scan_id == SCAN_ID)
        .order_by(ScanTopic.display_order)
        .all()
    )
    print(f"\nCurrent topics: {len(current)}")
    for t in current:
        kw = db.query(ScanKeyword).filter(ScanKeyword.topic_id == t.id).count()
        print(f"  {t.name!r} → {kw} kw, active={t.is_active}")

    # 2. Score each existing topic against each target.
    def _matches(existing_name: str, key: str) -> bool:
        n = (existing_name or "").lower()
        if key == "cheveux_secs":
            return "sec" in n or "abîm" in n or "abim" in n or "nutrition" in n or "réparation" in n or "kératine" in n
        if key == "chute":
            return "chute" in n or "alopéci" in n or "anti-chute" in n or "affaibli" in n
        if key == "pousse":
            return "pousse" in n or "densit" in n or "croissance" in n or "clairsem" in n or "fortifi" in n
        return False

    matched_target_to_existing: dict[str, ScanTopic] = {}
    for cfg in TARGET_TOPICS:
        for t in current:
            if _matches(t.name, cfg["key"]):
                if cfg["key"] not in matched_target_to_existing:
                    matched_target_to_existing[cfg["key"]] = t
    print(f"\nTarget → existing matches found:")
    for k, t in matched_target_to_existing.items():
        print(f"  {k:20s} ← {t.name!r}")

    # 3. Deactivate every non-matched existing topic.
    matched_ids = {str(t.id) for t in matched_target_to_existing.values()}
    deactivated = 0
    for t in current:
        if str(t.id) in matched_ids:
            t.is_active = True
            continue
        if t.is_active:
            t.is_active = False
            deactivated += 1
    print(f"\n→ Deactivated {deactivated} non-matched topics")

    # 4. For each target, ensure a topic exists (rename matched OR create new).
    max_order = (
        db.query(ScanTopic.display_order)
        .filter(ScanTopic.scan_id == SCAN_ID)
        .order_by(ScanTopic.display_order.desc())
        .first()
    )
    next_order = (max_order[0] or 0) + 1 if max_order else 1
    key_to_topic_id: dict[str, str] = {}

    for cfg in TARGET_TOPICS:
        existing = matched_target_to_existing.get(cfg["key"])
        if existing:
            existing.name = cfg["name"]
            existing.description = cfg["description"]
            existing.is_active = True
            key_to_topic_id[cfg["key"]] = str(existing.id)
        else:
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
            print(f"  + Created topic for '{cfg['key']}': {cfg['name']!r}")
    db.flush()

    # 5. Pool orphans from deactivated topics → topic_id = NULL.
    deactivated_topics = (
        db.query(ScanTopic)
        .filter(ScanTopic.scan_id == SCAN_ID, ScanTopic.is_active == False)
        .all()
    )
    orphan_kws = (
        db.query(ScanKeyword)
        .filter(
            ScanKeyword.scan_id == SCAN_ID,
            ScanKeyword.topic_id.in_([t.id for t in deactivated_topics]),
        )
        .all()
    )
    for kw in orphan_kws:
        kw.topic_id = None
    db.flush()
    null_kws = (
        db.query(ScanKeyword)
        .filter(ScanKeyword.scan_id == SCAN_ID, ScanKeyword.topic_id.is_(None))
        .all()
    )
    print(f"\nFreed orphans from inactive topics : {len(orphan_kws)}")
    print(f"Total to classify (incl. pre-NULL)  : {len(null_kws)}")

    # 6. Claude classifies the orphans
    valid_topic_ids = set(key_to_topic_id.values())
    active_topics = [t for t in db.query(ScanTopic).filter(
        ScanTopic.scan_id == SCAN_ID, ScanTopic.is_active == True
    ).order_by(ScanTopic.display_order).all() if str(t.id) in valid_topic_ids]
    assigned = nulled = 0
    n_batches = (len(null_kws) + _BATCH - 1) // _BATCH
    print(f"Classifying {len(null_kws)} keywords via Haiku (batches of {_BATCH}, {n_batches} total)…")
    for bi in range(n_batches):
        batch = null_kws[bi * _BATCH:(bi + 1) * _BATCH]
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

    # 7. Recompute keyword_count
    for t in active_topics:
        t.keyword_count = db.query(ScanKeyword).filter(ScanKeyword.topic_id == t.id).count()
    scan.updated_at = datetime.utcnow()
    db.commit()

    print(f"\n=========================")
    print(f"SUMMARY (align_topics René Furterer)")
    print(f"  topics deactivated : {deactivated}")
    print(f"  orphans processed  : {len(null_kws)}")
    print(f"  assigned           : {assigned}")
    print(f"  kept null          : {nulled}")
    print(f"\nFinal active topics:")
    for t in active_topics:
        print(f"  {t.name!r} → {t.keyword_count} keywords")
    print(f"=========================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
