"""Handler : Sprint 8 Reddit opportunity finder (v1 contexte-only).

Background : Reddit closed their Data API for commercial use in 2023 and
blocks cloud-provider IPs on the public .json endpoint (confirmed May
2026 - Hetzner IP → HTTP 403 regardless of User-Agent). Their commercial
tier starts at $12k/mo enterprise. We don't qualify for the free
non-commercial tier as a SaaS.

So v1 mines what we already have legitimately : the LLM citation
snippets in `scan_llm_results.citations[].contexte`. For each unique
Reddit URL the LLMs cite in this scan we :

  1. Aggregate the LLM citation snippets (each ~200 chars) into a single
     "what the LLMs said about this thread" corpus.
  2. Parse the subreddit from the URL.
  3. Regex-match brand mentions (target + competitors) on the corpus.
  4. Classify : competitor_wins (competitor named, target absent) /
     you_win (target named) / neutral.
  5. Run a Haiku sentiment pass on the snippets if at least one brand is
     in scope (saves budget on context-noise threads).
  6. Score by `citation_count × classification_weight × sentiment_lever`.

The user clicks an URL to read the full thread in their own browser
(residential IP works fine ; Reddit only blocks server IPs). Same
ethical stance as Sprint 7 #17 - sen-ai is the SaaS that doesn't scrape.

Cost : ~$0.0003 per Haiku call × 100 threads × ~half-detected = ~$0.015
per scan worst case. Bounded by the 100-thread cap and the brand-detected
gate.

If Reddit ever re-opens commercial API access, the OAuth fetcher lives
in commit f21bc32 - restore and swap the URL source ; the rest of the
pipeline keeps working.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

from sqlalchemy import text as _text
from sqlalchemy.orm import Session

from adapters.reddit_client import canonical_url, is_reddit_url, parse_subreddit, parse_title_slug
from adapters.reddit_sentiment import classify_snippets

logger = logging.getLogger(__name__)

MAX_THREADS_PER_RUN = 100
MIN_BRAND_LEN_FOR_REGEX = 3


def _cited_reddit_threads(db: Session, scan_id: str, limit: int) -> list[dict]:
    """Mine every Reddit thread URL cited by an LLM during this scan.
    Returns one entry per canonical URL with the aggregated metadata :

        {
          url, subreddit,
          citation_count,                # how many LLM responses cited it
          contextes : [snippet, ...],    # for the Haiku sentiment pass
          winning_questions : [{question_id, question, provider, contexte, slr_id}],
        }
    """
    sql = _text(
        """
        SELECT slr.id::text AS slr_id,
               slr.question_id::text AS question_id,
               sq.question AS question,
               slr.provider AS provider,
               citation->>'url' AS raw_url,
               lower(citation->>'domaine') AS domaine,
               citation->>'contexte' AS contexte
          FROM scan_llm_results slr
          JOIN LATERAL jsonb_array_elements(slr.citations) AS citation ON true
          LEFT JOIN scan_questions sq ON sq.id = slr.question_id
         WHERE slr.scan_id = :scan_id
           AND citation->>'url' IS NOT NULL
           AND (lower(citation->>'domaine') LIKE '%reddit.com'
                OR citation->>'url' ILIKE '%reddit.com/%')
        """
    )
    raw_rows = db.execute(sql, {"scan_id": scan_id}).fetchall()

    bucket: dict[str, dict] = {}
    for r in raw_rows:
        url = r.raw_url
        if not url or not is_reddit_url(url):
            continue
        canonical = canonical_url(url)
        b = bucket.get(canonical)
        if b is None:
            b = {
                "url": canonical,
                "subreddit": parse_subreddit(canonical),
                "citation_count": 0,
                "contextes": [],
                "winning_questions": [],
            }
            bucket[canonical] = b
        b["citation_count"] += 1
        contexte = (r.contexte or "").strip()
        if contexte and contexte not in b["contextes"]:
            b["contextes"].append(contexte)
        if r.question:
            key = (r.question_id, r.provider)
            existing = {(q.get("question_id"), q.get("provider")) for q in b["winning_questions"]}
            if key not in existing:
                b["winning_questions"].append({
                    "question_id": r.question_id,
                    "question": r.question,
                    "provider": r.provider,
                    "contexte": contexte,
                    "slr_id": r.slr_id,
                })

    out = sorted(bucket.values(), key=lambda x: -x["citation_count"])
    return out[:limit]


def _scan_brands(db: Session, scan_id: str) -> tuple[set[str], set[str]]:
    """Return (target_names, competitor_names) - lowercased canonical names
    + aliases of every brand classified for this scan."""
    target: set[str] = set()
    competitor: set[str] = set()
    rows = db.execute(_text(
        """
        SELECT cb.name, cb.canonical_name, cb.aliases, sbc.classification
          FROM scan_brand_classifications sbc
          JOIN client_brands cb ON cb.id = sbc.brand_id
         WHERE sbc.scan_id = :scan_id
           AND sbc.classification IN ('my_brand', 'competitor')
        """
    ), {"scan_id": scan_id}).fetchall()
    for name, canonical, aliases, cls in rows:
        names = {n for n in [name, canonical, *(aliases or [])] if n}
        cleaned = {n.lower().strip() for n in names if n and len(n) >= MIN_BRAND_LEN_FOR_REGEX}
        if cls == "my_brand":
            target |= cleaned
        else:
            competitor |= cleaned
    return target, competitor


def _detect_brands(corpus_lower: str, candidates: set[str]) -> set[str]:
    """Find which brand names appear as whole words in the lowercased
    corpus. Skip very short tokens so we don't match 'fr' inside text."""
    if not corpus_lower or not candidates:
        return set()
    hits: set[str] = set()
    for name in candidates:
        if len(name) < MIN_BRAND_LEN_FOR_REGEX:
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", corpus_lower):
            hits.add(name)
    return hits


def _classify(target_hits: set[str], competitor_hits: set[str]) -> str:
    if competitor_hits and not target_hits:
        return "competitor_wins"
    if target_hits:
        return "you_win"
    return "neutral"


def _leverage_score(citation_count: int, classification: str, sentiment: str | None) -> int:
    """Composite 0-100 priority score for the contexte-only mode.

    Without upvotes/comment counts we use `citation_count` as engagement
    proxy : the more LLMs cite this thread, the broader its visibility
    in the AI-search ecosystem. Caps at 8+ citations = full engagement
    points (rare ; most threads get 1-3).

    Same breakdown as the original full-thread version :
      55 engagement (now : citation_count log-normalized to 8)
      25 classification (competitor_wins=25, neutral=10, you_win=0)
      20 sentiment lever (negative=20, neutral/mixed/None=10, positive=0)
    """
    import math

    cc = max(0, int(citation_count or 0))
    engagement_raw = math.log10(cc + 1) / math.log10(9)  # log scale, cap at 8 = 1.0
    engagement = min(55, int(round(engagement_raw * 55)))

    cls_pts = {"competitor_wins": 25, "neutral": 10, "you_win": 0}.get(classification, 0)

    if sentiment == "negative":
        sent_pts = 20
    elif sentiment == "positive":
        sent_pts = 0
    else:
        sent_pts = 10

    return max(0, min(100, engagement + cls_pts + sent_pts))


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Audit Reddit threads cited by LLMs in this scan, contexte-only mode.

    job_payload :
      - reset (bool)         : drop existing rows before re-running
      - limit (int)          : cap thread count (default MAX_THREADS_PER_RUN)
      - sentiment (bool)     : run Haiku sentiment pass (default true)
    """
    from models import Scan, ScanRedditThread
    from config import settings
    from datetime import datetime

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    reset = bool(job_payload.get("reset"))
    limit = int(job_payload.get("limit") or MAX_THREADS_PER_RUN)
    run_sentiment = bool(job_payload.get("sentiment", True))

    if reset:
        db.query(ScanRedditThread).filter(ScanRedditThread.scan_id == scan_id).delete()
        db.commit()

    threads = _cited_reddit_threads(db, scan_id, limit)
    if not threads:
        logger.info(f"audit_reddit_threads: no Reddit citations in scan {scan_id}")
        return {"threads": 0, "errors": 0, "sentiment_runs": 0}

    target_names, competitor_names = _scan_brands(db, scan_id)
    all_brand_names = sorted(list(target_names | competitor_names))[:30]
    api_key = (settings.anthropic_api_key or "").strip() if run_sentiment else ""

    audited = 0
    sentiment_runs = 0

    for t in threads:
        url = t["url"]
        contextes = t["contextes"]
        # Build the brand-detection corpus from the URL slug (often the
        # thread title verbatim) + the LLM citation snippets. The slug is
        # the highest-signal source in contexte-only mode because LLM
        # snippets are sometimes just `[Source: reddit.com]` with no body.
        slug = parse_title_slug(url)
        subreddit = t["subreddit"] or ""
        corpus = "\n".join([slug, subreddit, *contextes]).lower()

        target_hits = _detect_brands(corpus, target_names)
        competitor_hits = _detect_brands(corpus, competitor_names)
        target_mentioned = bool(target_hits)
        competitors_hit = sorted(list(competitor_hits))
        classification = _classify(target_hits, competitor_hits)

        sentiment = None
        sentiment_summary = None
        if api_key and (target_mentioned or competitors_hit):
            # Prepend the slug-derived title to the Haiku input so the
            # model can read what the thread is about, not just what the
            # LLM said when citing it.
            snippets_for_haiku = ([f"Thread title (from URL): {slug}"] if slug else []) + contextes
            res = classify_snippets(
                url=url,
                subreddit=t["subreddit"],
                snippets=snippets_for_haiku,
                brand_names=all_brand_names,
                api_key=api_key,
            )
            if res:
                sentiment = res.get("sentiment")
                sentiment_summary = res.get("summary")
                sentiment_runs += 1

        leverage = _leverage_score(t["citation_count"], classification, sentiment)
        # Synthetic "title" : the URL slug humanized, capitalized.
        title_from_slug = slug.title() if slug else None
        # body_excerpt = the concatenated LLM snippets so the UI can render
        # "what the LLMs said about this thread" inline.
        body_excerpt = "\n\n".join(contextes)[:4000]

        existing = (
            db.query(ScanRedditThread)
            .filter(ScanRedditThread.scan_id == scan_id, ScanRedditThread.url == url)
            .first()
        )
        if existing:
            existing.subreddit = t["subreddit"]
            existing.title = title_from_slug
            existing.fetched_at = datetime.utcnow()
            existing.fetch_status = None
            existing.fetch_error = None
            existing.citation_count = t["citation_count"]
            existing.target_mentioned = target_mentioned
            existing.competitors_mentioned = competitors_hit
            existing.classification = classification
            existing.sentiment = sentiment
            existing.sentiment_summary = sentiment_summary
            existing.body_excerpt = body_excerpt
            existing.top_comments = []  # contexte-only ; full thread fetch deferred
            existing.winning_questions = t["winning_questions"]
            existing.leverage_score = leverage
        else:
            db.add(ScanRedditThread(
                scan_id=scan_id,
                url=url,
                subreddit=t["subreddit"],
                title=title_from_slug,
                author=None,
                score=None,
                num_comments=None,
                fetch_status=None,
                fetch_error=None,
                citation_count=t["citation_count"],
                target_mentioned=target_mentioned,
                competitors_mentioned=competitors_hit,
                classification=classification,
                sentiment=sentiment,
                sentiment_summary=sentiment_summary,
                body_excerpt=body_excerpt,
                top_comments=[],
                winning_questions=t["winning_questions"],
                leverage_score=leverage,
            ))

        audited += 1
        if audited % 25 == 0:
            db.commit()
            logger.info(
                f"reddit audit progress {audited}/{len(threads)} "
                f"(sentiment_runs={sentiment_runs})"
            )

    db.commit()
    logger.info(
        f"reddit audit complete : threads={audited} sentiment_runs={sentiment_runs}"
    )
    return {
        "threads": audited,
        "sentiment_runs": sentiment_runs,
        "total": len(threads),
        "mode": "contexte_only",
    }
