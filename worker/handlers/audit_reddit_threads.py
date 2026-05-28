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

    Sprint 8.4 (2026-05-28) : in addition to the citation contexte snippet
    (~200 chars) we also collect the FULL response_text of each citing
    LLM response. The contexte alone is often too thin (e.g. Gemini just
    writes "[Source: reddit.com]") to detect competitor mentions, but the
    surrounding response often co-recommends multiple brands. Without
    this broader corpus the classifier under-detects head_to_head /
    you_lost cases.

    Returns one entry per canonical URL with :
        {
          url, subreddit,
          citation_count,                # how many LLM responses cited it
          contextes : [snippet, ...],    # for the Haiku sentiment pass
          response_texts : [text, ...],  # full LLM responses citing this URL
          winning_questions : [...],
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
               citation->>'contexte' AS contexte,
               slr.response_text AS response_text
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

    # Cap each response_text contribution so a single very long LLM answer
    # can't blow up the in-memory corpus when a URL is cited many times.
    RESPONSE_TEXT_CAP = 4000

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
                "response_texts": [],
                "_seen_slr": set(),
                "winning_questions": [],
            }
            bucket[canonical] = b
        b["citation_count"] += 1
        contexte = (r.contexte or "").strip()
        if contexte and contexte not in b["contextes"]:
            b["contextes"].append(contexte)
        # Each (slr_id, URL) maps to ONE response_text - dedupe so we don't
        # add the same response twice when an LLM cites the same URL
        # multiple times within its answer.
        if r.slr_id and r.slr_id not in b["_seen_slr"]:
            b["_seen_slr"].add(r.slr_id)
            rt = (r.response_text or "").strip()
            if rt:
                b["response_texts"].append(rt[:RESPONSE_TEXT_CAP])
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

    # Drop the helper set before returning so the dict is JSON-serializable
    # for downstream consumers if needed.
    for b in bucket.values():
        b.pop("_seen_slr", None)

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


def _classify(
    target_hits: set[str],
    competitor_hits: set[str],
    target_sentiment: str | None,
    competitor_sentiment: str | None,
) -> str:
    """Classification matrix v2 - distinguishes head-to-head outcomes
    using per-brand sentiment (migration 052, Sprint 8 polish round 2).

    Values :
      competitor_wins  : competitor named, target absent (clear opportunity)
      you_lost         : BOTH named, competitor positive AND target negative
                         or neutral (you're losing the comparison)
      shared_crisis    : BOTH named, both sentiment negative (industry-wide issue)
      shared_win       : BOTH named, both sentiment positive (co-consideration)
      you_win_strong   : BOTH named, target positive AND competitor negative
      head_to_head     : BOTH named, mixed / unclear signals (investigate)
      you_win          : target named, no competitor (positive footprint)
      neutral          : neither named
    """
    has_target = bool(target_hits)
    has_competitor = bool(competitor_hits)

    if has_competitor and not has_target:
        return "competitor_wins"
    if not has_target and not has_competitor:
        return "neutral"
    if has_target and not has_competitor:
        return "you_win"

    # Both named — read per-brand sentiment.
    t = (target_sentiment or "").lower()
    c = (competitor_sentiment or "").lower()

    if t == "negative" and c == "negative":
        return "shared_crisis"
    if t == "positive" and c == "positive":
        return "shared_win"
    if t == "positive" and c == "negative":
        return "you_win_strong"
    if c == "positive" and t in ("negative", "neutral", "unclear", ""):
        return "you_lost"
    if t == "negative" and c in ("neutral", "unclear", ""):
        return "you_lost"
    return "head_to_head"


def _leverage_score(citation_count: int, classification: str, sentiment: str | None) -> int:
    """Composite 0-100 priority score (per-brand-aware).

    Classification base points (50 max) reflect the bucket's urgency :
      you_lost / shared_crisis  : 50  (you are losing or industry-wide issue)
      competitor_wins           : 45
      head_to_head              : 35
      shared_win                : 20
      you_win_strong            : 18  (already winning, low urgency)
      you_win                   : 12
      neutral                   : 0

    Sentiment lever (max 30) uses the OVERALL sentiment of the discussion.
    Engagement (max 20) = log10(citation_count + 1) * 20.
    """
    import math

    cc = max(0, int(citation_count or 0))
    engagement = min(20, int(round(math.log10(cc + 1) * 20)))

    cls_pts = {
        "you_lost":         50,
        "shared_crisis":    50,
        "competitor_wins":  45,
        "head_to_head":     35,
        "shared_win":       20,
        "you_win_strong":   18,
        "you_win":          12,
        "neutral":          0,
    }.get(classification or "neutral", 0)

    sent = sentiment or ""
    sent_pts = {
        "negative": 25, "mixed": 18, "neutral": 12,
        "unclear":  10, "positive": 5,
    }.get(sent, 8)

    return max(0, min(100, engagement + cls_pts + sent_pts))


def _recommended_action(classification: str, sentiment: str | None) -> dict:
    """Action label + tone derived from the classification × sentiment
    matrix. Drives the UI "Action" column."""
    cls = classification or "neutral"
    sent = sentiment or "unclear"

    if cls == "you_lost":
        return {"label": "Defend your position", "tone": "urgent"}
    if cls == "shared_crisis":
        return {"label": "Crisis response", "tone": "urgent"}
    if cls == "competitor_wins":
        if sent == "negative":
            return {"label": "Engage now", "tone": "urgent"}
        if sent == "mixed":
            return {"label": "Engage thoughtfully", "tone": "high"}
        if sent in ("neutral", "unclear"):
            return {"label": "Add your perspective", "tone": "medium"}
        return {"label": "Skip - they win", "tone": "low"}
    if cls == "head_to_head":
        return {"label": "Investigate", "tone": "high"}
    if cls == "shared_win":
        return {"label": "Co-consideration", "tone": "positive"}
    if cls == "you_win_strong":
        return {"label": "Amplify", "tone": "positive"}
    if cls == "you_win":
        if sent == "negative":
            return {"label": "Monitor crisis", "tone": "urgent"}
        return {"label": "Keep monitoring", "tone": "positive"}
    return {"label": "Context only", "tone": "low"}


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
    # Focus brand name for the Haiku prompt. Read from the FK
    # relationship (scan.focus_brand.name) ; fall back to the first
    # target_name when the focus brand isn't set (legacy scans).
    focus_brand_name = ""
    if getattr(scan, "focus_brand", None) and getattr(scan.focus_brand, "name", None):
        focus_brand_name = scan.focus_brand.name.strip()
    if not focus_brand_name:
        focus_brand_name = next(iter(target_names), "") or ""
    competitor_brand_names = sorted(list(competitor_names))[:20]
    api_key = (settings.anthropic_api_key or "").strip() if run_sentiment else ""

    audited = 0
    sentiment_runs = 0

    for t in threads:
        url = t["url"]
        contextes = t["contextes"]
        response_texts = t.get("response_texts") or []
        # Sprint 8.4 : the brand-detection corpus now includes the FULL
        # LLM response_text of each response that cited this Reddit URL,
        # not just the 200-char contexte. Without this, head_to_head and
        # you_lost were under-detected - the LLM often co-recommends
        # multiple brands in its answer while citing Reddit for just one
        # of them (audit Avène 2026-05-28 : 0 you_lost vs 35% of
        # Reddit-citing responses actually mentioning competitors).
        slug = parse_title_slug(url)
        subreddit = t["subreddit"] or ""
        corpus = "\n".join([slug, subreddit, *contextes, *response_texts]).lower()

        target_hits = _detect_brands(corpus, target_names)
        competitor_hits = _detect_brands(corpus, competitor_names)
        target_mentioned = bool(target_hits)
        competitors_hit = sorted(list(competitor_hits))

        sentiment = None
        sentiment_summary = None
        target_sentiment = None
        competitor_sentiment = None
        if api_key and (target_mentioned or competitors_hit):
            # Snippets fed to Haiku :
            #   1. URL-derived thread title (highest brand-mention signal)
            #   2. The 200-char contextes from the LLM citations
            #   3. Sprint 8.4 : extracts of response_texts when brands
            #      were detected there (gives Haiku enough surrounding
            #      content to read per-brand sentiment, otherwise the
            #      contextes alone are too thin).
            snippets_for_haiku: list[str] = []
            if slug:
                snippets_for_haiku.append(f"Thread title (from URL): {slug}")
            snippets_for_haiku.extend(contextes)
            # Add up to 2 response_text excerpts, each capped at 1500 chars,
            # so Haiku has the broader recommendation context but the
            # prompt stays bounded (~3 KB total).
            for rt in response_texts[:2]:
                if rt:
                    snippets_for_haiku.append(
                        f"Surrounding LLM response excerpt: {rt[:1500]}"
                    )
            res = classify_snippets(
                url=url,
                subreddit=t["subreddit"],
                snippets=snippets_for_haiku,
                target_brand=focus_brand_name,
                competitor_brands=competitor_brand_names,
                api_key=api_key,
            )
            if res:
                sentiment = res.get("sentiment")
                sentiment_summary = res.get("summary")
                target_sentiment = res.get("target_sentiment")
                competitor_sentiment = res.get("competitor_sentiment")
                sentiment_runs += 1

        # Classification is computed AFTER sentiment so head-to-head rows
        # can use the per-brand sentiment to decide you_lost / shared_*.
        classification = _classify(
            target_hits, competitor_hits, target_sentiment, competitor_sentiment
        )
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
            existing.target_sentiment = target_sentiment
            existing.competitor_sentiment = competitor_sentiment
            existing.body_excerpt = body_excerpt
            existing.top_comments = []
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
                target_sentiment=target_sentiment,
                competitor_sentiment=competitor_sentiment,
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
