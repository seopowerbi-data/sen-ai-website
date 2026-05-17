"""Fan-out query extractor & primary selector for netlinking article gen.

Replaces the YTG 150-char truncate hack in
`worker/handlers/generate_article.py` (commit `1605d05`) with proper
ground-truth fan-out queries from LLM responses.

## Architecture (B1 + B2 hybrid)

```
question (long GEO conversational ~150c)
    ↓
  Step 1 : Cache lookup on scan_questions.fan_out_queries
    │
    │ HIT → return cached primary + all fan-outs
    │
    │ MISS ↓
  Step 2 : aggregate_fanouts_from_scan (B1, per-provider direct capture)
    │ Pull scan_llm_results.web_search_queries per provider
    │
    │ If ≥1 fan-out captured → use them
    │
    │ If empty (legacy data / Claude without web_search tool / etc.) ↓
  Step 3 : extract_fanouts_post_hoc (B2 fallback)
    │ Haiku call with question + response_text + citation contexts
    │ Output : 3-5 reconstructed fan-outs
    │
    ↓
  Step 4 : select_primary_fanout (ranking algo + Haiku tiebreak)
    │ 5 criteria : consensus × 10, topic alignment, length sweet spot,
    │ position, best_competitor alignment
    │ Top-1 score wins if clear (>30% gap), else Haiku judges
    │
    ↓
  Step 5 : Persist on scan_questions.fan_out_queries (cache)
    Primary at index [0], additional fan-outs at [1+]
    ↓
  Return : list[str] primary-first
```

## Why long conversational questions are CORRECT for GEO (don't shorten)

User asks ChatGPT/Gemini/Perplexity in conversational long-form ("J'ai
entendu dire que... ?"). These conversational questions are :
  - Used by scan_llm_tests to test what LLMs say (realistic prompt)
  - Used by FAQ Schema.org Q/R block (natural FAQ language)
  - Used by validation page UI (human-readable)

But YTG (SEO SERP analyzer) wants Google-style 30-80c queries. The
SOLUTION isn't to shorten the question — it's to EXTRACT fan-outs (the
sub-search-intents the LLMs decompose into internally) and feed THOSE
to YTG. The conversational question stays preserved for its 3 other uses.

## Why this is better than Haiku synthesize from question alone

Method A (Haiku synthesize from question only) : Haiku GUESSES what
queries a user might issue. ~80% quality.

Method B1+B2 (this module) : extract from REAL LLM decomposition. Each
LLM ran its own fan-out internally (Google SGE patent style) — we
recover those queries from grounding_metadata / web_search_call /
tool_use blocks. Cross-provider consensus = strongest signal. ~99%
quality (B1) or ~92% (B2 reconstructed via Haiku from response_text).

See `project_phase_c1_article_handler.md` section C.1.5 for the full
architecture decision tree + ranking algo specification.

## Public surface

  extract_or_get_cached(question_id, scan_id, db) → list[str] primary-first
  aggregate_fanouts_from_scan(question_id, db) → dict per-provider + consensus
  extract_fanouts_post_hoc(question, scan_results, db) → list[str]
  select_primary_fanout(per_provider, question, topic, best_competitor)
      → dict {primary, rationale, all_ranked}
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict, Counter
from typing import TypedDict

import httpx
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)


# ─── Tunables ──────────────────────────────────────────────────────────

# Max fan-outs to return (primary + N additional). Beyond 5, marginal
# value (the content gen prompt gets too cluttered, FAQ Q seeds dilute).
_MAX_FANOUTS = 5

# Length sweet spot for YTG-compatible queries. Matches PF dim_fanout.csv
# distribution (mean 55c, 99% under 100c).
_LENGTH_SWEET_SPOT = (40, 80)
_LENGTH_HARD_MIN = 15  # too short = not a real query
_LENGTH_HARD_MAX = 150  # YTG hard limit, also fan-outs longer = synthesis bug

# Ranking score weights (see project_phase_c1_article_handler.md C.1.5).
_W_CONSENSUS_PER_PROVIDER = 10
_W_TOPIC_KEYWORD = 4
_W_TOPIC_KEYWORD_CAP = 8
_W_LENGTH_SWEET = 5
_W_LENGTH_NEAR_SWEET_LOW = 3   # 30-40c
_W_LENGTH_NEAR_SWEET_HIGH = 2  # 80-100c
_W_LENGTH_PENALTY = -5         # <25c or >120c
_W_POSITION_FIRST = 3
_W_POSITION_SECOND = 1
_W_COMPETITOR_ALIGNED = 7

# Tiebreak threshold : if top-1 score >= top-2 × this ratio, no Haiku call.
_CLEAR_WINNER_RATIO = 1.3

# Haiku model for post-hoc extraction + tiebreak
_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_HAIKU_TIMEOUT_S = 30
_HAIKU_MAX_TOKENS = 2000


# ─── Public API ────────────────────────────────────────────────────────


def extract_or_get_cached(
    question_id: str,
    scan_id: str,
    db: Session,
) -> list[str]:
    """Main entry point. Returns fan-outs primary-first, cached or fresh.

    Cache lookup → B1 aggregation → B2 fallback → ranking → persist.

    Returns empty list when :
      - question_id not found
      - no scan_llm_results for this question
      - Haiku extraction also failed
    Caller should fall back to truncated question_text in that case.
    """
    if not question_id:
        return []

    from models import ScanQuestion

    q = db.query(ScanQuestion).filter(ScanQuestion.id == question_id).first()
    if not q:
        logger.warning(f"fan_out_extractor: question {question_id} not found")
        return []

    # ── Step 1 : cache lookup ─────────────────────────────────────────
    cached = list(q.fan_out_queries or [])
    if cached:
        logger.info(
            f"fan_out_extractor: cache HIT for question {question_id} "
            f"({len(cached)} fan-outs, primary='{cached[0][:60]}')"
        )
        return cached

    # ── Step 2 : aggregate from per-provider captured queries (B1) ──
    agg = aggregate_fanouts_from_scan(question_id, db)
    per_provider = agg["per_provider"]
    total_captured = sum(len(v) for v in per_provider.values())

    selection: dict | None = None

    if total_captured > 0:
        logger.info(
            f"fan_out_extractor: B1 captured {total_captured} queries "
            f"across {len(per_provider)} providers for question {question_id}"
        )
        # Get topic_name + best_competitor from item context (lookup via
        # scan_opportunities if available) for ranking
        topic_name, best_competitor = _resolve_question_context(
            question_id, scan_id, db,
        )
        selection = select_primary_fanout(
            per_provider, q.question, topic_name, best_competitor,
        )

    # ── Step 3 : B2 fallback (post-hoc Haiku from response_text) ────
    if not selection or not selection.get("all_ranked"):
        logger.info(
            f"fan_out_extractor: B1 empty → B2 fallback (Haiku from response_text) "
            f"for question {question_id}"
        )
        fanouts = extract_fanouts_post_hoc(q.question, question_id, db)
        if not fanouts:
            logger.warning(
                f"fan_out_extractor: BOTH B1 and B2 returned empty for "
                f"question {question_id} — caller must fall back to truncated question"
            )
            return []
        # Treat as single-provider 'haiku_synthesized' for ranking
        topic_name, best_competitor = _resolve_question_context(
            question_id, scan_id, db,
        )
        selection = select_primary_fanout(
            {"haiku_synthesized": fanouts}, q.question, topic_name, best_competitor,
        )

    # ── Step 4 : extract ordered list (primary at [0]) ──────────────
    if not selection or not selection.get("all_ranked"):
        return []
    ordered = [r["fanout"] for r in selection["all_ranked"][:_MAX_FANOUTS]]

    # ── Step 5 : persist on scan_questions.fan_out_queries ──────────
    q.fan_out_queries = ordered
    flag_modified(q, "fan_out_queries")
    try:
        db.commit()
        logger.info(
            f"fan_out_extractor: cached {len(ordered)} fan-outs for question "
            f"{question_id} (primary='{ordered[0][:60]}', "
            f"rationale='{selection.get('rationale', 'n/a')[:80]}')"
        )
    except Exception:
        logger.exception(
            f"fan_out_extractor: failed to persist fan_out_queries for {question_id}"
        )
        db.rollback()

    return ordered


def aggregate_fanouts_from_scan(question_id: str, db: Session) -> dict:
    """Pull web_search_queries per provider for a question. Returns :

      {
        "per_provider": {gemini: [...], openai: [...], claude: [...]},
        "consensus": list[str],     # queries cited by ≥2 providers (deduped)
        "all_unique": list[str],    # union of all queries deduped
      }
    """
    from models import ScanLLMResult

    rows = (
        db.query(ScanLLMResult)
        .filter(ScanLLMResult.question_id == question_id)
        .all()
    )

    per_provider: dict[str, list[str]] = {}
    for r in rows:
        queries = list(r.web_search_queries or [])
        provider = (r.provider or "unknown").lower()
        # Aggregate across multiple rows per provider (refresh runs etc.)
        existing = per_provider.setdefault(provider, [])
        for q in queries:
            q_clean = (q or "").strip()
            if q_clean and q_clean not in existing:
                existing.append(q_clean)

    # Consensus : normalized-equal queries appearing in ≥2 providers
    norm_to_providers: dict[str, set[str]] = defaultdict(set)
    norm_to_canonical: dict[str, str] = {}
    for provider, queries in per_provider.items():
        for q in queries:
            n = _normalize_for_dedup(q)
            if not n:
                continue
            norm_to_providers[n].add(provider)
            # Keep first-seen canonical form
            norm_to_canonical.setdefault(n, q)

    consensus = [
        norm_to_canonical[n]
        for n, providers in norm_to_providers.items()
        if len(providers) >= 2
    ]
    all_unique = list(norm_to_canonical.values())

    return {
        "per_provider": per_provider,
        "consensus": consensus,
        "all_unique": all_unique,
    }


def extract_fanouts_post_hoc(
    question: str, question_id: str, db: Session,
) -> list[str]:
    """B2 fallback : reconstruct fan-outs from response_text + citation
    contexts via Haiku, when explicit web_search_queries are unavailable
    (legacy scans, providers without grounding metadata, etc.).

    Uses CROSS-PROVIDER response data : aggregating responses from
    Gemini + OpenAI + Claude + Perplexity gives Haiku richer context to
    infer the actual sub-search-intents the LLMs decomposed into.

    Returns empty list on failure (no responses, Haiku error, parse error).
    """
    from models import ScanLLMResult

    rows = (
        db.query(ScanLLMResult)
        .filter(ScanLLMResult.question_id == question_id)
        .all()
    )
    if not rows:
        return []

    # Build extraction context (truncated to keep Haiku input small)
    context_blocks: list[str] = []
    for r in rows:
        if not r.response_text:
            continue
        provider = (r.provider or "unknown").upper()
        text = (r.response_text or "")[:1500]
        context_blocks.append(f"[{provider} response, truncated]\n{text}")
        # Add citation contexts (snippets around cited URLs)
        for c in (r.citations or [])[:5]:
            ctx = (c.get("contexte") or "")[:120].strip()
            if ctx:
                context_blocks.append(
                    f"  → [{provider}] cited {c.get('domaine', '?')}: {ctx}"
                )

    if not context_blocks:
        return []

    context_str = "\n\n".join(context_blocks)[:8000]  # cap total context

    prompt = f"""Tu es expert en GEO/SEO et décomposition d'intentions de recherche.

Une question utilisateur a été posée à plusieurs LLMs (ChatGPT, Gemini, Perplexity, etc.).
Voici leurs réponses agrégées (truncated). Chaque LLM a internalement décomposé
la question en sub-search-intents (fan-out queries Google) pour y répondre.

Extrais 3-5 fan-out queries QUI ÉMERGENT de ces réponses — les sub-intents
sur lesquels les LLMs ont CONVERGÉ (mêmes thèmes, mêmes sources citées,
mêmes angles d'approche).

QUESTION : {question}

RÉPONSES AGRÉGÉES :
{context_str}

RÈGLES STRICTES :
- Chaque fan-out : 30-80 caractères, format Google search (keyword-rich)
- Pas de question conversationnelle, juste les keywords intent
- Ordre par importance (le 1er = le plus représentatif du sujet central)
- Si plusieurs LLMs convergent sur un sub-intent → c'est un fan-out solide
- Langue : français (sauf si la question est en anglais)

Format JSON STRICT (no markdown) :
{{"fanouts": ["query1", "query2", "query3"]}}"""

    result = _call_haiku_json(prompt)
    if not result or "fanouts" not in result:
        return []

    raw = result.get("fanouts") or []
    if not isinstance(raw, list):
        return []

    # Validate + dedupe
    seen: set[str] = set()
    out: list[str] = []
    for fo in raw:
        if not isinstance(fo, str):
            continue
        fo_clean = fo.strip()
        if not (_LENGTH_HARD_MIN <= len(fo_clean) <= _LENGTH_HARD_MAX):
            continue
        n = _normalize_for_dedup(fo_clean)
        if n in seen:
            continue
        seen.add(n)
        out.append(fo_clean)

    return out[:_MAX_FANOUTS]


def select_primary_fanout(
    per_provider: dict[str, list[str]],
    question: str,
    topic_name: str | None,
    best_competitor: str | None,
) -> dict:
    """Rank fan-outs across providers + select primary (sent to YTG).

    Scoring : consensus × 10 + topic_match (capped 8) + length sweet
    spot (5/3/2/-5) + position (3/1) + competitor alignment (7).

    Tiebreak : if top-1 score < top-2 × _CLEAR_WINNER_RATIO, Haiku judges
    the close call. Else deterministic.

    Returns {primary, rationale, all_ranked: [{fanout, score, providers}]}.
    """
    # 1. Unique fan-outs across providers
    fanout_to_providers: dict[str, set[str]] = defaultdict(set)
    fanout_to_positions: dict[str, list[int]] = defaultdict(list)
    fanout_canonical: dict[str, str] = {}

    for provider, queries in per_provider.items():
        for pos, q in enumerate(queries):
            q_clean = (q or "").strip()
            if not q_clean or not (_LENGTH_HARD_MIN <= len(q_clean) <= _LENGTH_HARD_MAX):
                continue
            n = _normalize_for_dedup(q_clean)
            if not n:
                continue
            fanout_to_providers[n].add(provider)
            fanout_to_positions[n].append(pos)
            fanout_canonical.setdefault(n, q_clean)

    if not fanout_canonical:
        return {"primary": None, "rationale": "no_valid_fanouts", "all_ranked": []}

    # 2. Score each
    topic_keywords = _extract_keywords(topic_name) if topic_name else []
    competitor_norm = (best_competitor or "").lower().strip()

    scored: list[dict] = []
    for norm_key, canonical in fanout_canonical.items():
        providers = fanout_to_providers[norm_key]
        positions = fanout_to_positions[norm_key]
        score = _score_fanout(
            canonical, providers, positions, topic_keywords, competitor_norm,
        )
        scored.append({
            "fanout": canonical,
            "score": score,
            "providers": sorted(providers),
        })

    # 3. Sort by score DESC, tiebreak by length closer to 55c
    scored.sort(key=lambda x: (-x["score"], abs(len(x["fanout"]) - 55)))

    # 4. Clear-winner check (top-1 vs top-2 ratio)
    top1 = scored[0]
    top2 = scored[1] if len(scored) > 1 else None
    if top2 is None or top1["score"] >= max(top2["score"], 1) * _CLEAR_WINNER_RATIO:
        return {
            "primary": top1["fanout"],
            "rationale": (
                f"deterministic clear winner : score {top1['score']:.1f} vs "
                f"{(top2['score'] if top2 else 0):.1f}"
            ),
            "all_ranked": scored,
        }

    # 5. Close call → Haiku tiebreak
    top_3 = scored[:3]
    haiku_pick = _haiku_judge_primary(top_3, question, topic_name, best_competitor)
    if haiku_pick and haiku_pick.get("primary"):
        # Move Haiku's pick to top of all_ranked
        primary = haiku_pick["primary"]
        reordered = [r for r in scored if r["fanout"] == primary] + [
            r for r in scored if r["fanout"] != primary
        ]
        return {
            "primary": primary,
            "rationale": f"Haiku tiebreak : {haiku_pick.get('why', 'no rationale')}",
            "all_ranked": reordered,
        }

    # Haiku failed → fall back to deterministic top-1
    return {
        "primary": top1["fanout"],
        "rationale": (
            f"top-1 by score (Haiku tiebreak failed) : "
            f"{top1['score']:.1f} vs {top2['score']:.1f}"
        ),
        "all_ranked": scored,
    }


# ─── Internal helpers ──────────────────────────────────────────────────


def _score_fanout(
    fanout: str,
    providers: set[str],
    positions: list[int],
    topic_keywords: list[str],
    competitor_norm: str,
) -> float:
    score = 0.0

    # 1. Cross-provider consensus
    score += len(providers) * _W_CONSENSUS_PER_PROVIDER

    # 2. Topic alignment (cap at _W_TOPIC_KEYWORD_CAP)
    fo_lower = fanout.lower()
    matched = sum(1 for kw in topic_keywords if kw and kw in fo_lower)
    score += min(matched * _W_TOPIC_KEYWORD, _W_TOPIC_KEYWORD_CAP)

    # 3. Length sweet spot
    L = len(fanout)
    if _LENGTH_SWEET_SPOT[0] <= L <= _LENGTH_SWEET_SPOT[1]:
        score += _W_LENGTH_SWEET
    elif 30 <= L < _LENGTH_SWEET_SPOT[0]:
        score += _W_LENGTH_NEAR_SWEET_LOW
    elif _LENGTH_SWEET_SPOT[1] < L <= 100:
        score += _W_LENGTH_NEAR_SWEET_HIGH
    elif L < 25 or L > 120:
        score += _W_LENGTH_PENALTY

    # 4. Position (earliest across providers)
    if positions:
        best_pos = min(positions)
        if best_pos == 0:
            score += _W_POSITION_FIRST
        elif best_pos == 1:
            score += _W_POSITION_SECOND

    # 5. Best_competitor keyword in fanout (loose match : competitor brand
    # name appears as substring in the fan-out, indicating shared search
    # intent with the competitor's citation pattern).
    if competitor_norm and competitor_norm in fo_lower:
        score += _W_COMPETITOR_ALIGNED

    return score


def _extract_keywords(topic: str | None) -> list[str]:
    """Extract lowercase keywords (≥4 chars) from a topic_name string,
    stripping punctuation. e.g. 'Anti-âge & actifs cosmétiques (rétinol)'
    → ['anti-âge', 'actifs', 'cosmétiques', 'rétinol']."""
    if not topic:
        return []
    # Split on non-word + drop short tokens
    raw = re.split(r"[\s()&,;:/]+", topic.lower())
    return [t for t in raw if len(t) >= 4]


def _normalize_for_dedup(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for semantic dedup."""
    if not s:
        return ""
    n = s.lower().strip()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _resolve_question_context(
    question_id: str, scan_id: str, db: Session,
) -> tuple[str | None, str | None]:
    """Lookup topic_name + best_competitor for a question via scan_opportunities.

    Returns (topic_name, best_competitor) or (None, None) if no opportunity exists.
    """
    from models import ScanOpportunity

    opp = (
        db.query(ScanOpportunity)
        .filter(ScanOpportunity.question_id == question_id)
        .first()
    )
    if not opp:
        return None, None
    return (opp.topic_name or None, opp.best_competitor_name or None)


# ─── Haiku JSON call (shared by extract_post_hoc + tiebreak) ──────────


def _call_haiku_json(prompt: str) -> dict | None:
    """Call Anthropic Haiku, parse JSON response. Return None on any failure."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("fan_out_extractor: ANTHROPIC_API_KEY missing — Haiku call skipped")
        return None

    try:
        with httpx.Client(timeout=_HAIKU_TIMEOUT_S) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": _HAIKU_MODEL,
                    "max_tokens": _HAIKU_MAX_TOKENS,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]
    except Exception:
        logger.exception("fan_out_extractor: Haiku call failed")
        return None

    # Robust JSON parse (strip markdown fences, brace-counter fallback)
    text = (text or "").strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace-counter fallback (Haiku sometimes wraps in conversational text)
    try:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        return json.loads(text[start:end + 1])
    except Exception:
        logger.warning(
            "fan_out_extractor: Haiku JSON parse failed — text head: %r",
            text[:200],
        )
        return None


def _haiku_judge_primary(
    top_3: list[dict],
    question: str,
    topic_name: str | None,
    best_competitor: str | None,
) -> dict | None:
    """Haiku tiebreak when top-1 vs top-2 scores are close (<30% gap)."""
    if len(top_3) < 2:
        return None

    options = "\n".join(
        f"{i+1}. \"{r['fanout']}\" (cited by : {', '.join(r['providers'])})"
        for i, r in enumerate(top_3)
    )

    prompt = f"""Tu es expert SEO/GEO. Choisis UNE search query parmi ces options pour
optimiser un article SEO sur un sujet stratégique.

Sujet : {topic_name or 'non spécifié'}
Question originale utilisateur : {question}
Concurrent à dépasser : {best_competitor or 'non spécifié'}

Options (scores quasi-équivalents, départage par jugement éditorial) :
{options}

Critères de choix :
- Quelle query a le SERP Google le plus PROPRE et REPRÉSENTATIF du sujet ?
- Quelle query un user qui veut RÉPONDRE à la question initiale taperait ?
- Quelle query est la plus alignée avec la concurrence à dépasser ?

Réponds JSON strict (no markdown) :
{{"primary": "<query_exacte_choisie>", "why": "<une phrase de rationale>"}}"""

    result = _call_haiku_json(prompt)
    if not result or "primary" not in result:
        return None

    # Validate : the picked query must be one of the options
    picked = (result.get("primary") or "").strip()
    valid_fanouts = {r["fanout"] for r in top_3}
    if picked not in valid_fanouts:
        # Haiku might have slightly reformulated — try fuzzy match
        picked_norm = _normalize_for_dedup(picked)
        for f in valid_fanouts:
            if _normalize_for_dedup(f) == picked_norm:
                return {"primary": f, "why": result.get("why", "")}
        logger.warning(
            f"fan_out_extractor: Haiku returned non-option '{picked[:60]}' "
            f"— falling back to deterministic top-1"
        )
        return None

    return {"primary": picked, "why": result.get("why", "")}
