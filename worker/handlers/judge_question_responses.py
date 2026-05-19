"""Handler: LLM-as-judge per-(question, response) signal scoring.

Sprint J of project_phase_judge_and_entities.md.

For each ScanLLMResult that lacks a judgment row, this handler asks Haiku to
read the response against the per-question grille (signal_positif,
signal_negatif, intention_cachee from scan_questions) and emit structured
booleans + literal evidence spans. The output lands in
scan_question_judgments (migration 037) where Sprint M composite scoring
will consume it.

## Why this exists

Pre-Sprint J the only "site visibility" signal was a boolean
`signal_positif_detecte = site_cite` (URL-match in citation_extractor.py:244).
That misses two things the PDF SEO LLM framework asks for :

1. **RAPP Positivity** — quality of the brand envelope (is the citation a
   real recommendation or a footnote? is the sentiment supportive or
   neutral-listy?). A URL match doesn't answer that.

2. **Per-question grille** — `signal_positif` and `signal_negatif` are
   LLM-generated FOR THIS QUESTION and capture nuances like "the LLM
   suggests an article rather than redirecting to a forum". Generic URL
   match can't see those.

## Design choices

**Per-question batching** (not per-response). For one ScanQuestion there
are typically 3 ScanLLMResult rows (one per provider — openai, gemini,
claude). The judge sees all 3 responses in one Haiku call, sharing the
question + grille once, and emits 3 judgment entries indexed by provider
position. This halves the input-token cost vs sending the grille per row.

**No target_brand in the prompt** (memo foot-gun #2). If the judge knew the
brand, it would mark every brand-citation as positive regardless of the
envelope quality, defeating the point. The grille is the only positive
criterion the judge knows about. Brand resolution + `est_cible` mapping
happens downstream in Sprint M scoring code, never inside the LLM.

**Evidence-required contract** (memo foot-gun #3). `intention_cachee` is
free-form French. The judge can hallucinate "yes" by default. The handler
post-processes : if a `*_hit` or `intent_addressed` is true but the matching
evidence string is empty, we force the bool to false. This converts
LLM-laziness into observable misses rather than fake positives.

**Response truncation** to bound cost — `_TRUNCATE_RESPONSE_CHARS` per
provider response. The signals look for citations and recommendations
which sit at the top of LLM responses 90% of the time, and the long
educational disclaimers at the bottom rarely change the judgment.

**Idempotent re-run** — handler queries only ScanLLMResult IDs that don't
yet appear in scan_question_judgments. Crash-safe : if Haiku returns N
judgments, we commit them, then loop to the next batch. A failed batch
leaves earlier results intact and retries on the next job poll.

**Budget cap** via `assert_within_budget` BEFORE the loop — projected cost
estimated at $0.005/question (rough Haiku math for 3 responses at
~1500 chars each + grille overhead). Calls `assert_within_budget` once
upfront; tight runaway protection requires log_llm_usage to update the
client's daily total which only happens at handler end. For now the
daily cap absorbs at-most-one-scan-overflow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
from sqlalchemy.orm import Session

from config import settings
from schemas import QuestionJudgmentEntry, validate_items
from utils import max_tokens_for

logger = logging.getLogger(__name__)


# Truncate each provider response sent to the judge. Most signal-relevant
# content (citations, recommendations, brand mentions) lives in the first
# ~2000 chars of a typical LLM reply. Going above that doubles token cost
# without measurably moving the judgment.
_TRUNCATE_RESPONSE_CHARS = 2200

# Rough per-question cost estimate. Used only for upfront budget guard ;
# real cost is logged via log_llm_usage at handler end.
_COST_PER_QUESTION_USD = 0.005


_JUDGE_PROMPT = """You are an impartial reading judge. You read 1-3 AI-generated answers to a user question, then check them against a per-question observation grid. Output JSON only.

# Question asked to the AI
{question}

# Hidden intent of this question (what the test really measures)
{intention_cachee}

# Positive signal — text patterns that count as the question being answered well from the site-owner's perspective
{signal_positif}

# Negative signal — text patterns that count as the question being answered poorly from the site-owner's perspective
{signal_negatif}

# AI answers to judge (one per line, prefixed by [idx])
{responses_block}

# For each answer, output one judgment entry with these fields:
- idx                         : integer matching the [idx] prefix
- positive_signal_hit         : true ONLY if the answer matches the positive-signal description above
- positive_signal_evidence    : literal text span from the answer (≤200 chars) that justifies the hit ; "" if no hit
- negative_signal_hit         : true ONLY if the answer matches the negative-signal description above
- negative_signal_evidence    : literal text span (≤200 chars) ; "" if no hit
- intent_addressed            : true ONLY if the answer addresses the hidden intent (not just the surface question)
- intent_evidence             : literal text span (≤200 chars) ; "" if not addressed
- citation_quality            : one of "lead" (cited as primary recommendation), "alternative" (cited among others), "footnote" (cited in passing / sources), "absent" (no clear citation of any specific source/brand)
- enveloppement_score         : 0-5 — overall quality of how the answer presents recommendations (0 = bare list, 5 = rich contextualized recommendation with reasons). null when no recommendation is present.

# Rules
- Output ONLY a JSON object. No prose, no markdown fences.
- Both *_hit booleans CAN be true at the same time (an answer can hit positive AND negative criteria simultaneously — that's a real signal, not a contradiction).
- Never invent evidence : if you set a hit to true, the evidence string MUST be a verbatim slice from the answer. If you can't quote, set the hit to false.
- Do NOT consider whether a specific brand is "the right answer" — only judge against the grids above. Brand identity is not your concern.

# Output JSON shape
{{
  "judgments": [
    {{
      "idx": 0,
      "positive_signal_hit": false,
      "positive_signal_evidence": "",
      "negative_signal_hit": true,
      "negative_signal_evidence": "...",
      "intent_addressed": true,
      "intent_evidence": "...",
      "citation_quality": "alternative",
      "enveloppement_score": 3
    }}
  ]
}}"""


def _format_responses_block(responses: list[str]) -> str:
    """One numbered response per block, truncated to _TRUNCATE_RESPONSE_CHARS."""
    parts = []
    for i, text in enumerate(responses):
        snippet = (text or "").strip()
        if len(snippet) > _TRUNCATE_RESPONSE_CHARS:
            snippet = snippet[:_TRUNCATE_RESPONSE_CHARS] + "…[truncated]"
        parts.append(f"[idx={i}]\n{snippet}")
    return "\n\n---\n\n".join(parts)


async def _call_haiku(prompt: str, api_key: str, model: str) -> dict:
    """One Haiku call, returns parsed JSON dict with `_usage` attached.

    Mirrors classify_question_intent._call_haiku — brace-counter JSON
    extraction tolerates leading prose despite temperature=0.
    """
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens_for(model, cap=2048),
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]

        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in judge response")
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
        parsed = json.loads(text[start:end + 1])
        parsed["_usage"] = data.get("usage", {})
        return parsed


def _enforce_evidence_contract(entry: QuestionJudgmentEntry) -> QuestionJudgmentEntry:
    """Foot-gun #3: a `*_hit` true is only believed if its evidence is non-empty.

    This converts the Haiku tendency to lazily say "yes" on intent_addressed
    into observable misses. Better a false-negative we can revisit than a
    false-positive that pollutes downstream dashboards.
    """
    if entry.positive_signal_hit and not entry.positive_signal_evidence:
        entry.positive_signal_hit = False
    if entry.negative_signal_hit and not entry.negative_signal_evidence:
        entry.negative_signal_hit = False
    if entry.intent_addressed and not entry.intent_evidence:
        entry.intent_addressed = False
    return entry


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Judge all unjudged ScanLLMResult rows for one scan.

    Re-runnable : filters rows that don't yet have a scan_question_judgments
    entry. A previous partial run keeps its commits ; the next invocation
    picks up the remaining responses.

    Skips rows where the question has no grille (NULL signal_positif AND
    NULL signal_negatif AND NULL intention_cachee) — those are legacy
    questions where the judge has nothing to evaluate against. They stay
    unjudged and are revisited if/when the user regenerates the persona.
    """
    from models import (
        Scan, ScanLLMResult, ScanQuestion, ScanQuestionJudgment,
    )
    from services.llm_budget import assert_within_budget

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    api_key = settings.anthropic_api_key
    if not api_key:
        logger.warning(
            f"ANTHROPIC_API_KEY missing; skipping judge for scan {scan_id}"
        )
        return {"judged": 0, "skipped": 0, "questions": 0}

    # Find ScanLLMResult rows lacking a judgment, grouped by question.
    # LEFT JOIN here returns ALL llm_results + a NULL on the join row when
    # no judgment exists yet.
    unjudged_rows = (
        db.query(ScanLLMResult, ScanQuestion)
        .outerjoin(
            ScanQuestionJudgment,
            ScanQuestionJudgment.scan_llm_result_id == ScanLLMResult.id,
        )
        .join(ScanQuestion, ScanQuestion.id == ScanLLMResult.question_id)
        .filter(
            ScanLLMResult.scan_id == scan_id,
            ScanLLMResult.response_text.isnot(None),
            ScanQuestionJudgment.id.is_(None),
        )
        .all()
    )

    if not unjudged_rows:
        logger.info(f"judge_question_responses: nothing to judge for scan {scan_id}")
        return {"judged": 0, "skipped": 0, "questions": 0}

    # Group by question_id. Questions missing all 3 grille fields are
    # filtered out — legacy rows where there's nothing to judge against.
    by_question: dict[str, list[tuple]] = {}
    skipped_no_grille = 0
    for llm_result, question in unjudged_rows:
        if not (question.signal_positif or question.signal_negatif or question.intention_cachee):
            skipped_no_grille += 1
            continue
        by_question.setdefault(str(question.id), []).append((llm_result, question))

    if not by_question:
        logger.info(
            f"judge_question_responses: scan {scan_id} has {skipped_no_grille} "
            "unjudged rows but none with a grille — nothing to do"
        )
        return {"judged": 0, "skipped": skipped_no_grille, "questions": 0}

    # Upfront budget guard — single call before the loop. Cost is tiny per
    # batch ($0.005/question rough estimate) and the daily cap will trip on
    # the next handler poll if we exceed it mid-run.
    projected_cost = max(0.01, _COST_PER_QUESTION_USD * len(by_question))
    assert_within_budget(scan.client_id, db, projected_cost_usd=projected_cost)

    model = settings.task_models["judge_question_responses"]
    judged_count = 0
    failed_questions = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_ts = time.time()

    for question_id, pairs in by_question.items():
        # pairs = [(llm_result, question), ...] all sharing the same question
        question = pairs[0][1]
        responses_text = [llm.response_text for llm, _ in pairs]
        prompt = _JUDGE_PROMPT.format(
            question=question.question or "",
            intention_cachee=question.intention_cachee or "(non spécifiée)",
            signal_positif=question.signal_positif or "(non spécifié)",
            signal_negatif=question.signal_negatif or "(non spécifié)",
            responses_block=_format_responses_block(responses_text),
        )

        try:
            result = asyncio.run(_call_haiku(prompt, api_key, model))
        except Exception:
            logger.exception(
                f"judge_question_responses: question {question_id} batch failed "
                f"(scan {scan_id}) — rows stay unjudged, retried next run"
            )
            failed_questions += 1
            continue

        usage = result.pop("_usage", {})
        total_input_tokens += usage.get("input_tokens", 0) or 0
        total_output_tokens += usage.get("output_tokens", 0) or 0

        raw_judgments = result.get("judgments") or []
        validated = validate_items(
            raw_judgments,
            QuestionJudgmentEntry,
            f"judge_question_responses.judgments[q={question_id}]",
        )

        # Index validated entries by idx for safe pairing with the responses
        # list we sent. Missing idx (Haiku omitted one) = that row stays
        # unjudged and the next handler poll re-tries.
        by_idx = {e.idx: e for e in validated}

        for idx, (llm_result, _) in enumerate(pairs):
            entry = by_idx.get(idx)
            if entry is None:
                continue
            entry = _enforce_evidence_contract(entry)
            db.add(ScanQuestionJudgment(
                scan_llm_result_id=llm_result.id,
                scan_id=scan_id,
                question_id=llm_result.question_id,
                positive_signal_hit=entry.positive_signal_hit,
                positive_signal_evidence=entry.positive_signal_evidence or None,
                negative_signal_hit=entry.negative_signal_hit,
                negative_signal_evidence=entry.negative_signal_evidence or None,
                intent_addressed=entry.intent_addressed,
                intent_evidence=entry.intent_evidence or None,
                citation_quality=entry.citation_quality,
                enveloppement_score=entry.enveloppement_score,
                judge_model=model,
                input_tokens=usage.get("input_tokens", 0) or 0,
                output_tokens=usage.get("output_tokens", 0) or 0,
            ))
            judged_count += 1

        db.commit()

    duration_ms = int((time.time() - start_ts) * 1000)

    try:
        from adapters.llm_logger import log_llm_usage
        log_llm_usage(
            db,
            provider="anthropic",
            model=model,
            operation="judge_question_responses",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            duration_ms=duration_ms,
            scan_id=scan_id,
            client_id=str(scan.client_id),
        )
    except Exception:
        logger.exception("log_llm_usage failed for judge_question_responses")

    logger.info(
        f"judge_question_responses: scan {scan_id} — judged {judged_count} rows "
        f"over {len(by_question) - failed_questions} questions "
        f"(failed_questions={failed_questions}, skipped_no_grille={skipped_no_grille}, "
        f"{duration_ms}ms)"
    )

    return {
        "judged": judged_count,
        "skipped": skipped_no_grille,
        "questions": len(by_question),
        "failed_questions": failed_questions,
        "duration_ms": duration_ms,
    }
