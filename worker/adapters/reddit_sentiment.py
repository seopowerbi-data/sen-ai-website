"""Per-thread sentiment classification via Claude Haiku.

v1 contexte-only mode : input is the concatenated LLM citation snippets
(scan_llm_results.citations[].contexte) for a single Reddit URL, NOT
the full thread (Reddit blocks our cloud IP - see reddit_client.py
docstring). At ~200 chars per snippet × ~5 snippets per URL, input is
~1KB - even cheaper than the original full-thread version, ~$0.0003 per
thread, ~$0.03 per 100-thread scan worst case.

Why we run sentiment despite the thin input :
  - At 100 threads per scan, manual triage is impractical.
  - "Competitor mentioned" + "negative sentiment" = highest leverage
    opportunity (user can step in with a better answer).
  - "Competitor mentioned" + "positive sentiment" = lower leverage (the
    crowd already loves them ; harder to flip).
  - "Target brand mentioned + negative" = crisis signal worth flagging.
  - The LLM's chosen snippet usually captures the strongest sentiment
    cue from the thread (it's why the LLM grabbed that exact passage),
    so signal density per byte is high.

Output is bounded to 5 enum values + one short summary :
  sentiment ∈ {positive, negative, neutral, mixed, unclear}
    - unclear : Haiku couldn't read the sentiment because the snippets
      are too thin (no body text from the discussion, just a citation
      marker). This is distinct from "neutral" which means "factual,
      no opinion expressed" - the user reads them very differently.
  summary  ≤ 200 chars, neutral observer voice
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
TIMEOUT = 60.0

_PROMPT = """You read these snippets that an LLM (ChatGPT or Gemini) captured around a Reddit thread URL when answering a user question. The snippets are short (~200 chars each) and may include the link inline. Tell me the overall sentiment toward the brands of interest. Stay neutral - you are an analyst, not an advocate.

Subreddit: r/{subreddit}
Reddit URL: {url}
Brands of interest: {brands}

LLM citation snippets (what the LLMs wrote when citing this thread):
{snippets}

Return ONLY this JSON (no markdown):

{{
  "sentiment": "positive" | "negative" | "neutral" | "mixed" | "unclear",
  "summary": "one neutral sentence (<= 200 chars) describing what the snippets suggest about the brand(s) of interest in this Reddit discussion. If no brand is clearly mentioned, describe the topic of the citation."
}}

Rules:
- "positive" : the snippets suggest redditors recommend / praise the brand(s)
- "negative" : the snippets suggest redditors complain / warn against the brand(s)
- "mixed"    : signals of both pros AND cons
- "neutral"  : the brand is referenced as fact / source, no clear sentiment expressed in the discussion
- "unclear"  : you cannot determine sentiment from the snippets because they are too thin (e.g. just "[Source: reddit.com]") and contain no actual discussion content. DO NOT default to "neutral" in this case - the user reads them very differently.
"""


def _format_snippets(snippets: list[str]) -> str:
    """Join the LLM citation snippets one per bullet. Each is already
    pre-truncated (~200 chars) so we don't need additional trimming."""
    cleaned = [s.strip() for s in (snippets or []) if s and s.strip()]
    if not cleaned:
        return "(no snippets captured)"
    return "\n".join(f"- {s}" for s in cleaned[:10])


def _build_prompt_from_snippets(url: str, subreddit: str | None, snippets: list[str], brand_names: list[str]) -> str:
    return _PROMPT.format(
        url=url or "(unknown)",
        subreddit=subreddit or "?",
        snippets=_format_snippets(snippets),
        brands=", ".join(brand_names) or "(none specified)",
    )


async def _call_haiku(prompt: str, api_key: str) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": HAIKU_MODEL,
                "max_tokens": 300,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        body = data["content"][0]["text"]
        # Be permissive : the model sometimes wraps in ```json...```.
        body = body.strip()
        if body.startswith("```"):
            body = body.strip("` \n")
            if body.lower().startswith("json"):
                body = body[4:].lstrip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # Salvage : find the first {...} block.
            i, j = body.find("{"), body.rfind("}")
            if i >= 0 and j > i:
                try:
                    return json.loads(body[i:j + 1])
                except json.JSONDecodeError:
                    pass
            raise


def classify_snippets(
    url: str,
    subreddit: str | None,
    snippets: list[str],
    brand_names: list[str],
    api_key: str,
) -> Optional[dict]:
    """Run Haiku on one URL's LLM citation snippets. Returns {sentiment,
    summary} or None on failure. Always non-fatal - the caller persists
    the row regardless."""
    if not api_key:
        return None
    cleaned = [s for s in (snippets or []) if s and s.strip()]
    if not cleaned:
        return None
    prompt = _build_prompt_from_snippets(url, subreddit, cleaned, brand_names)
    try:
        result = asyncio.run(_call_haiku(prompt, api_key))
    except Exception:  # noqa: BLE001
        logger.exception(f"reddit_sentiment failed for {url}")
        return None
    sentiment = (result.get("sentiment") or "").lower().strip()
    if sentiment not in ("positive", "negative", "neutral", "mixed", "unclear"):
        sentiment = "unclear"
    summary = (result.get("summary") or "").strip()[:300]
    return {"sentiment": sentiment, "summary": summary}
