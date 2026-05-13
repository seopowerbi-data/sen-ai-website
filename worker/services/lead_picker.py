"""Lead picker — auto-suggest the most relevant primary brand per FAQ opportunity.

Runs once per `materialize_content_items` execution. Takes the workspace brand
catalog + per-item topic/question context, asks Claude to map each item to the
brand whose product / positioning best fits, returns `{item_id: brand_id}` for
the caller to write into `item.promoted_brand_ids`.

Why batched
-----------
Materialize creates 5-15 items per scan (FAQ critique + haute). Instead of N
calls (~5-10s each, ~$0.005 each on Haiku → $0.05/scan), we send a SINGLE
prompt listing every item + every primary brand, and let the model emit one
mapping. Cost: ~$0.005-0.01 per scan, latency ~3-5s, and the model can reason
across items (e.g. avoid recommending the same gamme to every baby-skin
question if multiple gammes target that audience).

Fallback policy
---------------
- Client has 0 or 1 primary brand with a domain → no choice to make, return {}
  (caller falls back to workspace default behavior).
- LLM call fails / response unparseable → return {} (same fallback).
- LLM returns brand_id not in the workspace catalog → silently drop that
  suggestion (other items still pass through).
- LLM omits an item from the response → that item keeps the workspace default.

Cost cap rationale (see `feedback_cap_user_triggered_llm_ops.md`)
-----------------------------------------------------------------
This op is system-triggered (scan-end), not user-triggered, AND already bounded
by scan frequency × items/scan. Per-call cost is ~1 cent. No additional cap
needed — but if user clicks "Find a different LEAD" in the future, that
endpoint will need its own per-item cap.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterable
from uuid import UUID

import httpx
from sqlalchemy.orm import Session

from config import settings

logger = logging.getLogger(__name__)


# Haiku is fine for this mapping task — the input is short (brand catalog +
# 5-15 items), the reasoning is shallow ("which brand is most semantically
# relevant to this question"), and we already use Haiku for classify_topics /
# generate_personas. Sonnet would 10x cost for marginal quality gain here.
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_OUTPUT_TOKENS = 2048  # 15 items × ~100 tok each = 1500 + buffer
_TIMEOUT_SECONDS = 45


_SYSTEM_PROMPT = """You are picking the single most relevant lead brand for each FAQ opportunity in a content-marketing pipeline.

You will receive:
- A workspace context block (industry, positioning, editorial voice) that frames the brand portfolio.
- A catalog of primary brands the workspace promotes — each with an ID, name, domain, and short positioning.
- A list of FAQ opportunities — each with an ID, a topic, the user question, and the audience persona.

For each opportunity, choose the ONE brand from the catalog whose products / positioning is the best semantic fit for that specific question. Your goal is editorial relevance, not portfolio balance — if 5 questions all map to the same brand, that's correct.

Return ONLY valid JSON (no markdown, no commentary) with this exact structure:

{
  "assignments": [
    {"item_id": "<uuid>", "brand_id": "<uuid>", "reason": "<5-15 word justification>"}
  ]
}

Rules:
- Use the EXACT item_id and brand_id values from the input — never invent IDs.
- Every input item MUST appear exactly once in the output. If you genuinely cannot decide between two brands, pick the one whose domain looks more authoritative for the question's persona.
- Keep `reason` short — it's read by the user, not by another LLM.
- If the catalog has only one brand, assign every item to that brand."""


def _format_brands_block(brands: list[dict]) -> str:
    lines = []
    for b in brands:
        desc = (b.get("description") or "").strip()
        desc_suffix = f" — {desc}" if desc else ""
        lines.append(
            f"- id={b['id']} | name={b['name']} | domain={b.get('domain') or '(none)'}"
            f"{desc_suffix}"
        )
    return "\n".join(lines)


def _format_items_block(items: list[dict]) -> str:
    lines = []
    for it in items:
        persona = (it.get("persona") or "").strip() or "general"
        topic = (it.get("topic") or "").strip() or "(no topic)"
        lines.append(
            f"- id={it['id']} | topic={topic} | persona={persona}\n"
            f"    question: {it['question']}"
        )
    return "\n".join(lines)


def _format_context_block(brief: dict | None, client_name: str) -> str:
    if not brief:
        return f"Workspace: {client_name}. (No detailed brief — use the brand catalog alone.)"
    parts = [f"Workspace: {client_name}."]
    for key, label in (
        ("industry", "Industry"),
        ("brand_positioning", "Positioning"),
        ("editorial_voice", "Editorial voice"),
        ("target_audience", "Audience"),
    ):
        val = (brief.get(key) or "").strip()
        if val:
            parts.append(f"{label}: {val}")
    return "\n".join(parts)


def _extract_json(text: str) -> dict | None:
    """Tolerant JSON extraction — strips code fences, falls back to outer braces."""
    if not text:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", text.strip())
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _call_claude(prompt: str, model: str, api_key: str) -> tuple[str, dict]:
    payload = {
        "model": model,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "temperature": 0.2,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT_SECONDS) as http:
        resp = http.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    text = data.get("content", [{}])[0].get("text", "")
    usage = data.get("usage", {})
    return text, {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def pick_leads_for_items(
    client_id,
    scan_id,
    items: list[dict],
    db: Session,
) -> dict[str, dict]:
    """Map each item to the most relevant primary brand.

    Args:
        client_id: workspace UUID.
        scan_id: scan UUID — for logging only.
        items: list of {id (str), topic (str), question (str), persona (str)}.
        db: session — read-only here, plus llm_usage_log insert.

    Returns:
        {item_id (str): {"brand_id": str, "reason": str, "model": str}}.
        Empty dict on any failure or when no choice exists.
    """
    if not items:
        return {}

    from models import Client, ClientBrand

    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        logger.warning(f"lead_picker: client {client_id} not found")
        return {}

    primary_ids = list(client.primary_brand_ids or [])
    if not primary_ids:
        return {}

    brand_rows = (
        db.query(ClientBrand)
        .filter(ClientBrand.id.in_(primary_ids))
        .all()
    )
    by_id = {b.id: b for b in brand_rows}
    # Preserve workspace order. Skip brands without a domain — a LEAD without
    # a domain can't drive FAQPageMatcher, so it's not a useful suggestion.
    catalog = []
    for bid in primary_ids:
        b = by_id.get(bid)
        if b and b.domain and b.domain.strip():
            catalog.append(b)

    if len(catalog) < 2:
        # 0 or 1 candidate — workspace default is already correct.
        logger.info(
            f"lead_picker: client {client_id} has {len(catalog)} usable primary brand(s) — "
            f"skipping LLM, falling back to workspace default"
        )
        return {}

    if not settings.anthropic_api_key:
        logger.warning("lead_picker: ANTHROPIC_API_KEY not set — skipping")
        return {}

    # Build brand descriptions from the workspace brief if available, so the
    # LLM sees positioning context per brand. Falls back gracefully if the
    # brief hasn't been generated.
    apps = client.apps or {}
    brief = apps.get("client_brief") or {}
    brief_primary = {
        (b.get("name") or "").strip().lower(): (b.get("description") or "").strip()
        for b in (brief.get("primary_brands") or [])
        if isinstance(b, dict)
    }

    brands_payload = [
        {
            "id": str(b.id),
            "name": b.name,
            "domain": b.domain,
            "description": brief_primary.get((b.name or "").strip().lower(), ""),
        }
        for b in catalog
    ]
    valid_brand_ids = {b["id"] for b in brands_payload}

    items_payload = [
        {
            "id": str(it["id"]),
            "topic": it.get("topic") or "",
            "question": it.get("question") or "",
            "persona": it.get("persona") or "",
        }
        for it in items
    ]

    prompt = (
        _format_context_block(brief, client.name)
        + "\n\nBrand catalog:\n"
        + _format_brands_block(brands_payload)
        + "\n\nFAQ opportunities:\n"
        + _format_items_block(items_payload)
    )

    model = _DEFAULT_MODEL
    start = time.monotonic()
    try:
        raw, usage = _call_claude(prompt, model, settings.anthropic_api_key)
    except Exception as e:
        logger.warning(f"lead_picker: Claude call failed for scan {scan_id}: {e}")
        try:
            from adapters.llm_logger import log_llm_usage
            log_llm_usage(
                db, provider="anthropic", model=model,
                operation="pick_leads", scan_id=str(scan_id),
                client_id=str(client_id), error=True,
            )
        except Exception:
            pass
        return {}

    duration_ms = int((time.monotonic() - start) * 1000)
    parsed = _extract_json(raw)
    if not parsed or not isinstance(parsed.get("assignments"), list):
        logger.warning(
            f"lead_picker: unparseable response for scan {scan_id} — "
            f"raw[:200]={raw[:200]!r}"
        )
        try:
            from adapters.llm_logger import log_llm_usage
            log_llm_usage(
                db, provider="anthropic", model=model,
                operation="pick_leads",
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                duration_ms=duration_ms,
                scan_id=str(scan_id), client_id=str(client_id), error=True,
            )
        except Exception:
            pass
        return {}

    out: dict[str, dict] = {}
    valid_item_ids = {it["id"] for it in items_payload}
    for entry in parsed["assignments"]:
        if not isinstance(entry, dict):
            continue
        iid = str(entry.get("item_id") or "").strip()
        bid = str(entry.get("brand_id") or "").strip()
        if iid not in valid_item_ids:
            continue
        if bid not in valid_brand_ids:
            logger.info(
                f"lead_picker: dropping invalid brand_id={bid!r} for item {iid} "
                f"(not in workspace primary catalog)"
            )
            continue
        # Re-normalize to canonical UUID string form so downstream array writes
        # don't trip on case / whitespace differences.
        try:
            iid_canon = str(UUID(iid))
            bid_canon = str(UUID(bid))
        except ValueError:
            continue
        out[iid_canon] = {
            "brand_id": bid_canon,
            "reason": (entry.get("reason") or "").strip()[:200],
            "model": model,
        }

    try:
        from adapters.llm_logger import log_llm_usage
        log_llm_usage(
            db, provider="anthropic", model=model,
            operation="pick_leads",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            duration_ms=duration_ms,
            scan_id=str(scan_id), client_id=str(client_id),
        )
    except Exception:
        pass

    logger.info(
        f"lead_picker: scan={scan_id} mapped {len(out)}/{len(items)} items "
        f"to leads ({duration_ms}ms, model={model})"
    )
    return out
