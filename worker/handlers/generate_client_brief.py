"""Handler: generate workspace-level client brief (separate from per-scan domain_brief).

Where per-scan `domain_brief` describes the SCANNED domain (often a competitor),
the **client brief** describes the WORKSPACE itself — the user's company, their
brand portfolio, voice, positioning. This is the durable context that downstream
content generators (Phase B FAQ, Phase C Article) inject so the output sounds
like the user's brand even when generated from an opportunity discovered on a
competitor scan.

Reads :
- client.name                                 → company hint
- client.primary_brand_ids → ClientBrand rows → canonical brand list (with domains)

Writes :
- client.apps['client_brief'] = {
      company_overview, industry, primary_brands[{name, domain, role, description}],
      key_competitors, target_audience, products_services,
      editorial_voice, brand_positioning,
      generated_via, generated_at, edited_by_user (defaults False)
  }

Provider strategy mirrors generate_domain_brief.py — OpenAI + web_search primary,
Gemini + grounding fallback, Claude (training-only) last resort.
"""

import json
import logging
import re
from datetime import datetime

import httpx
import openai
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from config import settings
from models import Client, ClientBrand

logger = logging.getLogger(__name__)


CLIENT_BRIEF_PROMPT = """You are producing a long-lived workspace brief for a content-marketing platform. The brief will be injected into FAQ + article generation prompts as the *brand voice* context. It must describe the COMPANY, not any single website.

Workspace name: {client_name}
Primary brands the workspace promotes:
{primary_brands_block}

Use web search to verify and enrich. Return ONLY valid JSON (no markdown, no commentary) with this exact structure :

{{
  "company_overview": "2-3 sentences naming the parent company / group, what it does, where it operates, and at what scale",
  "industry": "Industry / sub-industry — be specific (e.g., 'Dermo-cosmetics — sensitive skin care' not just 'Cosmetics')",
  "primary_brands": [
    {{
      "name": "Brand name (matches input)",
      "domain": "official domain or null",
      "role": "lead | sister | acquired | distribution",
      "description": "1 sentence positioning the brand in the workspace's portfolio"
    }}
  ],
  "key_competitors": ["3-8 direct competitor companies (not brand-vs-brand — group level)"],
  "target_audience": "1-2 sentences describing who the company serves (B2C/B2B mix, age, profile, need)",
  "products_services": ["Top product/service categories — be concrete"],
  "editorial_voice": "1 sentence on tone (e.g., 'expert, reassuring, science-led — never salesy or alarmist')",
  "brand_positioning": "1 sentence on positioning (e.g., 'premium dermo-cosmetics for compromised/sensitive skin, trusted by pharmacists')"
}}

Rules :
- The first brand in primary_brands MUST have role="lead" (it's the input order).
- key_competitors are COMPANIES, not products. If a competitor company owns multiple brands, list the company name once.
- Stay concise — this brief is read by other LLMs, not humans. Every word counts as input tokens.
- If the workspace has only ONE primary brand, primary_brands has exactly 1 entry with role="lead".
"""


CLAUDE_FALLBACK_PROMPT = """You are producing a long-lived workspace brief based on training knowledge (no web access).

Workspace name: {client_name}
Primary brands the workspace promotes:
{primary_brands_block}

If you don't recognise a specific brand, infer conservatively from its name + the company name. Note uncertainty in description rather than fabricating.

Return ONLY valid JSON with this exact structure :

{{
  "company_overview": "2-3 sentences",
  "industry": "Industry / sub-industry",
  "primary_brands": [
    {{
      "name": "...", "domain": "... or null",
      "role": "lead | sister | acquired | distribution",
      "description": "1 sentence"
    }}
  ],
  "key_competitors": ["3-8 competitor companies"],
  "target_audience": "1-2 sentences",
  "products_services": ["Top categories"],
  "editorial_voice": "1 sentence",
  "brand_positioning": "1 sentence"
}}

First brand MUST have role="lead". key_competitors are COMPANIES not brands.
"""


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{', text)
    if not match:
        return None
    depth = 0
    for i in range(match.start(), len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[match.start():i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _format_primary_brands_block(brands: list[ClientBrand]) -> str:
    if not brands:
        return "(none — workspace primary_brand_ids is empty)"
    lines = []
    for i, b in enumerate(brands, 1):
        domain = f" — {b.domain}" if b.domain else ""
        lines.append(f"  {i}. {b.name}{domain}")
    return "\n".join(lines)


def _try_openai(client_name: str, primary_brands_block: str, api_key: str,
                model: str) -> tuple[dict | None, str, dict]:
    client = openai.OpenAI(api_key=api_key, timeout=60)
    prompt = CLIENT_BRIEF_PROMPT.format(
        client_name=client_name, primary_brands_block=primary_brands_block,
    )
    response = client.responses.create(
        model=model, tools=[{"type": "web_search"}],
        input=prompt, temperature=0.3,
    )
    text = response.output_text or ""
    usage_obj = getattr(response, "usage", None)
    usage = {
        "input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0,
    }
    return _extract_json(text), text, usage


def _try_gemini(client_name: str, primary_brands_block: str, api_key: str,
                model: str) -> tuple[dict | None, str, dict]:
    from seo_llm.src.llm_client import LLMClient
    llm = LLMClient(provider="gemini", api_key=api_key, model=model)
    prompt = CLIENT_BRIEF_PROMPT.format(
        client_name=client_name, primary_brands_block=primary_brands_block,
    )
    response = llm.generate(
        prompt, temperature=0.3, max_tokens=8000, use_grounding=True,
        agent_name="generate_client_brief_gemini",
    )
    text = response.get("text", "")
    usage_raw = response.get("usage", {}) or {}
    usage = {
        "input_tokens": usage_raw.get("prompt_tokens", 0) or usage_raw.get("input_tokens", 0),
        "output_tokens": usage_raw.get("completion_tokens", 0) or usage_raw.get("output_tokens", 0),
    }
    return _extract_json(text), text, usage


def _try_claude(client_name: str, primary_brands_block: str, api_key: str,
                model: str) -> tuple[dict | None, str, dict]:
    prompt = CLAUDE_FALLBACK_PROMPT.format(
        client_name=client_name, primary_brands_block=primary_brands_block,
    )
    payload = {
        "model": model, "max_tokens": 4096, "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key, "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    with httpx.Client(timeout=60) as http:
        resp = http.post("https://api.anthropic.com/v1/messages",
                         json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    text = data.get("content", [{}])[0].get("text", "")
    usage = data.get("usage", {})
    return _extract_json(text), text, {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    """Generate the workspace brief for a client. scan_id is unused (workspace-scoped)."""
    client_id = job_payload.get("client_id")
    if not client_id:
        raise ValueError("generate_client_brief requires client_id in job payload")

    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise ValueError(f"Client {client_id} not found")

    # Refuse to overwrite if user has manually edited the brief — they should
    # explicitly DELETE it via the UI before regeneration.
    apps = client.apps or {}
    existing = apps.get("client_brief") or {}
    if existing.get("edited_by_user"):
        logger.info(f"Brief for client {client_id} already edited by user — skipping")
        return {"status": "skipped", "reason": "user_edited"}

    primary_ids = list(client.primary_brand_ids or [])
    primary_brands = []
    if primary_ids:
        primary_brands = (
            db.query(ClientBrand)
            .filter(ClientBrand.id.in_(primary_ids))
            .all()
        )
        # Preserve the user-defined order from primary_brand_ids
        by_id = {b.id: b for b in primary_brands}
        primary_brands = [by_id[bid] for bid in primary_ids if bid in by_id]

    primary_brands_block = _format_primary_brands_block(primary_brands)
    logger.info(
        f"Generating client brief for {client.name} ({len(primary_brands)} primary brands)"
    )

    brief = None
    used_provider = None
    raw_texts: dict[str, str] = {}

    # ── Tier 1: OpenAI + web_search ─────────────────────────────────────
    if settings.openai_api_key:
        primary_model = settings.task_models.get("generate_domain_brief", "gpt-4.1-mini")
        try:
            parsed, raw, usage = _try_openai(client.name, primary_brands_block,
                                             settings.openai_api_key, primary_model)
            raw_texts["openai"] = raw
            if parsed:
                brief = parsed
                used_provider = "openai"
                from adapters.llm_logger import log_llm_usage
                log_llm_usage(
                    db, provider="openai", model=primary_model,
                    operation="generate_client_brief",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    client_id=client_id,
                )
            else:
                logger.warning(
                    f"OpenAI returned malformed JSON for client {client_id} "
                    f"({len(raw)} chars). Start: {raw[:200]}"
                )
        except Exception as e:
            logger.warning(f"OpenAI brief failed for client {client_id}: {e}")

    # ── Tier 2: Gemini with grounding ───────────────────────────────────
    from services.gemini_key_pool import get_gemini_pool
    gemini_pool = get_gemini_pool()
    if brief is None and gemini_pool.has_keys():
        gemini_model = settings.task_models.get("generate_domain_brief_gemini",
                                                "gemini-2.5-flash")
        logger.warning(
            f"Falling back to Gemini ({gemini_model}) for client brief {client_id}"
        )
        try:
            parsed, raw, usage = _try_gemini(client.name, primary_brands_block,
                                             gemini_pool.next_key(), gemini_model)
            raw_texts["gemini"] = raw
            if parsed:
                brief = parsed
                used_provider = "gemini"
                from adapters.llm_logger import log_llm_usage
                log_llm_usage(
                    db, provider="gemini", model=gemini_model,
                    operation="generate_client_brief_gemini",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    client_id=client_id,
                )
        except Exception as e:
            logger.warning(f"Gemini brief failed for client {client_id}: {e}")

    # ── Tier 3: Claude (training only) ──────────────────────────────────
    if brief is None and settings.anthropic_api_key:
        claude_model = settings.task_models.get("generate_domain_brief_claude",
                                                "claude-sonnet-4-6")
        logger.warning(
            f"Falling back to Claude ({claude_model}) for client brief {client_id}"
        )
        try:
            parsed, raw, usage = _try_claude(client.name, primary_brands_block,
                                             settings.anthropic_api_key, claude_model)
            raw_texts["claude"] = raw
            if parsed:
                brief = parsed
                used_provider = "claude"
                from adapters.llm_logger import log_llm_usage
                log_llm_usage(
                    db, provider="anthropic", model=claude_model,
                    operation="generate_client_brief_claude",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    client_id=client_id,
                )
        except Exception as e:
            logger.warning(f"Claude brief failed for client {client_id}: {e}")

    if brief is None:
        sizes = {p: len(t) for p, t in raw_texts.items()}
        raise RuntimeError(
            f"Client brief generation failed across all 3 providers for client "
            f"{client_id}. Response sizes: {sizes}"
        )

    # ── Persist on client.apps['client_brief'] ──────────────────────────
    brief["generated_via"] = used_provider
    brief["generated_at"] = datetime.utcnow().isoformat()
    brief["edited_by_user"] = False

    apps = dict(client.apps or {})
    apps["client_brief"] = brief
    client.apps = apps
    flag_modified(client, "apps")
    db.commit()

    logger.info(
        f"Client brief saved for {client.name} via {used_provider} "
        f"({len(brief.get('primary_brands', []))} brands, "
        f"{len(brief.get('key_competitors', []))} competitors)"
    )
    return {
        "status": "ok", "provider": used_provider,
        "primary_brands": len(brief.get("primary_brands", [])),
        "key_competitors": len(brief.get("key_competitors", [])),
    }
