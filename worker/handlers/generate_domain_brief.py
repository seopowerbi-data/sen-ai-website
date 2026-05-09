"""Handler: generate domain brief with multi-provider fallback chain.

Produces a structured business-intelligence document about the scanned domain.
Stores in scan.config.domain_brief. Pre-populates Gate 2 with competitors from brief.

Provider strategy (3-tier fallback):
  1. Primary       — OpenAI Responses API + web_search tool (current/up-to-date data)
  2. Fallback #1   — Gemini with grounding (also web-aware, alternative provider)
  3. Last resort   — Anthropic Claude (training-knowledge only, no web access)
                     Quality lower for very recent/niche sites but useful for
                     well-known brands which is the dominant use case.
  4. All 3 fail    — RAISE. Three independent providers returning malformed JSON
                     in the same attempt is almost certainly a code/prompt bug, not
                     a transient provider issue. Worker retries up to max_attempts
                     (3); if still failing across all 9 calls, scan is marked failed
                     and the user sees a real error to investigate.

The brief is OPTIONAL context injected into 5 downstream prompts. We try hard
to produce one, but ultimately a parse failure is treated as a real bug rather
than silently skipped — the user explicitly asked for visibility on this case.
"""

import json
import logging
import re
from datetime import datetime

import httpx
import openai
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from config import settings
from models import Scan, ClientBrand, ScanBrandClassification
from schemas import DomainBrief, validate_object

logger = logging.getLogger(__name__)

WEB_BRIEF_PROMPT = """Research the website {domain} using web search and provide structured business intelligence.

You MUST search the web to find accurate, up-to-date information about this website/company.

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
  "company": "Full company name with parent group if applicable",
  "description": "2-3 sentence description of what the company does, what they sell, through which channels",
  "industry": "Industry / Sub-industry",
  "country": "Primary market country",
  "brands": ["Brand names owned by this company"],
  "product_lines": ["Product line name (purpose/category)" for each major product range],
  "services": ["Any services offered beyond products"],
  "competitors": [
    {{"name": "Competitor Name", "products": ["Their competing product lines"]}}
  ],
  "topics": ["Key themes/topics the website covers"],
  "target_audience": "Description of who their customers are, demographics, needs"
}}

Be thorough and specific. For competitors, list 5-10 direct competitors with their key product lines.
For product_lines, list the actual product range names, not generic categories.
"""

CLAUDE_FALLBACK_PROMPT = """Based on your training knowledge, provide structured business intelligence about the website {domain}.

If you don't have specific information about this exact domain, infer from the domain name and any related companies/brands you know about. Make conservative inferences and note uncertainty in the description rather than fabricating specifics.

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
  "company": "Full company name with parent group if applicable",
  "description": "2-3 sentence description of what the company does, what they sell, through which channels",
  "industry": "Industry / Sub-industry",
  "country": "Primary market country",
  "brands": ["Brand names owned by this company"],
  "product_lines": ["Product line names"],
  "services": ["Any services offered beyond products"],
  "competitors": [
    {{"name": "Competitor Name", "products": ["Their competing product lines"]}}
  ],
  "topics": ["Key themes/topics the website covers"],
  "target_audience": "Description of who their customers are"
}}

For competitors, list 5-10 direct competitors with their key product lines.
"""


def _extract_json(text: str) -> dict | None:
    """Robust JSON extraction: strips markdown fences, falls back to brace-counter."""
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


def _try_openai(domain: str, api_key: str, model: str) -> tuple[dict | None, str]:
    """Primary: OpenAI Responses API + web_search. Returns (parsed_or_None, raw_text)."""
    client = openai.OpenAI(api_key=api_key, timeout=60)
    prompt = WEB_BRIEF_PROMPT.format(domain=domain)
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        input=prompt,
        temperature=0.3,
    )
    text = response.output_text or ""
    return _extract_json(text), text


def _try_gemini(domain: str, api_key: str, model: str) -> tuple[dict | None, str, dict]:
    """Fallback #1: Gemini with grounding (web-aware). Returns (parsed_or_None, raw_text, usage)."""
    from seo_llm.src.llm_client import LLMClient
    client = LLMClient(provider="gemini", api_key=api_key, model=model)
    prompt = WEB_BRIEF_PROMPT.format(domain=domain)
    response = client.generate(
        prompt,
        temperature=0.3,
        max_tokens=8000,
        use_grounding=True,
        agent_name="generate_domain_brief_gemini",
    )
    text = response.get("text", "")
    usage = {
        "input_tokens": response.get("usage", {}).get("prompt_tokens", 0)
                        or response.get("usage", {}).get("input_tokens", 0),
        "output_tokens": response.get("usage", {}).get("completion_tokens", 0)
                         or response.get("usage", {}).get("output_tokens", 0),
    }
    return _extract_json(text), text, usage


def _try_claude(domain: str, api_key: str, model: str) -> tuple[dict | None, str, dict]:
    """Last resort: Claude with training knowledge. Returns (parsed_or_None, raw_text, usage)."""
    prompt = CLAUDE_FALLBACK_PROMPT.format(domain=domain)
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    text = data.get("content", [{}])[0].get("text", "")
    return _extract_json(text), text, data.get("usage", {})


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise ValueError(f"Scan {scan_id} not found")

    # Skip if user already edited the brief
    existing_brief = (scan.config or {}).get("domain_brief")
    if existing_brief and existing_brief.get("edited_by_user"):
        logger.info(f"Brief already edited by user for scan {scan_id}, skipping generation")
        return {"status": "skipped", "reason": "user_edited"}

    domain = scan.domain
    scan.progress_message = f"Researching {domain}..."
    db.commit()

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    brief = None
    used_provider = None
    raw_texts = {}

    # ── Tier 1: OpenAI + web_search ─────────────────────────────────────
    primary_model = settings.task_models["generate_domain_brief"]
    logger.info(f"Generating brief for {domain} via OpenAI ({primary_model}) + web_search")
    try:
        parsed, raw = _try_openai(domain, settings.openai_api_key, primary_model)
        raw_texts["openai"] = raw
        if parsed:
            brief = parsed
            used_provider = "openai"
        else:
            logger.warning(
                f"OpenAI returned malformed JSON for {domain} ({len(raw)} chars). "
                f"Raw start: {raw[:200]}"
            )
    except Exception as e:
        logger.warning(f"OpenAI request threw exception for {domain}: {e}")

    # ── Tier 2: Gemini with grounding ───────────────────────────────────
    from services.gemini_key_pool import get_gemini_pool
    gemini_pool = get_gemini_pool()
    if brief is None and gemini_pool.has_keys():
        gemini_model = settings.task_models["generate_domain_brief_gemini"]
        logger.warning(
            f"OpenAI did not produce a usable brief for {domain}, "
            f"falling back to Gemini ({gemini_model}) with grounding"
        )
        try:
            parsed, raw, usage = _try_gemini(domain, gemini_pool.next_key(), gemini_model)
            raw_texts["gemini"] = raw
            if parsed:
                brief = parsed
                used_provider = "gemini"
                from adapters.llm_logger import log_llm_usage
                log_llm_usage(
                    db, provider="gemini", model=gemini_model,
                    operation="generate_domain_brief_gemini",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    scan_id=scan_id, client_id=str(scan.client_id),
                )
            else:
                logger.warning(
                    f"Gemini also returned malformed JSON for {domain} ({len(raw)} chars). "
                    f"Raw start: {raw[:200]}"
                )
        except Exception as e:
            logger.warning(f"Gemini request threw exception for {domain}: {e}")

    # ── Tier 3: Claude (training only, no web) ──────────────────────────
    if brief is None and settings.anthropic_api_key:
        claude_model = settings.task_models["generate_domain_brief_claude"]
        logger.warning(
            f"Gemini did not produce a usable brief for {domain}, "
            f"falling back to Claude ({claude_model}, training-knowledge only)"
        )
        try:
            parsed, raw, usage = _try_claude(domain, settings.anthropic_api_key, claude_model)
            raw_texts["claude"] = raw
            if parsed:
                brief = parsed
                used_provider = "claude"
                from adapters.llm_logger import log_llm_usage
                log_llm_usage(
                    db, provider="anthropic", model=claude_model,
                    operation="generate_domain_brief_claude",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    scan_id=scan_id, client_id=str(scan.client_id),
                )
            else:
                logger.warning(
                    f"Claude also returned malformed JSON for {domain} ({len(raw)} chars). "
                    f"Raw start: {raw[:200]}"
                )
        except Exception as e:
            logger.warning(f"Claude request threw exception for {domain}: {e}")

    # ── All 3 failed → raise (real bug, not transient) ──────────────────
    if brief is None:
        sizes = {p: len(t) for p, t in raw_texts.items()}
        raise RuntimeError(
            f"Brief generation failed across all 3 providers for {domain}. "
            f"Response sizes: {sizes}. This is likely a prompt/code bug — "
            f"three independent providers don't produce malformed JSON simultaneously."
        )

    # ── Pydantic validation: skip on failure (don't fail the scan) ──────
    try:
        brief_validated = validate_object(brief, DomainBrief, "generate_domain_brief")
        brief = brief_validated.model_dump()
    except RuntimeError as e:
        logger.warning(
            f"Brief validation failed for {domain} (provider={used_provider}), "
            f"scan continues without brief: {e}"
        )
        return {
            "status": "skipped",
            "reason": "validation_failed",
            "provider": used_provider,
        }

    logger.info(
        f"Brief generated for {domain} (provider={used_provider}): "
        f"{brief.get('company', '?')} — {brief.get('industry', '?')}"
    )

    # ── Persist + pre-populate Gate 2 ───────────────────────────────────
    config = dict(scan.config or {})
    config["domain_brief"] = brief
    config["domain_brief_provider"] = used_provider  # for audit/debug
    scan.config = config
    flag_modified(scan, "config")
    scan.updated_at = datetime.utcnow()
    db.commit()

    # Pre-populate Gate 2 with competitors from brief
    competitors_created = 0
    for comp in brief.get("competitors", []):
        comp_name = (comp.get("name") or "").strip()
        if not comp_name:
            continue

        existing = db.query(ClientBrand).filter(
            ClientBrand.client_id == scan.client_id,
            func.lower(ClientBrand.name) == comp_name.lower(),
        ).first()

        if not existing:
            brand = ClientBrand(
                client_id=scan.client_id,
                name=comp_name,
                canonical_name=comp_name,
                detected_in_scan_id=scan_id,
                auto_detected=True,
                validated_by_user=False,
                last_seen_at=datetime.utcnow(),
            )
            db.add(brand)
            db.flush()
        else:
            brand = existing
            existing.last_seen_at = datetime.utcnow()

        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.brand_id == brand.id,
        ).first()
        if not sbc:
            db.add(ScanBrandClassification(
                scan_id=scan_id,
                brand_id=brand.id,
                classification="competitor",
                is_focus=False,
                classified_by="brief",
                source="brief",
            ))
            competitors_created += 1
        elif sbc.classification == "unclassified":
            sbc.classification = "competitor"
            sbc.classified_by = "brief"
            sbc.source = "brief"
            competitors_created += 1

    # Pre-populate own brands
    for own_brand_name in brief.get("brands", []):
        own_brand_name = (own_brand_name or "").strip()
        if not own_brand_name:
            continue

        existing = db.query(ClientBrand).filter(
            ClientBrand.client_id == scan.client_id,
            func.lower(ClientBrand.name) == own_brand_name.lower(),
        ).first()

        if not existing:
            brand = ClientBrand(
                client_id=scan.client_id,
                name=own_brand_name,
                canonical_name=own_brand_name,
                detected_in_scan_id=scan_id,
                auto_detected=True,
                validated_by_user=False,
                last_seen_at=datetime.utcnow(),
            )
            db.add(brand)
            db.flush()
        else:
            brand = existing

        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.brand_id == brand.id,
        ).first()
        if not sbc:
            db.add(ScanBrandClassification(
                scan_id=scan_id,
                brand_id=brand.id,
                classification="my_brand",
                is_focus=False,
                classified_by="brief",
                source="brief",
            ))

    db.commit()
    logger.info(f"Gate 2 pre-populated with {competitors_created} competitors from brief")

    return {
        "status": "completed",
        "provider": used_provider,
        "company": brief.get("company"),
        "competitors_created": competitors_created,
    }
