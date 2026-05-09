"""Auto-classify and clean detected brands using Claude.

Called after LLM scan to:
1. Filter non-brands (medical terms, generic terms, institutions)
2. Capitalize names properly
3. Auto-classify as target_brand, target_gamme, competitor, competitor_gamme
4. Match product lines to their parent brand
"""

import json
import logging
import time

import httpx

from config import settings
from schemas import BrandClassification, validate_items
from utils import max_tokens_for

logger = logging.getLogger(__name__)

BRAND_CLEANUP_PROMPT = """Here is a list of brand names detected in AI responses about the website {domain}.
The website belongs to the brand "{site_brand}".

{domain_context}

Detected names:
{brand_list}

Already classified brands (for context):
{existing_brands}

For each detected name, determine:
1. Is it a real commercial brand/product line? (NOT a medical term, ingredient, institution, or generic word)
2. If yes: proper capitalized name and category

Categories:
- "target_brand": the main brand of {domain}
- "target_gamme": a product line of the main brand
- "competitor": a competing brand
- "competitor_gamme": a product line of a competitor (specify which competitor)
- "ignore": not a brand (medical term, institution, ingredient, generic)

Reply ONLY in JSON:
{{
  "brands": [
    {{"original": "cerave", "name": "CeraVe", "category": "competitor", "parent": null}},
    {{"original": "eczéma", "name": null, "category": "ignore", "parent": null}},
    {{"original": "la roche-posay toleriane", "name": "Toleriane", "category": "competitor_gamme", "parent": "La Roche-Posay"}}
  ]
}}"""


async def classify_brands(domain: str, site_brand: str,
                          unclassified: list[str], existing: list[dict],
                          anthropic_api_key: str, domain_context: str = "") -> list[dict]:
    """
    Classify unclassified brands using Claude.

    Args:
        domain: scanned domain
        site_brand: main brand name of the site
        unclassified: list of brand names to classify
        existing: list of already classified brands [{name, category}]
        anthropic_api_key: Claude API key

    Returns:
        dict {brands: list of {original, name, category, parent}, model, input_tokens, output_tokens, duration_ms}
    """
    if not unclassified:
        return {"brands": [], "model": settings.task_models["cleanup_brands"],
                "input_tokens": 0, "output_tokens": 0, "duration_ms": 0}

    existing_str = "\n".join(f"- {b['name']} ({b['category']})" for b in existing[:20])
    brand_list = "\n".join(f"- {name}" for name in unclassified)

    prompt = BRAND_CLEANUP_PROMPT.format(
        domain=domain,
        site_brand=site_brand,
        domain_context=domain_context,
        brand_list=brand_list,
        existing_brands=existing_str or "None yet",
    )

    start = time.time()
    model = settings.task_models["cleanup_brands"]

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens_for(model, cap=4096),
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    text = data["content"][0]["text"].strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0].strip()

    result = json.loads(text)
    raw_brands = result.get("brands", [])

    # Pydantic validation: drop malformed items, log warnings
    brands_validated = validate_items(
        raw_brands, BrandClassification, "cleanup_brands.brands"
    )
    brands = [b.model_dump() for b in brands_validated]

    duration = int((time.time() - start) * 1000)
    logger.info(f"Brand cleanup: {len(unclassified)} → {len([b for b in brands if b.get('category') != 'ignore'])} brands, "
                f"{len([b for b in brands if b.get('category') == 'ignore'])} ignored, {duration}ms")

    usage = data.get("usage", {})
    return {
        "brands": brands,
        "model": model,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "duration_ms": duration,
    }
