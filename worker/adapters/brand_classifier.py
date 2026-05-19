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


# Chunk size for the Claude classification call. With 16K max_tokens output
# and ~100 tokens per brand object, 100 brands fits with ~6K tokens of headroom.
# Smaller chunks = more API calls but lower failure surface. 100 is the
# empirical sweet spot from the seo-llm port (`question_intent_classifier`
# batches at 30 — brand classifier prompt is denser so we do 100 here).
_BATCH_SIZE = 100


async def _classify_one_batch(
    domain: str, site_brand: str, batch: list[str],
    existing_str: str, domain_context: str, model: str,
    anthropic_api_key: str,
) -> tuple[list[dict], dict]:
    """One Claude call for one batch of brand names. Returns (brands, usage_dict)."""
    brand_list = "\n".join(f"- {name}" for name in batch)
    prompt = BRAND_CLEANUP_PROMPT.format(
        domain=domain,
        site_brand=site_brand,
        domain_context=domain_context,
        brand_list=brand_list,
        existing_brands=existing_str or "None yet",
    )

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens_for(model, cap=16384),
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    from adapters.json_utils import extract_json_object
    parsed = extract_json_object(data["content"][0]["text"])
    raw_brands = parsed.get("brands", [])
    validated = validate_items(raw_brands, BrandClassification, "cleanup_brands.brands")
    brands = [b.model_dump() for b in validated]
    return brands, data.get("usage", {})


async def classify_brands(domain: str, site_brand: str,
                          unclassified: list[str], existing: list[dict],
                          anthropic_api_key: str, domain_context: str = "") -> dict:
    """Classify unclassified brands using Claude, batched in chunks of 100.

    Pre-filters obvious noise (ingredients, domains, product types) before
    sending to Claude — sees `services.brand_noise_filter`. Noise items are
    returned as `category="ignore"` without burning a Claude call.

    Returns:
        dict {brands, model, input_tokens, output_tokens, duration_ms, batches}
    """
    from services.brand_noise_filter import filter_noise

    model = settings.task_models["cleanup_brands"]
    start = time.time()

    if not unclassified:
        return {"brands": [], "model": model, "batches": 0,
                "input_tokens": 0, "output_tokens": 0, "duration_ms": 0}

    # Pre-filter noise so we don't waste tokens on items the regex catches.
    real, noise = filter_noise(unclassified)
    pre_ignored = [
        {"original": n, "name": None, "category": "ignore", "parent": None}
        for n in noise
    ]

    # Run remaining real-candidates through Claude in batches.
    existing_str = "\n".join(f"- {b['name']} ({b['category']})" for b in existing[:20])
    batches = [real[i:i + _BATCH_SIZE] for i in range(0, len(real), _BATCH_SIZE)]

    all_brands: list[dict] = []
    total_input = 0
    total_output = 0
    failed_batches = 0
    for idx, batch in enumerate(batches, 1):
        try:
            brands, usage = await _classify_one_batch(
                domain, site_brand, batch,
                existing_str, domain_context, model, anthropic_api_key,
            )
            all_brands.extend(brands)
            total_input += usage.get("input_tokens", 0) or 0
            total_output += usage.get("output_tokens", 0) or 0
            logger.info(
                f"Brand cleanup batch {idx}/{len(batches)}: "
                f"{len(batch)} in → {len([b for b in brands if b.get('category') != 'ignore'])} real, "
                f"{len([b for b in brands if b.get('category') == 'ignore'])} ignored"
            )
        except Exception:
            # One failed batch shouldn't kill the entire cleanup — items
            # in that batch stay unclassified for next run. Log + continue.
            logger.exception(f"Brand cleanup batch {idx}/{len(batches)} failed")
            failed_batches += 1

    # Pre-ignored items go through verbatim (no Claude needed).
    all_brands.extend(pre_ignored)

    duration = int((time.time() - start) * 1000)
    real_count = len([b for b in all_brands if b.get("category") != "ignore"])
    ign_count = len([b for b in all_brands if b.get("category") == "ignore"])
    logger.info(
        f"Brand cleanup: {len(unclassified)} candidates → "
        f"{real_count} real, {ign_count} ignored "
        f"({len(noise)} via regex, {ign_count - len(noise)} via Claude), "
        f"{len(batches)} batches ({failed_batches} failed), {duration}ms"
    )

    return {
        "brands": all_brands,
        "model": model,
        "batches": len(batches),
        "failed_batches": failed_batches,
        "pre_filtered_noise": len(noise),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "duration_ms": duration,
    }
