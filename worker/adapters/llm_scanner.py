"""LLM Scanner — uses seo-llm LLMClient for Gemini, but bypasses it for OpenAI.

Why bypass for OpenAI : the seo_llm LLMClient retries 5 times with 4-60s exponential
backoff on any APIError (including transient HTTP 500s from web_search). A burst of
5xx can spend 4+8+16+32+60 ≈ 120 s purely on backoff before failing. For our SaaS
scan_test path (where users sit watching the progress bar) we want fail-fast on
persistent 5xx and a more aggressive web_search config. We keep LLMClient for Gemini
because grounding works well there with no equivalent issue.

Quick wins applied vs LLMClient OpenAI path :
  - max_retries=2 on the openai client (vs 5) — caps backoff at ~16s for transient 5xx
  - timeout=120 (vs 300) — fail faster on hung calls
  - tools=[{"type":"web_search_preview","search_context_size":"low",...}] — much faster
    than the default web_search (fewer sub-searches, less context fetched per query)
  - user_location passed in tool config — better-grounded URLs for FR scans
"""

import logging
import time

import openai

from seo_llm.src.llm_client import LLMClient
from seo_llm.src.citation_extractor import CitationExtractor
from seo_llm.src.brand_analyzer import BrandAnalyzer
from seo_llm.src.config import get_llm_test_prompt
from seo_llm.src.api_pricing import calculate_cost

logger = logging.getLogger(__name__)

# OpenAI client tuning — see module docstring for rationale
OPENAI_DIRECT_TIMEOUT = 120  # seconds per call (vs LLMClient's 300)
OPENAI_DIRECT_MAX_RETRIES = 2  # vs LLMClient's tenacity 5x (4-60s backoff)
OPENAI_DIRECT_MAX_OUTPUT_TOKENS = 8000  # match LLMClient default

# OpenAI web_search_preview tool config — search_context_size=low → fewer sub-searches
# = much faster scan tests. We don't need deep research per question; we need to know
# WHO the AI cites (a few SERP-ish URLs) more than WHAT it knows in depth.
_OPENAI_WEB_SEARCH_TOOL = {
    "type": "web_search_preview",
    "search_context_size": "low",
}


def create_llm_client(provider: str, api_key: str, model: str = None) -> LLMClient:
    """Create an LLMClient instance for Gemini (kept for backward compat — OpenAI now
    uses the direct path test_question_openai_direct)."""
    if provider == "openai":
        return LLMClient(provider="openai", api_key=api_key, model=model or "gpt-4.1-mini")
    elif provider == "gemini":
        return LLMClient(provider="gemini", api_key=api_key, model=model or "gemini-2.5-flash")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def format_persona_summary(persona: dict) -> str:
    """Format persona as context for the LLM test prompt."""
    profil = persona.get("profil_demographique", {})
    return (
        f"L'utilisateur est {persona.get('nom', 'un visiteur')}. "
        f"Âge: {profil.get('age', '?')}. "
        f"Profession: {profil.get('situation_professionnelle', '?')}. "
        f"Niveau d'expertise: {profil.get('niveau_expertise', '?')}."
    )


def test_question(question: str, persona: dict, llm_client: LLMClient,
                   target_domain: str, brand_analyzer: BrandAnalyzer = None) -> dict:
    """Test a question with an LLM, extract citations and analyze brand mentions.

    Args:
        question: The question text
        persona: Persona dict with profil_demographique
        llm_client: LLMClient instance (OpenAI or Gemini)
        target_domain: Domain to check in citations (user's site)
        brand_analyzer: Optional BrandAnalyzer instance for brand mention analysis
    """
    persona_summary = format_persona_summary(persona)
    prompt_template = get_llm_test_prompt(llm_client.provider)
    prompt = prompt_template.format(persona_summary=persona_summary, question=question)

    start = time.time()

    # 1. Generate LLM response with web search / grounding
    response = llm_client.generate(
        prompt,
        temperature=0.7,
        max_tokens=8000,
        agent_name="platform_scan",
        use_grounding=True,
    )
    duration_ms = int((time.time() - start) * 1000)

    # 2. Extract citations (seo-llm CitationExtractor)
    extractor = CitationExtractor(site_domain=target_domain)
    citations = extractor.extract_citations(
        response_text=response["text"],
        grounding_sources=response.get("grounding_sources", []),
        provider=llm_client.provider,
    )

    # Analyze citation results
    target_cited = any(c.get("est_site_cible") for c in citations)
    target_position = None
    competitor_domains = {}
    for i, c in enumerate(citations):
        if c.get("est_site_cible") and target_position is None:
            target_position = i + 1
        elif c.get("domaine") and not c.get("est_site_cible"):
            domain = c["domaine"]
            if domain not in ("google.com", "youtube.com"):
                competitor_domains[domain] = competitor_domains.get(domain, 0) + 1

    # 3. Brand mention analysis (seo-llm BrandAnalyzer)
    brand_mentions = []
    brand_analysis = {}
    if brand_analyzer:
        try:
            brand_result = brand_analyzer.analyze_response(response["text"], question)
            if brand_result:
                brand_mentions = brand_result.get("brand_mentions", [])
                brand_analysis = brand_result.get("brand_analyse", {})
        except Exception as e:
            logger.warning(f"BrandAnalyzer failed: {e}")

    return {
        "provider": llm_client.provider,
        "model": response.get("model", llm_client.model),
        "response_text": response["text"],
        "citations": citations,
        "target_cited": target_cited,
        "target_position": target_position,
        "total_citations": len(citations),
        "competitor_domains": competitor_domains,
        "brand_mentions": brand_mentions,
        "brand_analysis": brand_analysis,
        "duration_ms": duration_ms,
        "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
        "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
    }


def _user_location_param(country_code: str | None) -> dict:
    """Build OpenAI web_search user_location dict from a 2-letter country code."""
    cc = (country_code or "FR").strip().upper()[:2] or "FR"
    return {"type": "approximate", "country": cc}


def test_question_openai_direct(question: str, persona: dict, target_domain: str,
                                api_key: str, model: str,
                                brand_analyzer: BrandAnalyzer = None,
                                country: str | None = "FR") -> dict:
    """Optimized OpenAI scan_test path — bypasses seo_llm.LLMClient for latency wins.

    See module docstring for the rationale. Output shape mirrors test_question() so
    the handler doesn't need to branch on which path produced the result.
    """
    persona_summary = format_persona_summary(persona)
    prompt_template = get_llm_test_prompt("openai")
    prompt = prompt_template.format(persona_summary=persona_summary, question=question)

    client = openai.OpenAI(
        api_key=api_key,
        timeout=OPENAI_DIRECT_TIMEOUT,
        max_retries=OPENAI_DIRECT_MAX_RETRIES,
    )
    web_search_tool = dict(_OPENAI_WEB_SEARCH_TOOL)
    web_search_tool["user_location"] = _user_location_param(country)

    start = time.time()
    response = client.responses.create(
        model=model,
        tools=[web_search_tool],
        input=prompt,
        temperature=0.7,
        max_output_tokens=OPENAI_DIRECT_MAX_OUTPUT_TOKENS,
    )
    duration_ms = int((time.time() - start) * 1000)

    response_text = response.output_text or ""

    # Extract grounding URLs (mirror LLMClient._generate_openai_responses logic)
    grounding_sources = []
    seen_urls = set()
    for item in (response.output or []):
        if hasattr(item, "type") and item.type == "message":
            for content_block in getattr(item, "content", []):
                for ann in getattr(content_block, "annotations", []):
                    if hasattr(ann, "type") and ann.type == "url_citation":
                        url = getattr(ann, "url", "")
                        title = getattr(ann, "title", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            grounding_sources.append({"url": url, "title": title})

    extractor = CitationExtractor(site_domain=target_domain)
    citations = extractor.extract_citations(
        response_text=response_text,
        grounding_sources=grounding_sources,
        provider="openai",
    )

    target_cited = any(c.get("est_site_cible") for c in citations)
    target_position = None
    competitor_domains: dict[str, int] = {}
    for i, c in enumerate(citations):
        if c.get("est_site_cible") and target_position is None:
            target_position = i + 1
        elif c.get("domaine") and not c.get("est_site_cible"):
            domain = c["domaine"]
            if domain not in ("google.com", "youtube.com"):
                competitor_domains[domain] = competitor_domains.get(domain, 0) + 1

    brand_mentions = []
    brand_analysis = {}
    if brand_analyzer:
        try:
            brand_result = brand_analyzer.analyze_response(response_text, question)
            if brand_result:
                brand_mentions = brand_result.get("brand_mentions", [])
                brand_analysis = brand_result.get("brand_analyse", {})
        except Exception as e:
            logger.warning(f"BrandAnalyzer failed: {e}")

    input_tokens = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
    output_tokens = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0

    return {
        "provider": "openai",
        "model": model,
        "response_text": response_text,
        "citations": citations,
        "target_cited": target_cited,
        "target_position": target_position,
        "total_citations": len(citations),
        "competitor_domains": competitor_domains,
        "brand_mentions": brand_mentions,
        "brand_analysis": brand_analysis,
        "duration_ms": duration_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
