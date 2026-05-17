-- 034_web_search_queries_capture.sql
--
-- Phase C.1.5 — Fan-out extractor data foundation.
--
-- Adds two JSONB fields to support cross-provider fan-out capture +
-- post-hoc extraction caching. Replaces the YTG 150-char truncate hack
-- in worker/handlers/generate_article.py (commit `1605d05`) with proper
-- ground-truth fan-out queries from LLM responses.
--
-- ## scan_llm_results.web_search_queries
--
-- Per-row (= per scan × question × provider) list of search queries the
-- LLM ACTUALLY issued when generating its response. Populated by the
-- capture logic in `worker/handlers/run_llm_tests.py` per provider :
--
--   gemini  : response.candidates[0].grounding_metadata.web_search_queries
--   openai  : output[].web_search_call.query (one per tool call, aggregated)
--   claude  : content[].tool_use.input.query (where name=='web_search')
--   perplexity : search_metadata.search_queries (or reconstructed)
--
-- Empty list = provider didn't emit search queries (Claude without
-- web_search tool, older SDK version that doesn't expose grounding, ...).
-- Empty is safe — the fan_out_extractor falls back to post-hoc Haiku
-- extraction from response_text (B2 fallback in C.1.5 architecture).
--
-- Cross-provider aggregation happens at extraction time in
-- `worker/services/fan_out_extractor.py:aggregate_fanouts_from_scan` :
-- consensus fan-outs (appearing in ≥2 providers) get the strongest
-- ranking score for primary selection.
--
-- ## scan_questions.fan_out_queries
--
-- Per-question cache of the EXTRACTED + SELECTED fan-outs (post
-- aggregation + ranking). Populated by `fan_out_extractor.extract`
-- on first request, then served from cache on subsequent calls. Avoids
-- re-running Haiku tiebreak / aggregation logic for the same question
-- across multiple article gen attempts.
--
-- Shape : list[str] of fan-out queries ordered by primary-first.
--   [0]  = primary (sent to YTG)
--   [1+] = additional fan-outs (passed to content gen for coverage + FAQ Q seeds)
--
-- Empty list = first extraction attempt failed OR not yet attempted.
-- Invalidate by SET fan_out_queries = '[]'::jsonb if re-extraction needed.
--
-- See `project_phase_c1_article_handler.md` section C.1.5 for the full
-- architecture decision tree (why B1+B2 hybrid + ranking algo over A_arch
-- multi-guide YTG refactor).

ALTER TABLE scan_llm_results
    ADD COLUMN IF NOT EXISTS web_search_queries JSONB
    NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE scan_questions
    ADD COLUMN IF NOT EXISTS fan_out_queries JSONB
    NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN scan_llm_results.web_search_queries IS
    'Per-row list of search queries the LLM actually issued during response '
    'generation. Captured per-provider in run_llm_tests.py from '
    'grounding_metadata (Gemini) / web_search_call (OpenAI) / tool_use '
    'blocks (Claude). Empty = provider did not emit queries. '
    'See migration 034 + C.1.5 fan_out_extractor.';

COMMENT ON COLUMN scan_questions.fan_out_queries IS
    'Cached list of selected fan-out queries for this question, '
    'ordered with primary at index [0]. Populated lazily on first article '
    'gen by fan_out_extractor.extract — aggregates web_search_queries '
    'across providers, applies ranking algo, selects primary. Empty = '
    'not yet extracted OR extraction failed. SET to [] to invalidate cache.';
