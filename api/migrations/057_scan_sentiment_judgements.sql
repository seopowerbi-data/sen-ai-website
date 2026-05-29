-- 057_scan_sentiment_judgements.sql
--
-- Sentiment Judge - per-mention overturn / confirm layer on top of the
-- BrandAnalyzer's sentiment field. The BrandAnalyzer (Gemini, single-shot
-- per LLM response) misclassifies negation / disclaimer / usage-clarification
-- patterns as `négatif` because its prompt is too short and reads each
-- mention out of context. The Judge re-reads each negative mention with
-- Haiku and decides whether to confirm, overturn, or hedge.
--
-- One row per JUDGED brand_mention (negatives are the noisy bucket - we
-- don't judge positives or neutrals in v1, they're rarely false-positive).
-- The triplet (slr_id, mention_index, contexte_hash) is the identity key :
-- mention_index identifies the position in the JSONB array,
-- contexte_hash detects when the same slot was overwritten by a different
-- mention (e.g. after a re-run). We never UPDATE a judgement in place ;
-- a re-judge inserts a new row and the consumer reads the latest by
-- judge_run_at.
--
-- judge_verdict values :
--   confirm  : the négatif label IS appropriate (real negative sentiment)
--   overturn : the négatif label is wrong - corrected_sentiment carries
--              the true label (positif / neutre)
--   hedge    : ambiguous contexte, confidence is low. Consumers treat as
--              neutre for severity computation but flag as "verify manually"
--
-- Downstream consumers (build_crisis_radar.py, results.astro Overview chip,
-- future PR / sentiment-driven features) LEFT JOIN this table and prefer
-- the judged label when present, fall back to the raw label otherwise.
-- v1 reads only the latest judgement per (slr_id, mention_index).

CREATE TABLE IF NOT EXISTS scan_sentiment_judgements (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                UUID NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    slr_id                 UUID NOT NULL REFERENCES scan_llm_results(id) ON DELETE CASCADE,
    mention_index          INTEGER NOT NULL,
    brand_name             TEXT NOT NULL,
    contexte_hash          TEXT NOT NULL,
    raw_sentiment          TEXT NOT NULL,
    raw_justification      TEXT,
    judge_verdict          TEXT NOT NULL,
    judged_sentiment       TEXT,
    judge_reasoning        TEXT,
    judge_model            TEXT NOT NULL DEFAULT 'claude-haiku-4-5',
    judge_cost_usd         NUMERIC(10,6),
    judge_run_at           TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT ssj_verdict_values CHECK (
      judge_verdict IN ('confirm', 'overturn', 'hedge')
    ),
    CONSTRAINT ssj_sentiment_values CHECK (
      judged_sentiment IS NULL
      OR judged_sentiment IN ('positif', 'négatif', 'neutre', 'positive', 'negative', 'neutral')
    )
);

CREATE INDEX IF NOT EXISTS idx_ssj_scan      ON scan_sentiment_judgements(scan_id);
CREATE INDEX IF NOT EXISTS idx_ssj_slr       ON scan_sentiment_judgements(slr_id);
CREATE INDEX IF NOT EXISTS idx_ssj_slr_index ON scan_sentiment_judgements(slr_id, mention_index, judge_run_at DESC);
CREATE INDEX IF NOT EXISTS idx_ssj_verdict   ON scan_sentiment_judgements(scan_id, judge_verdict);

COMMENT ON TABLE scan_sentiment_judgements IS
    'Sentiment Judge - per-mention overturn layer on top of BrandAnalyzer. '
    'Re-reads each `négatif` mention with Claude Haiku 4.5 and either '
    'confirms, overturns, or hedges. Downstream consumers prefer judged_sentiment '
    'when verdict=overturn, fall back to raw brand_mentions[].sentiment otherwise. '
    'See worker/handlers/judge_sentiment.py + project_backlog_sentiment_quality.md.';

COMMENT ON COLUMN scan_sentiment_judgements.mention_index IS
    'Position (0-based) in scan_llm_results.brand_mentions JSONB array. '
    'Identifies which mention this judgement applies to within the row.';

COMMENT ON COLUMN scan_sentiment_judgements.contexte_hash IS
    'SHA-256 of the original contexte string. Detects when a re-run '
    'overwrote brand_mentions[i] with a different mention - in that case '
    'the consumer ignores stale judgements with non-matching hashes.';
