-- 037_scan_question_judgments.sql
--
-- Sprint J (project_phase_judge_and_entities.md) — LLM-as-judge per-question
-- scoring. Converts intention_cachee + signal_positif + signal_negatif from
-- decorative UI tooltips into actionable metrics by having Haiku read each
-- LLM response against the per-question grille and emit structured signals.
--
-- One row per scan_llm_result (= per (question, provider) pair). The judge
-- runs AFTER run_llm_tests via a dedicated job (worker/handlers/judge_question_responses.py)
-- chained alongside classify_question_intent in the post-scan DAG.
--
-- ## Foot-gun #2 (memo) — 2-pass judge to avoid target-brand bias
--
-- The judge is NOT told who the target brand is. Otherwise it would
-- over-mark positive every time the brand is cited regardless of envelope
-- quality. `est_cible` resolution happens downstream in code, never in the
-- LLM prompt.
--
-- ## Foot-gun #3 — intent_addressed requires evidence span
--
-- `intention_cachee` is LLM-generated free-form French. The judge can
-- hallucinate "yes" by default. Enforced contract: intent_addressed=true
-- REQUIRES a non-empty intent_evidence (literal span from response). Empty
-- evidence → row stored with intent_addressed=false even if Haiku said true.

CREATE TABLE IF NOT EXISTS scan_question_judgments (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- One judgment per (question, provider) tuple. UNIQUE so re-runs of the
    -- handler are idempotent : we skip rows that already have a judgment.
    scan_llm_result_id          UUID NOT NULL UNIQUE
                                REFERENCES scan_llm_results(id) ON DELETE CASCADE,

    -- Denormalized FKs for fast aggregation queries (avoid 3-way JOIN to
    -- scan_llm_results just to filter by scan).
    scan_id                     UUID NOT NULL
                                REFERENCES scans(id) ON DELETE CASCADE,
    question_id                 UUID
                                REFERENCES scan_questions(id) ON DELETE SET NULL,

    -- Core signals (PDF SEO LLM framework, per-question grille).
    positive_signal_hit         BOOLEAN NOT NULL,
    positive_signal_evidence    TEXT,
    negative_signal_hit         BOOLEAN NOT NULL,
    negative_signal_evidence    TEXT,
    intent_addressed            BOOLEAN NOT NULL,
    intent_evidence             TEXT,

    -- Citation quality (lead/alternative/footnote/absent).
    citation_quality            VARCHAR(20),

    -- RAPP Positivity per-response, 0-5. NULL when the judge couldn't score
    -- (response too short, refusal, etc.).
    enveloppement_score         SMALLINT,

    -- Provenance / cost accounting.
    judge_model                 VARCHAR(80),
    input_tokens                INT,
    output_tokens               INT,
    duration_ms                 INT,

    created_at                  TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT enveloppement_score_range
        CHECK (enveloppement_score IS NULL OR (enveloppement_score >= 0 AND enveloppement_score <= 5)),
    CONSTRAINT citation_quality_enum
        CHECK (citation_quality IS NULL OR citation_quality IN ('lead', 'alternative', 'footnote', 'absent'))
);

-- Aggregations the UI + revised dashboards will query:
--   "how many positive_signal_hit for this scan ?"
--   "positive hit rate per persona (filter on question)"
--   "intent coverage rate"
CREATE INDEX IF NOT EXISTS idx_sqj_scan_id
    ON scan_question_judgments(scan_id);

CREATE INDEX IF NOT EXISTS idx_sqj_question_id
    ON scan_question_judgments(question_id);

-- Partial indexes for the most common filters in Sprint M dashboards. Both
-- conditions are highly selective (~30-50% hit rate typical) so the index
-- pays off vs a full scan.
CREATE INDEX IF NOT EXISTS idx_sqj_scan_positive_hits
    ON scan_question_judgments(scan_id)
    WHERE positive_signal_hit;

CREATE INDEX IF NOT EXISTS idx_sqj_scan_negative_hits
    ON scan_question_judgments(scan_id)
    WHERE negative_signal_hit;

CREATE INDEX IF NOT EXISTS idx_sqj_scan_intent_addressed
    ON scan_question_judgments(scan_id)
    WHERE intent_addressed;

COMMENT ON TABLE scan_question_judgments IS
    'Sprint J: LLM-as-judge per-(question, provider) signals. Reads response_text '
    'against scan_questions.signal_positif/signal_negatif/intention_cachee and '
    'emits structured bools + evidence spans. One row per scan_llm_results.id. '
    'Consumed by Sprint M composite scores + UI chips. '
    'See project_phase_judge_and_entities.md.';
