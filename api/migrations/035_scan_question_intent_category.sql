-- 035_scan_question_intent_category.sql
--
-- Phase B Tier A — Intent classifier foundation.
--
-- Adds a categorical intent field to scan_questions so the opportunity
-- scorer (worker/handlers/generate_opportunities.py) can drop netlinking
-- opportunities on questions whose intent makes brand placement
-- editorially inappropriate (safety, side effects, contre-indications).
--
-- Empirical proof (2026-05-18, scan b0ea6068, Pierre Fabre, brand=Avène) :
--   APPROVED  "acide hyaluronique + rétinol routine"            -> 7 mentions
--   SKIP      "rétinol pèle/rougit dois-je arrêter"             -> 0 mentions
-- Same client, same persona, same topic, same model. Only the question
-- intent changed. The LOW_QUALITY_SKIP is editorially valid — the brand
-- cannot be naturally woven into a "should I stop" answer. The fix is
-- upstream: don't generate the opportunity in the first place.
--
-- ## intent_category values (Haiku-classified, multi-lingual)
--
--   promotional_fit       - brand placement fits naturally
--   informational_neutral - brand placement possible but low fit
--   safety_warning        - safety question, brand placement awkward
--   side_effects          - adverse-effects question, brand placement awkward
--   contre_indication     - contre-indications question, brand placement awkward
--   complaint_sav         - SAV / complaint, brand placement inappropriate
--   other                 - everything else (rare)
--
-- NULL = not yet classified. Existing rows stay NULL so the migration
-- never blocks the rollout; the classifier runs only on new scans and
-- via the explicit backfill script (worker/scripts/backfill_question_intent.py).
--
-- The opportunity scorer treats NULL as "promotional_fit" (= legacy
-- behavior, generate netlinking opportunity normally). Once the
-- backfill runs, future opportunity re-computes benefit from the
-- richer signal.
--
-- See project_phase_b_intent_classifier_gap.md for the architecture
-- discussion + 3-tier intervention plan.

ALTER TABLE scan_questions
    ADD COLUMN IF NOT EXISTS intent_category VARCHAR(40);

COMMENT ON COLUMN scan_questions.intent_category IS
    'Phase B intent classifier output. Drives opportunity scoring: '
    'safety_warning / side_effects / contre_indication block netlinking '
    'opportunity creation in worker/handlers/generate_opportunities.py. '
    'NULL = not yet classified (treated as promotional_fit by scorer for '
    'backward compat). Multi-lingual Haiku classification. '
    'See migration 035 + project_phase_b_intent_classifier_gap.';
