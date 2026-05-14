-- 026_target_url_score_and_candidates.sql
--
-- Phase D — Surface sitemap-matcher confidence + top-3 candidates on each content item.
--
-- - target_url_score : float in roughly [0, 1.5]. The matcher's final score
--   (cosine * authority_boost * gamme_boost). NULL = no score available
--   (item predates Phase D or matcher fell back to web_search).
--
-- - target_url_candidates : JSONB array of top-3 picks from the sitemap matcher,
--   shape : [{"url": str, "title": str, "score": float}]. Empty array = no candidates
--   (matcher didn't run, or fell through to web_search fallback). Drives the
--   top-3 picker UX in the validation page (Day 6).
--
-- - target_url_source now also accepts 'sitemap_index' (set by materialize when the
--   sitemap matcher's top-1 clears SITEMAP_THRESHOLD). The column is plain TEXT
--   with no CHECK constraint, so no schema change is needed to add the value —
--   we update the column comment as the canonical reference.
--
-- The trio is mutually consistent : when target_url_source = 'sitemap_index' then
-- target_url_score IS NOT NULL and target_url_candidates is non-empty. The API
-- serializer exposes all three so the UI can render either the new top-3 picker
-- or the legacy single-URL display without a regression.

ALTER TABLE scan_content_items
    ADD COLUMN IF NOT EXISTS target_url_score FLOAT;

ALTER TABLE scan_content_items
    ADD COLUMN IF NOT EXISTS target_url_candidates JSONB
    NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN scan_content_items.target_url_score IS
    'Sitemap matcher final score for target_url (cosine * authority_boost * gamme_boost). '
    'NULL when no sitemap match available. See migration 026.';

COMMENT ON COLUMN scan_content_items.target_url_candidates IS
    'Top-3 sitemap matcher picks: [{"url","title","score"}]. Drives the top-3 picker UX. '
    'Empty array when matcher did not run or fell back to web_search. See migration 026.';

COMMENT ON COLUMN scan_content_items.target_url_source IS
    'Provenance of target_url: scan_result | pending_user | user_input | auto_suggest | sitemap_index. '
    'sitemap_index added in migration 026 (Phase D). See migrations 020 and 026 for semantics.';
