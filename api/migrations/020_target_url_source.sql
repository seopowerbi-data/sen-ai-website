-- 020_target_url_source.sql
--
-- Audit column for how scan_content_items.target_url was set.
--
-- Foundation for the long-term sitemap-index + auto-suggest pipeline (Phase D,
-- Pilier 3 in project_roadmap_content_port.md). We store the *source* of each
-- target_url so we can later measure:
--   - % of items where the system auto-suggested correctly (KPI for Pilier 3)
--   - % of items where the user had to override (signals where suggestions fail)
--
-- Values :
--   'scan_result'  — set automatically from the scan pipeline (e.g. opportunity
--                    already had a target URL from the user-owned scan).
--   'pending_user' — competitor scan : target_url left NULL on purpose, user
--                    must pick a URL on their own site in the validation page
--                    before generation can run.
--   'user_input'   — user set the URL manually (after a 'pending_user' or to
--                    correct an auto-suggestion).
--   'auto_suggest' — RESERVED for Phase D : web_search / sitemap-index based
--                    suggestion. Unused today, exists in the enum so the UI
--                    can render the placeholder slot ("Coming soon" confidence).

ALTER TABLE scan_content_items
    ADD COLUMN IF NOT EXISTS target_url_source TEXT DEFAULT NULL;

COMMENT ON COLUMN scan_content_items.target_url_source IS
    'Provenance of target_url: scan_result | pending_user | user_input | auto_suggest. '
    'See migration 020 for semantics.';

-- Backfill: any existing row with a non-null target_url was set manually during
-- Phase B smoke tests, mark them 'user_input'. Rows with target_url IS NULL
-- get 'pending_user' so they show up in the new "Needs URL" Kanban filter.
UPDATE scan_content_items
SET target_url_source = CASE
    WHEN target_url IS NOT NULL AND target_url <> '' THEN 'user_input'
    ELSE 'pending_user'
END
WHERE target_url_source IS NULL;
