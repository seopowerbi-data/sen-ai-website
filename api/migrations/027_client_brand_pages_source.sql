-- 027_client_brand_pages_source.sql
--
-- Phase D — Manual URL escape hatch for brands without a sitemap (or with
-- an incomplete one). User can add a URL directly from Settings (Day 5
-- UX); the row enters the normal fetch -> embed pipeline, but is exempt
-- from the sitemap-diff "mark_gone" branch since it was never in the
-- sitemap to begin with.
--
-- source values :
--   'sitemap'  -- discovered by crawl_brand_sitemap (default)
--   'manual'   -- user-added via Settings UI
--
-- Filter rules :
--   crawl_brand_sitemap.mark_gone branch  -> WHERE source = 'sitemap'
--   crawl_brand_sitemap.bump branch       -> agnostic (manual rows that
--                                            happen to appear in the
--                                            sitemap get their last_seen_at
--                                            bumped, but source stays
--                                            'manual' to preserve user intent
--                                            and keep hard-delete semantics)
--   Manual delete                          -> hard DELETE (not soft 'gone')

ALTER TABLE client_brand_pages
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'sitemap';

COMMENT ON COLUMN client_brand_pages.source IS
    'Discovery source: sitemap | manual. Manual rows are exempt from the '
    'sitemap-diff mark_gone branch and are hard-deleted on user remove. '
    'See migration 027.';
