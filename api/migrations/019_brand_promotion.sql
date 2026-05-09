-- 019_brand_promotion.sql
--
-- Brand promotion mechanism for content generation.
--
-- The user wants : when generating FAQ / Article from an opportunity (especially
-- on a competitor scan), the output must promote the user's OWN brands
-- (Avène, Aderma, Ducray) — never the competitor that was scanned.
--
-- Resolution chain at content-gen time, highest priority first :
--   1. scans.promotion_brand_ids               (per-scan override)
--   2. scan_brand_classifications.classification = 'my_brand' for the scan
--   3. clients.primary_brand_ids               (cross-scan workspace default)
--   4. raise PromotionUnsetError               (UI prompts user to set defaults)
--
-- All three new columns are arrays of UUIDs referencing client_brands.id.
-- promoted_brand_ids on scan_content_items is an audit trail (which brands
-- were instructed to be promoted when this content was actually generated).

ALTER TABLE clients
    ADD COLUMN IF NOT EXISTS primary_brand_ids uuid[] DEFAULT NULL;

COMMENT ON COLUMN clients.primary_brand_ids IS
    'Ordered list of client_brands.id — first is the lead brand. '
    'Cross-scan persistent default for content promotion.';

ALTER TABLE scans
    ADD COLUMN IF NOT EXISTS promotion_brand_ids uuid[] DEFAULT NULL;

COMMENT ON COLUMN scans.promotion_brand_ids IS
    'Per-scan override of clients.primary_brand_ids. NULL = inherit from client.';

ALTER TABLE scan_content_items
    ADD COLUMN IF NOT EXISTS promoted_brand_ids uuid[] DEFAULT NULL;

COMMENT ON COLUMN scan_content_items.promoted_brand_ids IS
    'Audit trail: brands the system was instructed to promote when this '
    'content was generated. Useful for QA + analytics by brand.';

-- Backfill clients.primary_brand_ids from existing my_brand classifications.
-- For each client, take the union of all client_brands that have been classified
-- 'my_brand' in at least one of their scans. Order = deterministic (by name).
UPDATE clients c
SET primary_brand_ids = sub.ids
FROM (
    SELECT cb.client_id,
           array_agg(cb.id ORDER BY cb.name) AS ids
    FROM client_brands cb
    JOIN scan_brand_classifications sbc ON sbc.brand_id = cb.id
    WHERE sbc.classification = 'my_brand'
    GROUP BY cb.client_id
) sub
WHERE c.id = sub.client_id
  AND (c.primary_brand_ids IS NULL OR array_length(c.primary_brand_ids, 1) IS NULL);
