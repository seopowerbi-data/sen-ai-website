-- 033_domain_classifications.sql
--
-- Phase C.1.4 — Global domain → site_type cache for the media picker.
--
-- Symmetric in spirit to what `dim_domain.csv` is on the seo-llm SharePoint
-- pipeline : a per-domain classification tag used to filter netlinking
-- candidates down to the buyable medias (Health & Beauty Media, Blog, News)
-- and exclude Brand sites, Government, Encyclopedia, Forum, E-commerce.
--
-- Schema decisions
--
--   - `domain` is the PK (lowercase, no scheme, no www, no path). Same
--     normalization rule as services.url_filter._normalize_domain. Means we
--     never store two entries for the same brand under www. vs bare form.
--
--   - `site_type` is a free TEXT (no PG enum) because we want to add new
--     categories without a migration — e.g. "Auto Media", "Finance Media"
--     for non-dermo verticals. The list of acceptable values lives in
--     worker/services/domain_classifier.py (SITE_CATEGORIES) and validated
--     at write time by the service.
--
--   - `model` records which model classified this domain (e.g. 'gemini-2.5-flash',
--     'gemini-import' for seo-llm CSV imports, 'manual' for user overrides).
--     Useful for audit + future re-classification when prompts evolve.
--
--   - `source` adds another audit dimension : 'gemini' (fresh classification),
--     'import_seollm' (one-shot PF migration), 'manual' (admin override).
--
--   - `metadata` JSONB for future per-classification context (e.g. sample
--     URL paths used for disambiguation, confidence score, vertical hints).
--
-- Global table (not per-client) : a Brand site is a Brand site regardless of
-- which client's scan surfaced it. Cross-client cache hit rate is the win —
-- every new client benefits from prior clients' classifications.
--
-- The accompanying one-shot import script populates this table from PF's
-- existing 3503 classified domains in seo-llm SharePoint cache. After import,
-- fresh classifications are added on-demand by services.domain_classifier
-- when media_picker encounters a previously-unseen domain.

CREATE TABLE IF NOT EXISTS domain_classifications (
    domain         TEXT PRIMARY KEY,
    site_type      TEXT NOT NULL,
    classified_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    model          TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'gemini',
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_domain_classifications_site_type
    ON domain_classifications(site_type);

COMMENT ON TABLE domain_classifications IS
    'Global domain → site_type cache. Sourced from seo-llm import (PF historical), '
    'Gemini classifier (on-demand for new domains), or manual overrides. Used by '
    'worker.services.media_picker to filter netlinking candidates to buyable medias only. '
    'See migration 033.';

COMMENT ON COLUMN domain_classifications.site_type IS
    'One of : Government, Medical Reference, News, Health & Beauty Media, Brand, '
    'E-commerce, Encyclopedia, Forum, Blog, Other. Validated at write time by the '
    'classifier service against SITE_CATEGORIES.';

COMMENT ON COLUMN domain_classifications.source IS
    'gemini | import_seollm | manual — audit trail for the origin of this classification.';
