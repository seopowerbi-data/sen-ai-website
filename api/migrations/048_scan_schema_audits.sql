-- 048_scan_schema_audits.sql
--
-- Sprint 6 (Schema.org / JSON-LD generator) - feature #5 from
-- project_10_action_features.md. For every page the LLMs already cite from
-- the user's own site, we :
--   1. extract existing <script type="application/ld+json"> blocks
--   2. detect the page type (homepage / article / product / faq / about)
--   3. generate the missing schema.org blocks (Organization, WebSite,
--      BreadcrumbList, Article, Product, FAQPage) using brand_brief data
--      and on-page microdata fallbacks
--   4. validate each block against the schema.org required-property spec
--      (local validator - no Google API dependency)
--
-- Why a separate table from scan_page_audits (Sprint 5) :
--   - Different cadence. The Princeton audit is content quality ; the schema
--     audit is structured data. A page can pass one and fail the other.
--   - The generated_blocks JSONB is much heavier (full JSON-LD per block).
--     Coupling to scan_page_audits would bloat that row.
--   - The user may run one without the other.
--
-- existing_schemas JSONB shape :
--   [
--     { "type": "Organization", "valid": true,  "missing": [],
--       "raw": {...full block...} },
--     { "type": "FAQPage",      "valid": false, "missing": ["mainEntity"],
--       "raw": {...} }
--   ]
--
-- generated_blocks JSONB shape :
--   {
--     "Organization":    { "@context": "https://schema.org", "@type": "Organization", ... },
--     "BreadcrumbList":  { ... },
--     "FAQPage":         { ... }
--   }
--
-- missing_schemas TEXT[] :
--   Schema types that *should* be on this page given its detected type but
--   aren't (e.g. an article page missing "Article" + "BreadcrumbList").
--
-- schema_score (0-100) composite :
--   25 pts  Organization present and valid
--   25 pts  page-type-appropriate primary schema present and valid
--           (Article on /blog, FAQPage on /faq, Product on /product, …)
--   20 pts  BreadcrumbList present and valid (when URL has >1 path segment)
--   20 pts  % of existing blocks that parse and pass required-property check
--   10 pts  WebSite present (homepage only ; auto-credited elsewhere)

CREATE TABLE IF NOT EXISTS scan_schema_audits (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id             UUID NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    url                 TEXT NOT NULL,
    title               TEXT,
    page_type           TEXT,        -- homepage | article | product | faq | about | other
    fetched_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    fetch_status        INTEGER,
    fetch_error         TEXT,
    existing_schemas    JSONB NOT NULL DEFAULT '[]'::jsonb,
    missing_schemas     TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    generated_blocks    JSONB NOT NULL DEFAULT '{}'::jsonb,
    schema_score        INTEGER,
    citation_count      INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (scan_id, url),
    CONSTRAINT schema_score_range CHECK (schema_score IS NULL OR (schema_score >= 0 AND schema_score <= 100))
);

CREATE INDEX IF NOT EXISTS idx_ssa_scan        ON scan_schema_audits(scan_id);
CREATE INDEX IF NOT EXISTS idx_ssa_scan_score  ON scan_schema_audits(scan_id, schema_score ASC NULLS LAST);

COMMENT ON TABLE scan_schema_audits IS
    'Sprint 6 schema.org / JSON-LD audit + generator. One row per (scan, url) '
    'where url is a page of the user''s own site cited by at least one LLM. '
    'No LLM cost - heuristic page-type detection + template fills from '
    'brand_brief and on-page microdata. See worker/handlers/audit_scan_schemas.py '
    'and project_10_action_features.md #5.';
