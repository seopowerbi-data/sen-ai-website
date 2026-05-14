-- 025_client_brand_pages.sql
--
-- Phase D — Sitemap-index per primary brand, semantic target_url matching.
-- See ~/.claude/plans/sen-ai-phase-d-sitemap-index.md for the full design.
--
-- One row per discovered URL on a client_brand's site. Lifecycle :
--
--   pending_fetch -> fetched -> embedded
--                \         \
--                 \         +-> error  (after fetch_retry_count >= 3)
--                  \
--                   +-> error  (sitemap-level failure)
--
--   embedded <-> gone        (toggled each crawl based on sitemap diff;
--                             a 'gone' row that reappears flips back to
--                             'embedded' and preserves its embedding)
--
--   gone -> HARD DELETE      (when now() - gone_since > 30 days, by
--                             purge_stale_pages)
--
-- Embeddings stored as JSONB (list[float], length 1536). Migration to
-- pgvector becomes a single ALTER TABLE later; for v1 numpy cosine over
-- ~3-5k vectors per brand is well under 10ms and avoids the pgvector
-- ops/extension dance.
--
-- internal_inlink_count = # of OTHER pages WITHIN this brand's own sitemap
-- that link to this URL. Used as an authority signal in the matcher score
-- (cosine * (1 + 0.15 * log10(1+inlinks))). Architectural intent = hub pages.

CREATE TABLE IF NOT EXISTS client_brand_pages (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_brand_id       UUID NOT NULL REFERENCES client_brands(id) ON DELETE CASCADE,

    -- Identity
    url                   TEXT NOT NULL,
    url_canonical         TEXT,        -- from <link rel="canonical">, NULL when same as url

    -- Extracted page metadata (filled by fetch_brand_pages handler, Day 2)
    title                 TEXT,
    meta_description      TEXT,
    h1                    TEXT,
    body_excerpt          TEXT,        -- first 300 words from <main>/<article>/<body>, nav/footer/aside stripped
    lang                  TEXT,        -- from <html lang>, e.g. 'fr', 'en'

    -- Sitemap signals
    lastmod               TIMESTAMP,   -- from sitemap, nullable

    -- Change detection
    content_hash          TEXT,        -- sha256(lower(title|meta|h1|body_excerpt)) — short-circuits useless re-embeds

    -- Authority signal (filled in single post-fetch pass, Day 3)
    internal_inlink_count INTEGER NOT NULL DEFAULT 0,

    -- Embedding (filled by embed_brand_pages handler, Day 3)
    embedding             JSONB,       -- list[float] length 1536, NULL while pending
    embedding_model       TEXT,        -- e.g. 'text-embedding-3-small' — audit + lazy migration trigger

    -- Lifecycle
    status                TEXT NOT NULL DEFAULT 'pending_fetch',
                                       -- pending_fetch | fetched | embedded | gone | error
    fetch_error           TEXT,        -- last error message (NULL on success)
    fetch_retry_count     INTEGER NOT NULL DEFAULT 0,
    http_status           INTEGER,     -- last HTTP status from page fetch

    -- Timestamps
    first_seen_at         TIMESTAMP NOT NULL DEFAULT NOW(),
    last_seen_at          TIMESTAMP NOT NULL DEFAULT NOW(),  -- bumped each time URL re-appears in sitemap
    last_crawled_at       TIMESTAMP,   -- last page-meta fetch attempt
    last_embedded_at      TIMESTAMP,   -- last successful embed
    gone_since            TIMESTAMP,   -- set when status flips to 'gone'; drives 30-day TTL purge

    CONSTRAINT uq_brand_url UNIQUE (client_brand_id, url)
);

CREATE INDEX IF NOT EXISTS idx_cbp_brand_status
    ON client_brand_pages (client_brand_id, status);

CREATE INDEX IF NOT EXISTS idx_cbp_gone_since
    ON client_brand_pages (gone_since)
    WHERE gone_since IS NOT NULL;

COMMENT ON TABLE client_brand_pages IS
    'Phase D sitemap index: one row per URL discovered on a primary brand domain. '
    'Stores extracted page meta + body excerpt + JSONB embedding + internal inlink count, '
    'used by sitemap_matcher to suggest target_url for FAQ items.';

COMMENT ON COLUMN client_brand_pages.status IS
    'Lifecycle: pending_fetch -> fetched -> embedded; embedded <-> gone (sitemap diff); error terminal. '
    '30-day TTL hard-delete on gone_since by purge_stale_pages.';

COMMENT ON COLUMN client_brand_pages.internal_inlink_count IS
    'Number of OTHER pages within this brand''s own sitemap that link to this URL. '
    'Authority signal in matcher score: cosine * (1 + 0.15 * log10(1 + count)).';

COMMENT ON COLUMN client_brand_pages.content_hash IS
    'sha256(lower(title|meta|h1|body_excerpt)). When unchanged on re-fetch we skip re-embed '
    '(avoids the lastmod-trap: many CMSes lie about lastmod).';
