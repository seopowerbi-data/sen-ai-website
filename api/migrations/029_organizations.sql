-- 029_organizations.sql
--
-- Phase E.C — Agency layer / Multi-tenancy foundation.
--
-- Introduces an Organization concept above Client. Each Client lives in
-- exactly one Organization. Users belong to Organizations and within an
-- Organization have per-Client roles. This unlocks the agency use case
-- (one user managing N brands across M clients via one org) while keeping
-- the existing single-client flow as a degenerate case (1 user, 1 client,
-- 1 personal org).
--
-- Migration safety :
--   1. New tables are additive. Existing routers keep using `user_clients`
--      as the source of truth for now. The new `services/access.py`
--      reads BOTH tables (orgs first, user_clients as fallback) so we
--      can refactor endpoints incrementally.
--   2. clients.organization_id is nullable initially. Backfill populates
--      it for every existing client. New clients should set it explicitly.
--   3. Backfill is idempotent (skips clients that already have an org).
--
-- Schema summary :
--   organizations         — the org entity (agency or personal workspace)
--   organization_users    — who is a member of an org + org-level role
--   org_user_clients      — per-client role within an org (granular access)
--   clients.organization_id — each client points to its owning org

CREATE TABLE IF NOT EXISTS organizations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,
    slug          TEXT UNIQUE,
    is_personal   BOOLEAN NOT NULL DEFAULT FALSE,  -- single-client backfill flag
    branding      JSONB NOT NULL DEFAULT '{}'::jsonb,
    pool_billing  BOOLEAN NOT NULL DEFAULT FALSE,  -- master sub vs per-client billing
    created_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);

ALTER TABLE clients
    ADD COLUMN IF NOT EXISTS organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_clients_organization_id ON clients(organization_id);

-- Org-level membership : who belongs to an org, with what org-level role.
-- 'owner' can manage members + billing ; 'admin' can manage clients ;
-- 'member' has only the per-client roles granted in org_user_clients.
CREATE TABLE IF NOT EXISTS organization_users (
    organization_id    UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id            UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role               TEXT NOT NULL DEFAULT 'member',
                       -- 'owner' | 'admin' | 'member'
    invited_by_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    joined_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (organization_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_organization_users_user_id ON organization_users(user_id);

-- Per-client role inside an org. A user can be 'editor' on client A and
-- 'viewer' on client B within the same org (account-manager scoping).
-- Constraint via FKs : (org, client) pair must match clients.organization_id,
-- but we don't enforce it via CHECK here (would need a trigger or app-level
-- guard). Application code validates on write.
CREATE TABLE IF NOT EXISTS org_user_clients (
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    client_id       UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
                    -- 'viewer' | 'editor' | 'owner' — mirrors the legacy
                    -- user_clients.role values for drop-in compatibility.
    PRIMARY KEY (organization_id, user_id, client_id)
);

CREATE INDEX IF NOT EXISTS idx_oucl_user_id ON org_user_clients(user_id);
CREATE INDEX IF NOT EXISTS idx_oucl_client_id ON org_user_clients(client_id);

-- ── Backfill ─────────────────────────────────────────────────────────
-- For each existing client without an organization_id, create a personal
-- workspace named '{client.name} workspace' and link the client + its
-- existing UserClient rows. Idempotent on re-run (skips clients that
-- already have an org).
DO $$
DECLARE
    c_row RECORD;
    new_org_id UUID;
    base_slug TEXT;
BEGIN
    FOR c_row IN
        SELECT id, name FROM clients WHERE organization_id IS NULL
    LOOP
        base_slug := LOWER(REGEXP_REPLACE(c_row.name, '[^a-zA-Z0-9]+', '-', 'g'));
        base_slug := TRIM(BOTH '-' FROM base_slug);
        IF base_slug = '' THEN base_slug := 'workspace'; END IF;
        -- Append client-id prefix to guarantee slug uniqueness
        base_slug := base_slug || '-' || SUBSTRING(c_row.id::text, 1, 8);

        INSERT INTO organizations (name, slug, is_personal)
        VALUES (
            c_row.name || ' workspace',
            base_slug,
            TRUE
        )
        RETURNING id INTO new_org_id;

        UPDATE clients SET organization_id = new_org_id WHERE id = c_row.id;

        -- Mirror existing user_clients rows into the new tables.
        INSERT INTO organization_users (organization_id, user_id, role, joined_at)
        SELECT
            new_org_id, uc.user_id,
            -- Promote the user with the highest legacy role to 'owner' of the
            -- new personal org so SOMEONE can manage it. Lower roles → 'member'.
            CASE WHEN uc.role = 'owner' THEN 'owner' ELSE 'member' END,
            NOW()
        FROM user_clients uc
        WHERE uc.client_id = c_row.id
        ON CONFLICT (organization_id, user_id) DO NOTHING;

        INSERT INTO org_user_clients (organization_id, user_id, client_id, role)
        SELECT new_org_id, uc.user_id, uc.client_id, uc.role
        FROM user_clients uc
        WHERE uc.client_id = c_row.id
        ON CONFLICT (organization_id, user_id, client_id) DO NOTHING;
    END LOOP;
END $$;

COMMENT ON TABLE organizations IS
    'Phase E.C. The agency / workspace entity. is_personal=TRUE means '
    'the org was auto-created by migration 029 for an existing client.';

COMMENT ON COLUMN clients.organization_id IS
    'Owning organization (Phase E.C). Nullable for legacy rows, but the '
    'migration backfill ensures every existing client has one.';

COMMENT ON TABLE organization_users IS
    'Org-level membership. Per-client roles live in org_user_clients.';

COMMENT ON TABLE org_user_clients IS
    'Per-client access within an org. Replaces user_clients as the source '
    'of truth once services/access.py is fully wired through every router.';
