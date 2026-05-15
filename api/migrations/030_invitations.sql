-- 030_invitations.sql
--
-- Phase E.C.4 — Org invitation flow.
--
-- A pending invite : email + target org + intended org role + signed-random
-- token + expiry. On accept, the invitee becomes a member of
-- organization_users with the offered role. Per-client access is NOT
-- granted here — admins must explicitly add the new member to clients via
-- the members page (C.5). Keeps the "I invited someone, now they see the
-- whole org by default" surprise off.
--
-- Lifecycle :
--   created  → row inserted, email sent
--   accepted → accepted_at set, accepted_by_user_id set, org_users row created
--   revoked  → revoked_at set (admin clicked Revoke), token becomes inert
--   expired  → expires_at < NOW() ; token rejected at accept time
--
-- Idempotency on accept : if the invitee is already an org member, the
-- accept handler treats the call as already-applied (200, no new row), so
-- a duplicated email link click never errors.

CREATE TABLE IF NOT EXISTS invitations (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id       UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email                 TEXT NOT NULL,
    org_role              TEXT NOT NULL DEFAULT 'member',
    token                 TEXT NOT NULL UNIQUE,
    invited_by_user_id    UUID REFERENCES users(id) ON DELETE SET NULL,
    message               TEXT,
    created_at            TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at            TIMESTAMP NOT NULL,
    accepted_at           TIMESTAMP,
    accepted_by_user_id   UUID REFERENCES users(id) ON DELETE SET NULL,
    revoked_at            TIMESTAMP,
    CONSTRAINT invitations_role_valid CHECK (org_role IN ('owner','admin','member'))
);

CREATE INDEX IF NOT EXISTS invitations_org_idx ON invitations(organization_id);
CREATE INDEX IF NOT EXISTS invitations_email_idx ON invitations(LOWER(email));
-- The pending lookup (accept handler) hits by token only ; UNIQUE on token
-- already gives us the index. No extra index needed.
