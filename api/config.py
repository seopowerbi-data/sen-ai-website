from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://senai:senai-change-in-prod@postgres:5432/senai"

    # JWT
    jwt_secret: str = "CHANGE-THIS-SECRET-IN-PROD"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours

    # Shared with the worker, set via INTERNAL_SERVICE_TOKEN in .env. Used by
    # the worker to call `POST /scans/{id}/auto-rescan` when the auto-rescan
    # cron sweep detects a due scan. Header : X-Internal-Token.
    internal_service_token: str = ""

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "https://sen-ai.fr/api/auth/google/callback"

    # Stripe
    stripe_api_key: str = ""
    stripe_webhook_secret: str = ""

    # OAuth delegation (Phase 0)
    oauth_fernet_key: str = ""  # Fernet symmetric key for encrypting tokens at rest
    oauth_google_redirect_uri: str = "https://sen-ai.fr/api/oauth/google/callback"

    # Email (Resend) — optional, logs reset URL if not configured
    resend_api_key: str = ""
    resend_from_email: str = "sen-ai.fr <noreply@sen-ai.fr>"
    audit_notification_email: str = "data@sen-ai.fr"

    # Frontend
    frontend_url: str = "https://sen-ai.fr"

    # Registration toggle - set to True to reopen account creation
    # (audit-gratuit form remains open regardless - it's a separate flow without account)
    registration_open: bool = False

    # Observability - Sentry error reporting (optional; logs warning if empty)
    sentry_dsn: str = ""
    sentry_environment: str = "production"

    # Sprint 4 - In-app chatbot (Anthropic SDK tool-loop)
    anthropic_api_key: str = ""
    agent_model: str = "claude-sonnet-4-6"
    agent_max_iterations: int = 10
    agent_daily_cap_free: int = 30
    agent_daily_cap_premium: int = 300

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
