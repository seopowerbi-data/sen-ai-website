from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://senai:senai-change-in-prod@postgres:5432/senai"

    # HaloScan
    haloscan_api_key: str = ""
    haloscan_base_url: str = "https://api.haloscan.com"

    # LLM (platform defaults — clients can override with their own keys)
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # OAuth (worker needs to decrypt tokens for sync jobs)
    oauth_fernet_key: str = ""

    # Google OAuth (for token refresh in worker)
    google_client_id: str = ""
    google_client_secret: str = ""

    # Worker
    worker_id: str = "worker-1"
    poll_interval: int = 2

    # Per-task model overrides — read from MODEL_<TASK_UPPER> env vars.
    # Defaults reflect the production state at the time of TASK_MODELS introduction;
    # changing model_* defaults in code is fine, but env override is the supported way
    # to A/B test or upgrade a single task without touching code.
    model_classify_topics: str = "claude-haiku-4-5-20251001"
    model_generate_personas: str = "claude-haiku-4-5-20251001"
    model_generate_persona_questions: str = "claude-haiku-4-5-20251001"
    model_cleanup_brands: str = "claude-sonnet-4-6"
    model_generate_domain_brief: str = "gpt-4.1-mini"
    model_generate_domain_brief_gemini: str = "gemini-2.5-flash"
    model_generate_domain_brief_claude: str = "claude-sonnet-4-6"
    model_generate_editorial: str = "claude-sonnet-4-6"
    model_scan_test_openai: str = "gpt-4.1-mini"
    model_scan_test_gemini: str = "gemini-2.5-flash"
    model_brand_analyzer: str = "gemini-2.5-flash-lite"
    model_classify_question_intent: str = "claude-haiku-4-5-20251001"

    @property
    def task_models(self) -> dict[str, str]:
        """Lookup table for per-task model selection — `settings.task_models["classify_topics"]`."""
        return {
            "classify_topics": self.model_classify_topics,
            "generate_personas": self.model_generate_personas,
            "generate_persona_questions": self.model_generate_persona_questions,
            "cleanup_brands": self.model_cleanup_brands,
            "generate_domain_brief": self.model_generate_domain_brief,
            "generate_domain_brief_gemini": self.model_generate_domain_brief_gemini,
            "generate_domain_brief_claude": self.model_generate_domain_brief_claude,
            "generate_editorial": self.model_generate_editorial,
            "scan_test_openai": self.model_scan_test_openai,
            "scan_test_gemini": self.model_scan_test_gemini,
            "brand_analyzer": self.model_brand_analyzer,
            "classify_question_intent": self.model_classify_question_intent,
        }

    # extra='ignore' lets services own their own env vars without forcing every
    # one to be declared here. Observability vars (SENTRY_DSN, HEALTHCHECK_*,
    # LLM_DAILY_COST_CAP_USD) are read via os.environ inside their respective
    # services modules — same pattern as worker/services/embeddings.py.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
