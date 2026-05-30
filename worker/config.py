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

    # Reddit OAuth (Sprint 8) - script-type app at reddit.com/prefs/apps.
    # Used to bypass Reddit's IP-based block on the public *.json endpoint
    # from cloud providers. We use app-only auth (client_credentials grant)
    # so no Reddit user account is involved.
    reddit_client_id: str = ""
    reddit_client_secret: str = ""

    # Worker
    worker_id: str = "worker-1"
    poll_interval: int = 2

    # Shared with the api container - same env var INTERNAL_SERVICE_TOKEN.
    # Sent as X-Internal-Token header when calling `POST /scans/{id}/auto-rescan`
    # from the auto-rescan cron sweep.
    internal_service_token: str = ""
    api_internal_base_url: str = "http://api:8000"

    # Job-type filtering for multi-worker fan-out (Step 1 of worker scaling).
    # Comma-separated lists. Empty = no filter (legacy behavior, one worker
    # handles everything). Typical split:
    #   worker-content : WORKER_JOB_TYPES_INCLUDE=generate_article,generate_faq
    #   worker-scan    : WORKER_JOB_TYPES_EXCLUDE=generate_article,generate_faq
    # The 10-min generate_article no longer blocks the (seconds-fast) scan
    # pipeline jobs. Both workers stay single-threaded so we don't introduce
    # cross-job races inside the same scan_id.
    worker_job_types_include: str = ""
    worker_job_types_exclude: str = ""

    @property
    def job_types_include(self) -> list[str]:
        return [s.strip() for s in self.worker_job_types_include.split(",") if s.strip()]

    @property
    def job_types_exclude(self) -> list[str]:
        return [s.strip() for s in self.worker_job_types_exclude.split(",") if s.strip()]

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
    # Phase BB per-brand brief — same 3-tier defaults as the workspace brief.
    model_generate_brand_brief: str = "gpt-4.1-mini"
    model_generate_brand_brief_gemini: str = "gemini-2.5-flash"
    model_generate_brand_brief_claude: str = "claude-sonnet-4-6"
    model_generate_editorial: str = "claude-sonnet-4-6"
    model_scan_test_openai: str = "gpt-4.1-mini"
    model_scan_test_gemini: str = "gemini-2.5-flash"
    model_brand_analyzer: str = "gemini-2.5-flash-lite"
    model_classify_question_intent: str = "claude-haiku-4-5-20251001"
    model_judge_question_responses: str = "claude-haiku-4-5-20251001"

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
            "generate_brand_brief": self.model_generate_brand_brief,
            "generate_brand_brief_gemini": self.model_generate_brand_brief_gemini,
            "generate_brand_brief_claude": self.model_generate_brand_brief_claude,
            "generate_editorial": self.model_generate_editorial,
            "scan_test_openai": self.model_scan_test_openai,
            "scan_test_gemini": self.model_scan_test_gemini,
            "brand_analyzer": self.model_brand_analyzer,
            "classify_question_intent": self.model_classify_question_intent,
            "judge_question_responses": self.model_judge_question_responses,
        }

    # extra='ignore' lets services own their own env vars without forcing every
    # one to be declared here. Observability vars (SENTRY_DSN, HEALTHCHECK_*,
    # LLM_DAILY_COST_CAP_USD) are read via os.environ inside their respective
    # services modules — same pattern as worker/services/embeddings.py.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
