import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, ForeignKey, Enum, Boolean, create_engine
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import settings

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    password_hash = Column(String(255))  # null if Google OAuth only
    google_id = Column(String(255), unique=True)
    is_superadmin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    client_links = relationship("UserClient", back_populates="user")


class Client(Base):
    __tablename__ = "clients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    brand = Column(String(255))
    stripe_customer_id = Column(String(255))
    apps = Column(JSONB)  # feature flags + workspace state (client_brief lives here)
    primary_brand_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)  # see brand_resolver.py
    created_at = Column(DateTime, default=datetime.utcnow)

    user_links = relationship("UserClient", back_populates="client")
    subscriptions = relationship("Subscription", back_populates="client")
    api_keys = relationship("ClientApiKey", back_populates="client")


class UserClient(Base):
    __tablename__ = "user_clients"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), primary_key=True)
    role = Column(Enum("owner", "editor", "viewer", name="user_role"), default="viewer")

    user = relationship("User", back_populates="client_links")
    client = relationship("Client", back_populates="user_links")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    stripe_subscription_id = Column(String(255))
    plan = Column(String(50))  # "ai_scan", "store_impact", "both"
    status = Column(String(50), default="active")  # active, canceled, past_due
    created_at = Column(DateTime, default=datetime.utcnow)

    client = relationship("Client", back_populates="subscriptions")


class ClientCredit(Base):
    """Credit ledger — each row is a transaction (purchase or consumption)."""
    __tablename__ = "client_credits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=False)
    credit_type = Column(String(20), nullable=False)  # 'scan' | 'content'
    amount = Column(Integer, nullable=False)  # positive = credit, negative = debit
    balance_after = Column(Integer, nullable=False)
    description = Column(String(255))
    stripe_session_id = Column(String(255))
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=datetime.utcnow)


class ClientApiKey(Base):
    __tablename__ = "client_api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    provider = Column(String(50), nullable=False)  # "openai", "anthropic", "gemini"
    api_key_encrypted = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    client = relationship("Client", back_populates="api_keys")


class ClientModule(Base):
    __tablename__ = "client_modules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    module_key = Column(String(50), nullable=False)  # "ai_scan", "store_impact"
    is_active = Column(Boolean, default=True)
    activated_at = Column(DateTime, default=datetime.utcnow)


class ClientBrand(Base):
    __tablename__ = "client_brands"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("client_brands.id", ondelete="SET NULL"))
    name = Column(String(255), nullable=False)
    canonical_name = Column(String(255))  # normalized name for dedup (Phase 1)
    aliases = Column(ARRAY(String))
    category = Column(String(30), default="unclassified")
        # DEPRECATED in Phase 1 (lazy deprecation). Classification now lives in scan_brand_classifications.
        # target_brand, target_gamme, target_product, competitor, competitor_gamme, unclassified, ignored
    domain = Column(String(255))
    first_detected_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime)  # updated when brand is re-detected in a subsequent scan
    detected_in_scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="SET NULL"))
    detection_source = Column(String(30))  # keywords, llm_response, haloscan_competitors, manual
    auto_detected = Column(Boolean, default=True)
    validated_by_user = Column(Boolean, default=False)
    # Phase D sitemap config (migration 028). See api/migrations/028_*.sql.
    sitemap_urls_override = Column(JSONB, nullable=False, default=list)
    locale_path_prefix = Column(Text)
    # Phase C.1 multi-vertical article generator (migration 032).
    # See api/migrations/032_client_brand_vertical_metadata.sql.
    # PARITÉ obligatoire avec api/models.py — la duplication worker/models.py
    # vs api/models.py est un foot-gun connu : oublier d'updater l'un fait
    # crasher l'autre avec AttributeError sur la nouvelle colonne.
    expert_section_paths = Column(JSONB, nullable=False, default=list)
    product_lines = Column(JSONB, nullable=False, default=list)

    parent = relationship("ClientBrand", remote_side=[id])


class ClientBrandPage(Base):
    """Sitemap-discovered page for a client_brand domain.

    Phase D — see api/migrations/025_client_brand_pages.sql for the lifecycle
    diagram and column semantics.
    """

    __tablename__ = "client_brand_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_brand_id = Column(
        UUID(as_uuid=True),
        ForeignKey("client_brands.id", ondelete="CASCADE"),
        nullable=False,
    )
    url = Column(Text, nullable=False)
    url_canonical = Column(Text)
    title = Column(Text)
    meta_description = Column(Text)
    h1 = Column(Text)
    body_excerpt = Column(Text)
    lang = Column(Text)
    lastmod = Column(DateTime)
    content_hash = Column(Text)
    internal_inlink_count = Column(Integer, nullable=False, default=0)
    embedding = Column(JSONB)
    embedding_model = Column(Text)
    status = Column(Text, nullable=False, default="pending_fetch")
        # pending_fetch | fetched | embedded | gone | error
    fetch_error = Column(Text)
    fetch_retry_count = Column(Integer, nullable=False, default=0)
    http_status = Column(Integer)
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)
    last_crawled_at = Column(DateTime)
    last_embedded_at = Column(DateTime)
    gone_since = Column(DateTime)
    # Discovery source: 'sitemap' (default) or 'manual' (user-added).
    # Migration 027.
    source = Column(Text, nullable=False, default="sitemap")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    name = Column(String(255))  # user-facing scan name, defaults to domain
    domain = Column(String(255), nullable=False)
    status = Column(String(30), default="draft")
        # draft → fetching_keywords → keywords_fetched → topics_ready
        # → assigning_keywords → brands_ready → generating_personas → personas_ready
        # → scanning → completed | failed
    focus_brand_id = Column(UUID(as_uuid=True), ForeignKey("client_brands.id", ondelete="SET NULL"))
    promotion_brand_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)  # per-scan override of client.primary_brand_ids
    # User-declared scan intent: own_brand | competitor_audit | NULL.
    # See migration 022 + worker/services/brand_resolver.py is_competitor_scan().
    scan_type = Column(Text)
    parent_scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="SET NULL"))
    schedule = Column(String(20), default="manual")  # manual | weekly | monthly
    next_run_at = Column(DateTime)
    run_index = Column(Integer, default=1)  # 1 = initial, 2+ = rescan
    config = Column(JSONB, default={})
    progress_pct = Column(Integer, default=0)
    progress_message = Column(Text)
    summary = Column(JSONB)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)

    focus_brand = relationship("ClientBrand", foreign_keys=[focus_brand_id])
    parent_scan = relationship("Scan", remote_side=[id])
    keywords = relationship("ScanKeyword", back_populates="scan", cascade="all, delete-orphan")
    topics = relationship("ScanTopic", back_populates="scan", cascade="all, delete-orphan")
    personas = relationship("ScanPersona", back_populates="scan", cascade="all, delete-orphan")
    content_items = relationship("ScanContentItem", back_populates="scan", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="scan", cascade="all, delete-orphan")
    brand_classifications = relationship("ScanBrandClassification", back_populates="scan", cascade="all, delete-orphan")


class ScanBrandClassification(Base):
    """Per-scan brand classification (Phase 1 scan-as-brand model).

    Each scan classifies brands independently. One focus brand per scan
    (enforced via partial unique index idx_sbc_one_focus_per_scan at the DB level).
    """
    __tablename__ = "scan_brand_classifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("client_brands.id", ondelete="CASCADE"), nullable=False)
    classification = Column(String(20), nullable=False)
        # my_brand | competitor | ignored | unclassified
    is_focus = Column(Boolean, default=False)
    classified_by = Column(String(20), default="auto")  # auto | claude | user
    source = Column(String(30))  # inherited from ClientBrand.detection_source
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    scan = relationship("Scan", back_populates="brand_classifications")
    brand = relationship("ClientBrand")


class ScanBrandTopic(Base):
    """Brand-topic junction: which brands are relevant to which topics (per scan).

    Populated by Claude during classify_topics. A brand can appear in multiple topics.
    """
    __tablename__ = "scan_brand_topics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("client_brands.id", ondelete="CASCADE"), nullable=False)
    topic_id = Column(UUID(as_uuid=True), ForeignKey("scan_topics.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    scan = relationship("Scan")
    brand = relationship("ClientBrand")
    topic = relationship("ScanTopic")


class ScanKeyword(Base):
    __tablename__ = "scan_keywords"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    topic_id = Column(UUID(as_uuid=True), ForeignKey("scan_topics.id", ondelete="SET NULL"))
    url = Column(Text, nullable=False)
    keyword = Column(String(500), nullable=False)
    position = Column(Integer)
    traffic = Column(Integer)
    search_volume = Column(Integer)

    scan = relationship("Scan", back_populates="keywords")
    topic = relationship("ScanTopic")


class ScanTopic(Base):
    __tablename__ = "scan_topics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    example_keywords = Column(ARRAY(String))
    matching_terms = Column(ARRAY(String))  # Terms for programmatic keyword matching
    keyword_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    display_order = Column(Integer, default=0)

    scan = relationship("Scan", back_populates="topics")
    personas = relationship("ScanPersona", back_populates="topic")


class ScanPersona(Base):
    __tablename__ = "scan_personas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    topic_id = Column(UUID(as_uuid=True), ForeignKey("scan_topics.id", ondelete="SET NULL"))
    name = Column(String(255), nullable=False)
    data = Column(JSONB, nullable=False)
    is_active = Column(Boolean, default=True)

    scan = relationship("Scan", back_populates="personas")
    topic = relationship("ScanTopic", back_populates="personas")
    questions = relationship("ScanQuestion", back_populates="persona", cascade="all, delete-orphan")


class ScanQuestion(Base):
    __tablename__ = "scan_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    persona_id = Column(UUID(as_uuid=True), ForeignKey("scan_personas.id", ondelete="CASCADE"), nullable=False)
    question = Column(Text, nullable=False)
    type_question = Column(String(30))
    is_active = Column(Boolean, default=True)

    persona = relationship("ScanPersona", back_populates="questions")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id"), nullable=True)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=True)
    job_type = Column(String(50), nullable=False)
    status = Column(String(30), default="pending")
    payload = Column(JSONB, default={})
    result = Column(JSONB)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    scan = relationship("Scan", back_populates="jobs")


class ScanContentItem(Base):
    __tablename__ = "scan_content_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    content_type = Column(String(30), nullable=False)  # "faq" | "netlinking_article"
    topic_name = Column(String(255))
    persona_name = Column(String(255))

    # Target
    target_url = Column(Text)
    target_page_title = Column(String(500))
    target_question = Column(Text)
    # Provenance of target_url — drives Kanban "Needs URL" badge + validation UI.
    # Values: scan_result | pending_user | user_input | auto_suggest | sitemap_index.
    # See api/migrations/020_target_url_source.sql + 026_target_url_score_and_candidates.sql.
    target_url_source = Column(Text)
    # Sitemap matcher final score (cosine × authority × gamme bias) for target_url.
    # NULL when no sitemap match available. Migration 026.
    target_url_score = Column(Float)
    # Top-3 sitemap matcher picks [{"url","title","score"}] — drives top-3 picker UX.
    # Migration 026.
    target_url_candidates = Column(JSONB, nullable=False, default=list)

    # Content
    content_html = Column(Text)
    content_text = Column(Text)
    article_outline = Column(Text)
    gdrive_doc_url = Column(String(500))

    # Opportunity metrics
    priority = Column(String(20))  # "critique", "haute", "moyenne"
    opportunity_score = Column(Float)
    brand_position = Column(Float)
    best_competitor = Column(String(500))
    nb_competitors_cited = Column(Integer)

    # Netlinking specific
    estimated_price = Column(Float)
    platform_link = Column(String(500))

    # Audit trail: brands instructed to be promoted at generation time
    promoted_brand_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)

    # Workflow
    status = Column(String(30), default="identified")
    validation = Column(String(20))  # "approved", "needs_revision", "rejected"
    validated_by = Column(String(255))
    validated_at = Column(DateTime)

    # Lifecycle dates
    identified_at = Column(DateTime, default=datetime.utcnow)
    ordered_at = Column(DateTime)
    published_at = Column(DateTime)
    published_url = Column(Text)

    # Post-publication tracking
    latest_scan_date = Column(DateTime)
    latest_position = Column(Float)
    position_delta = Column(Float)

    # Rejected target URLs (accumulated across "Find a different page" clicks)
    rejected_target_urls = Column(JSONB, nullable=False, default=list)
    # Audit payload written by the worker on every successful generation —
    # quality_score, sources cited (with org names), denylist drops. Migration
    # 024.
    content_metadata = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)

    scan = relationship("Scan", back_populates="content_items")


class ScanLLMResult(Base):
    __tablename__ = "scan_llm_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    question_id = Column(UUID(as_uuid=True), ForeignKey("scan_questions.id", ondelete="SET NULL"))
    provider = Column(String(30), nullable=False)  # "openai", "gemini"
    model = Column(String(100))
    response_text = Column(Text)
    citations = Column(JSONB)  # [{url, domain, source_type, title}]
    target_cited = Column(Boolean)
    target_position = Column(Integer)
    total_citations = Column(Integer)
    competitor_domains = Column(JSONB)  # {domain: count}
    brand_mentions = Column(JSONB)     # [{brand_name, sentiment, est_recommandation, position_index, ...}]
    brand_analysis = Column(JSONB)     # {nb_marques, marque_cible_mentionnee, sentiment_marque_cible, ...}
    duration_ms = Column(Integer)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class ScanOpportunity(Base):
    __tablename__ = "scan_opportunities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    question_id = Column(UUID(as_uuid=True), ForeignKey("scan_questions.id", ondelete="SET NULL"))
    topic_name = Column(String(255))
    persona_name = Column(String(255))

    # Brand position
    brand_cited = Column(Boolean)
    brand_position = Column(Integer)
    brand_sentiment = Column(String(20))
    brand_recommended = Column(Boolean)

    # Competitor position
    best_competitor_name = Column(String(255))
    best_competitor_position = Column(Integer)
    best_competitor_domain = Column(String(255))
    nb_competitors_cited = Column(Integer)

    # Scoring
    priority = Column(String(20), nullable=False)  # critique, haute, moyenne
    opportunity_score = Column(Float)

    # Recommended action
    recommended_action = Column(String(30))  # faq, netlinking, content_update
    target_url = Column(Text)
    media_domain = Column(String(255))

    created_at = Column(DateTime, default=datetime.utcnow)


class OAuthConnection(Base):
    """OAuth delegation — needed by worker to read tokens for sync jobs."""
    __tablename__ = "oauth_connections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(50), nullable=False)
    product = Column(String(50), nullable=False)
    account_id = Column(String(255))
    account_email = Column(String(255))
    access_token_encrypted = Column(Text)
    refresh_token_encrypted = Column(Text)
    token_expires_at = Column(DateTime)
    scopes = Column(ARRAY(String))
    config = Column(JSONB, default={})
    status = Column(String(20), nullable=False, default="active")
    authorized_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    authorized_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LlmUsageLog(Base):
    """Platform-wide LLM API usage tracking for cost monitoring."""
    __tablename__ = "llm_usage_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider = Column(String(30), nullable=False)
    model = Column(String(100), nullable=False)
    operation = Column(String(50), nullable=False)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0)
    duration_ms = Column(Integer)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id", ondelete="SET NULL"))
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"))
    error = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
