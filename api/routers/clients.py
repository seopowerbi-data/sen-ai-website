from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from services.rate_limit import limiter
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sqlalchemy.orm.attributes import flag_modified

from models import Client, ClientBrand, ClientBrandPage, ClientCredit, Job, ScanBrandClassification, UserClient, get_db
from services.auth_service import get_current_user
from services.request_context import current_request_method
from services.sanitize import strip_tags

router = APIRouter()

# RBAC mirror of brands.py — viewer can read, editor+ can write
_ROLE_RANK = {"viewer": 0, "editor": 1, "owner": 2}
_DESTRUCTIVE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _check_client_access(client_id: str, user, db: Session):
    """Same role-gate pattern as brands.py:_check_client_access."""
    link = db.query(UserClient).filter(
        UserClient.user_id == user.id, UserClient.client_id == client_id,
    ).first()
    if not link:
        raise HTTPException(403, "Access denied")
    method = current_request_method.get()
    if method in _DESTRUCTIVE_METHODS:
        rank = _ROLE_RANK.get(link.role, -1)
        if rank < _ROLE_RANK["editor"]:
            raise HTTPException(
                403,
                f"Insufficient role: '{link.role}' cannot {method} client settings "
                f"(requires 'editor' or 'owner')",
            )


class ClientResponse(BaseModel):
    id: str
    name: str
    brand: str | None
    apps: dict | None = None

    model_config = {"from_attributes": True}


class ClientCreate(BaseModel):
    name: str
    brand: str | None = None


@router.get("/", response_model=list[ClientResponse])
async def list_clients(user=Depends(get_current_user), db: Session = Depends(get_db)):
    links = db.query(UserClient).filter(UserClient.user_id == user.id).all()
    client_ids = [link.client_id for link in links]
    clients = db.query(Client).filter(Client.id.in_(client_ids)).all()
    return [ClientResponse(id=str(c.id), name=c.name, brand=c.brand, apps=c.apps) for c in clients]


@router.post("/")
async def create_client(req: ClientCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if user already has a client
    existing = db.query(UserClient).filter(UserClient.user_id == user.id).first()
    if existing:
        client = db.query(Client).filter(Client.id == existing.client_id).first()
        return ClientResponse(id=str(client.id), name=client.name, brand=client.brand, apps=client.apps)

    # Create new client + link user as owner
    # Welcome bonus is now granted on email verification (H3), not here
    client = Client(name=strip_tags(req.name), brand=strip_tags(req.brand))
    db.add(client)
    db.flush()

    db.add(UserClient(user_id=user.id, client_id=client.id, role="owner"))

    db.commit()
    db.refresh(client)

    return ClientResponse(id=str(client.id), name=client.name, brand=client.brand, apps=client.apps)


# ── Brand promotion settings ────────────────────────────────────────────
# These endpoints back the Workspace Settings → "My primary brands" UI.
# primary_brand_ids is the cross-scan default for content-gen promotion
# (FAQ / Article generation). Resolution chain documented in
# worker/services/brand_resolver.py.

class PromotionUpdate(BaseModel):
    primary_brand_ids: list[str]  # ordered, [0] = lead brand


@router.get("/{client_id}/promotion")
async def get_client_promotion(client_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return current promotion settings + all client brands + auto-detected suggestions.

    Brands are nested : children (gammes / product lines linked via parent_id)
    appear inside their parent's `children` array rather than as flat entries.
    Mirrors the GET /scans/{id}/brands shape so the workspace settings UI can
    reuse the same hierarchical drag-drop pattern (Gate 3).
    """
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    primary_ids = list(client.primary_brand_ids or [])
    primary_id_set = {str(bid) for bid in primary_ids}

    all_brands = (
        db.query(ClientBrand)
        .filter(ClientBrand.client_id == client_id)
        .order_by(ClientBrand.name)
        .all()
    )

    # Auto-detected my_brand brands (across all scans for this client)
    suggested_rows = (
        db.query(ScanBrandClassification.brand_id)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(
            ClientBrand.client_id == client_id,
            ScanBrandClassification.classification == "my_brand",
        )
        .distinct()
        .all()
    )
    suggested_id_set = {str(r.brand_id) for r in suggested_rows}

    by_id = {str(b.id): b for b in all_brands}

    def _to_dict(b: ClientBrand) -> dict:
        return {
            "id": str(b.id),
            "name": b.name,
            "domain": b.domain,
            "parent_id": str(b.parent_id) if b.parent_id else None,
            "is_primary": str(b.id) in primary_id_set,
            "is_suggested": str(b.id) in suggested_id_set,
            "children": [],
        }

    # Children-by-parent map (only for nesting under a primary OR another root).
    # Children are not added to the root flat list; they live inside parent.children.
    child_ids_to_skip: set[str] = set()
    children_by_parent: dict[str, list[dict]] = {}
    for b in all_brands:
        if not b.parent_id:
            continue
        pid_str = str(b.parent_id)
        children_by_parent.setdefault(pid_str, []).append(_to_dict(b))
        child_ids_to_skip.add(str(b.id))

    # Emit roots in deterministic order : primary brands first (in primary_ids
    # order = drag-reorder authority), then everything else alphabetical.
    serialized: list[dict] = []
    for bid in primary_ids:
        bid_str = str(bid)
        b = by_id.get(bid_str)
        if not b or bid_str in child_ids_to_skip:
            continue
        d = _to_dict(b)
        d["children"] = children_by_parent.get(bid_str, [])
        serialized.append(d)
    for b in all_brands:
        bid_str = str(b.id)
        if bid_str in primary_id_set:
            continue
        if bid_str in child_ids_to_skip:
            continue
        d = _to_dict(b)
        d["children"] = children_by_parent.get(bid_str, [])
        serialized.append(d)

    return {
        "primary_brand_ids": [str(bid) for bid in primary_ids],
        "all_brands": serialized,
    }


class BrandParentUpdate(BaseModel):
    parent_id: str | None = None  # null = detach (top-level)


@router.patch("/{client_id}/brands/{brand_id}/parent")
async def update_brand_parent(client_id: str, brand_id: str, req: BrandParentUpdate,
                              user=Depends(get_current_user),
                              db: Session = Depends(get_db)):
    """Set or clear the parent_id on a client_brand.

    Used by the workspace-settings drag-drop UI : when the user drops a brand
    onto a primary brand, we PATCH parent_id; when they detach a child to the
    Available column, we PATCH parent_id=NULL.

    Same brand graph the per-scan classifier (Gate 3) uses — a brand has at
    most one parent, children can't have grand-children (enforced here).
    """
    _check_client_access(client_id, user, db)
    brand = (
        db.query(ClientBrand)
        .filter(ClientBrand.id == brand_id, ClientBrand.client_id == client_id)
        .first()
    )
    if not brand:
        raise HTTPException(404, "Brand not found")

    new_parent_id = req.parent_id
    if new_parent_id is None or new_parent_id == "":
        brand.parent_id = None
        db.commit()
        return {"ok": True, "brand_id": str(brand.id), "parent_id": None}

    if new_parent_id == brand_id:
        raise HTTPException(400, "A brand cannot be its own parent")

    parent = (
        db.query(ClientBrand)
        .filter(ClientBrand.id == new_parent_id, ClientBrand.client_id == client_id)
        .first()
    )
    if not parent:
        raise HTTPException(400, "Parent brand not found in this client")

    # Prevent grand-children : if `parent` is itself a child, refuse.
    if parent.parent_id is not None:
        raise HTTPException(400, {
            "error": "grand_child_disallowed",
            "message": "Can't make a gamme a parent. Attach to the top-level brand instead.",
        })

    brand.parent_id = parent.id
    db.commit()
    return {
        "ok": True, "brand_id": str(brand.id),
        "parent_id": str(parent.id), "parent_name": parent.name,
    }


@router.put("/{client_id}/promotion")
async def update_client_promotion(client_id: str, req: PromotionUpdate,
                                  user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Replace the client's primary_brand_ids (workspace default for content gen)."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    valid_ids = {
        str(b.id) for b in
        db.query(ClientBrand).filter(ClientBrand.client_id == client_id).all()
    }
    invalid = [bid for bid in req.primary_brand_ids if bid not in valid_ids]
    if invalid:
        raise HTTPException(400, f"Brand IDs not in this client: {invalid[:3]}")

    try:
        client.primary_brand_ids = [UUID(bid) for bid in req.primary_brand_ids]
    except ValueError as e:
        raise HTTPException(400, f"Malformed UUID: {e}")
    db.commit()

    return {
        "ok": True,
        "primary_brand_ids": req.primary_brand_ids,
        "count": len(req.primary_brand_ids),
    }


# ── Workspace brief (client.apps.client_brief) ──────────────────────────
# The workspace brief describes the COMPANY (not any single scanned domain).
# It's injected into FAQ + article generation so the output sounds like the
# user's brand even when generated from a competitor scan opportunity.

class BriefUpdate(BaseModel):
    brief: dict  # full brief object — caller is responsible for shape


@router.get("/{client_id}/brief")
async def get_client_brief(client_id: str, user=Depends(get_current_user),
                           db: Session = Depends(get_db)):
    """Return the workspace brief + regen budget, or null brief if not yet generated."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")
    apps = client.apps or {}
    brief = apps.get("client_brief")
    used = int((brief or {}).get("generations_count") or 0)
    return {
        "brief": brief,
        "generations_used": used,
        "generations_cap": MAX_CLIENT_BRIEF_GENERATIONS,
        "can_regenerate": used < MAX_CLIENT_BRIEF_GENERATIONS,
    }


# Hard cap on per-client workspace brief regenerations. Each call fires
# OpenAI web_search (~$0.02-0.05). Workspace brief is regenerated rarely
# in practice (1-2x to seed, then user edits). 5 leaves room for LLM
# garbage on a brand-new workspace without enabling spam.
# See feedback_cap_user_triggered_llm_ops.
MAX_CLIENT_BRIEF_GENERATIONS = 5


@router.post("/{client_id}/brief/generate")
@limiter.limit("5/minute")
async def generate_client_brief(request: Request, client_id: str,
                                user=Depends(get_current_user),
                                db: Session = Depends(get_db)):
    """Enqueue a generate_client_brief worker job. Returns the job id for polling."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    # Block regeneration if user has explicitly edited — they should DELETE
    # via PUT (with edited_by_user=false) before regenerating.
    apps = client.apps or {}
    existing = apps.get("client_brief") or {}
    if existing.get("edited_by_user"):
        raise HTTPException(
            409,
            "Brief has been manually edited — delete edited_by_user via PUT before regenerating",
        )

    # Hard cap : 429 once the workspace brief has burned the regen budget.
    # Counter incremented in worker on success (failed runs don't count).
    used = int(existing.get("generations_count") or 0)
    if used >= MAX_CLIENT_BRIEF_GENERATIONS:
        raise HTTPException(429, {
            "error": "client_brief_regen_cap_reached",
            "message": f"Workspace brief has been generated {used} times "
                       f"(max {MAX_CLIENT_BRIEF_GENERATIONS}). Edit the brief manually "
                       f"in workspace settings — further regenerations are blocked.",
            "generations_used": used,
            "cap": MAX_CLIENT_BRIEF_GENERATIONS,
        })

    # Avoid duplicate in-flight jobs
    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "generate_client_brief",
            Job.status.in_(["pending", "running"]),
        )
        .first()
    )
    if in_flight:
        return {"ok": True, "job_id": str(in_flight.id), "status": in_flight.status,
                "message": "Already in flight",
                "generations_used": used, "cap": MAX_CLIENT_BRIEF_GENERATIONS}

    job = Job(
        client_id=client_id,
        job_type="generate_client_brief",
        status="pending",
        payload={"client_id": client_id},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"ok": True, "job_id": str(job.id), "status": "pending",
            "generations_used": used, "cap": MAX_CLIENT_BRIEF_GENERATIONS}


@router.put("/{client_id}/brief")
async def update_client_brief(client_id: str, req: BriefUpdate,
                              user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Replace the workspace brief with a manual edit (sets edited_by_user=true).

    Pass an explicit `edited_by_user: false` inside `brief` to re-allow regeneration.
    """
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    new_brief = dict(req.brief)
    # Default edited_by_user=true unless caller explicitly opts out (e.g., to clear flag)
    if "edited_by_user" not in new_brief:
        new_brief["edited_by_user"] = True

    apps = dict(client.apps or {})
    apps["client_brief"] = new_brief
    client.apps = apps
    flag_modified(client, "apps")
    db.commit()
    return {"ok": True, "edited_by_user": new_brief.get("edited_by_user", True)}


# ─── Trust sources (per-client authoritative reference domains) ──────────
# Discovery is automatically chained from generate_client_brief on success,
# but these endpoints expose manual control for: seeding existing clients
# whose brief predates the trust-sources feature, refreshing when the
# discovered list looks off, and future Settings UI integration.


@router.get("/{client_id}/trust-sources")
async def get_trust_sources(client_id: str, user=Depends(get_current_user),
                            db: Session = Depends(get_db)):
    """Read the current trust_sources payload for a client.

    Returns the persisted structure verbatim (or an empty stub if discovery
    has never run). The Settings UI uses this to render the list + last
    refresh date + refresh button.
    """
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    payload = (client.apps or {}).get("trust_sources") or {}
    return {
        "ok": True,
        "trust_sources": {
            "domains": payload.get("domains") or [],
            "details": payload.get("details") or [],
            "extra_domains": payload.get("extra_domains") or [],
            "industry_text": payload.get("industry_text") or "",
            "discovered_at": payload.get("discovered_at"),
            "sources_count": payload.get("sources_count") or 0,
        },
    }


@router.post("/{client_id}/trust-sources/discover")
@limiter.limit("3/minute")
async def discover_trust_sources(request: Request, client_id: str,
                                  force: bool = False,
                                  user=Depends(get_current_user),
                                  db: Session = Depends(get_db)):
    """Enqueue a discover_trust_sources worker job.

    Requires `client_brief.industry` to be set (otherwise the worker no-ops).
    Idempotent on the worker side : returns 'fresh' if the cached payload is
    still within TTL and industry hasn't changed. Pass ?force=true to bypass.
    """
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    apps = client.apps or {}
    brief = apps.get("client_brief") or {}
    if not (brief.get("industry") or "").strip():
        raise HTTPException(
            409,
            {
                "error": "missing_industry",
                "message": "Trust source discovery needs a workspace brief with "
                           "an `industry` field. Generate the workspace brief first.",
            },
        )

    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "discover_trust_sources",
            Job.status.in_(["pending", "running"]),
        )
        .first()
    )
    if in_flight:
        return {
            "ok": True, "job_id": str(in_flight.id), "status": in_flight.status,
            "message": "Already in flight",
        }

    job = Job(
        client_id=client_id,
        job_type="discover_trust_sources",
        status="pending",
        payload={"client_id": client_id, "force": bool(force)},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"ok": True, "job_id": str(job.id), "status": "pending"}


# ─── extra_domains : user-managed prefer-hint extension slot ─────────────
# These let the user widen the soft prefer-hint list when the discovered
# domains miss something they trust (e.g., an internal scientific publisher,
# a niche industry portal). HARD denylist (competitor) is unaffected — these
# only feed the SOFT prefer-hint list returned by
# `get_trust_sources_for_client`. The discover handler carries them forward
# on every refresh.

class ExtraDomainBody(BaseModel):
    domain: str


def _normalize_extra_domain(raw: str) -> str:
    """Mirror of worker.services.trust_sources._normalize_domain — kept local
    to avoid importing the worker module from the API container."""
    import re as _re
    if not raw or not isinstance(raw, str):
        return ""
    nd = _re.sub(r"^https?://", "", raw.strip().lower())
    if nd.startswith("www."):
        nd = nd[4:]
    nd = nd.split("/", 1)[0].strip().rstrip(".")
    if "." not in nd or len(nd) > 253:
        return ""
    # Disallow control chars / spaces — defensive against pasted junk
    if _re.search(r"[\s<>\"']", nd):
        return ""
    return nd


@router.post("/{client_id}/trust-sources/extra-domains")
async def add_trust_source_extra(client_id: str, body: ExtraDomainBody,
                                  user=Depends(get_current_user),
                                  db: Session = Depends(get_db)):
    """Append a user-managed domain to the prefer-hint list."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    domain = _normalize_extra_domain(body.domain)
    if not domain:
        raise HTTPException(400, {
            "error": "invalid_domain",
            "message": "Domain must be a bare hostname (e.g. example.com).",
        })

    apps = dict(client.apps or {})
    trust = dict(apps.get("trust_sources") or {})
    extras = list(trust.get("extra_domains") or [])
    discovered = {(d or "").lower() for d in (trust.get("domains") or [])}

    if domain in discovered:
        raise HTTPException(409, {
            "error": "already_discovered",
            "message": f"{domain} is already in the discovered list — no need to add it manually.",
        })
    if domain in extras:
        raise HTTPException(409, {
            "error": "duplicate",
            "message": f"{domain} is already in your extras.",
        })

    extras.append(domain)
    trust["extra_domains"] = extras
    apps["trust_sources"] = trust
    client.apps = apps
    flag_modified(client, "apps")
    db.commit()
    return {"ok": True, "extra_domains": extras}


# ─── Sitemap-index pages : per-brand crawl + manual URL escape hatch ──────
# Phase D. Lets the user :
#   - See the pages crawled from a brand's sitemap.xml + how many embedded
#   - Trigger a manual refresh (re-runs crawl_brand_sitemap, which chains
#     fetch_brand_pages and — Day 3 onward — embed_brand_pages)
#   - Add a page manually when the sitemap is missing / incomplete. Manual
#     rows enter the same fetch -> embed pipeline but are exempt from the
#     sitemap-diff mark_gone branch (see migration 027).

def _brand_for_client(brand_id: str, client_id: str, db: Session) -> ClientBrand:
    """Resolve a brand belonging to a client, 404 otherwise.

    Centralizes the access pattern : `_check_client_access` has already
    enforced the user can touch this client_id, so a brand mismatch is a
    not-found rather than a permission error.
    """
    brand = (
        db.query(ClientBrand)
        .filter(ClientBrand.id == brand_id, ClientBrand.client_id == client_id)
        .first()
    )
    if not brand:
        raise HTTPException(404, "Brand not found for this client")
    return brand


def _normalize_brand_host(domain: str) -> str:
    """Strip scheme/path/www/trailing slash, lowercase."""
    if not domain:
        return ""
    import re as _re
    d = _re.sub(r"^https?://", "", (domain or "").strip().lower())
    if d.startswith("www."):
        d = d[4:]
    return d.split("/", 1)[0].strip().rstrip(".")


def _validate_manual_page_url(raw_url: str, brand: ClientBrand) -> str:
    """Validate that a user-supplied URL is acceptable for this brand.

    Rules :
      - Must be https:// (we don't crawl http in v1 — sitemap_crawler tries
        https only too, so this keeps the pipeline coherent)
      - Hostname must match the brand's registered domain, with or without
        the www. prefix on either side. This prevents an editor from
        seeding URLs that point at unrelated sites (would pollute the
        matcher corpus + leak the user's brand index across competitors)
      - Must parse as a real URL — bare hostnames rejected
      - Returns the canonical-form URL (no trailing slash on bare host,
        scheme normalized to https, fragment stripped — fragments duplicate
        the parent page in the index)

    Raises HTTPException(400/422) with a structured error payload on
    failure so the UI can render a precise message.
    """
    if not raw_url or not isinstance(raw_url, str):
        raise HTTPException(400, {"error": "invalid_url", "message": "URL is required."})
    raw_url = raw_url.strip()
    if not raw_url:
        raise HTTPException(400, {"error": "invalid_url", "message": "URL is required."})

    from urllib.parse import urlparse, urlunparse
    try:
        parsed = urlparse(raw_url)
    except ValueError:
        raise HTTPException(400, {"error": "invalid_url", "message": "URL is malformed."})

    if parsed.scheme not in ("https",):
        raise HTTPException(
            422,
            {"error": "scheme_unsupported",
             "message": "URL must start with https://. http is not crawled in v1."},
        )
    if not parsed.hostname:
        raise HTTPException(400, {"error": "invalid_url", "message": "URL must include a host."})

    brand_host = _normalize_brand_host(brand.domain or "")
    if not brand_host:
        raise HTTPException(
            422,
            {"error": "brand_domain_missing",
             "message": "This brand has no registered domain. Set the brand domain in Settings → My primary brands first."},
        )
    url_host = parsed.hostname.lower()
    if url_host.startswith("www."):
        url_host = url_host[4:]
    if url_host != brand_host:
        raise HTTPException(
            422,
            {"error": "host_mismatch",
             "message": f"URL host must be {brand_host} (got {parsed.hostname}). Manual pages can only point to the brand's own domain."},
        )

    # Strip fragment + normalize trailing slash on path-empty URLs to keep
    # the dedup tight against the sitemap rows.
    clean = parsed._replace(fragment="")
    if clean.path == "":
        clean = clean._replace(path="/")
    return urlunparse(clean)


@router.get("/{client_id}/brands/{brand_id}/pages")
async def list_brand_pages(
    client_id: str, brand_id: str,
    source: str | None = None,   # 'sitemap' | 'manual' | None (all)
    status: str | None = None,
    limit: int = 200,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List indexed pages for a brand + summary stats.

    Drives the Settings sitemaps card : the user sees the global counts
    and (optionally) the manual-URLs list. Sitemap rows are large enough
    to need pagination ; for the default UI we surface only manual rows
    in detail (the user can't act on sitemap rows, only refresh).
    """
    _check_client_access(client_id, user, db)
    brand = _brand_for_client(brand_id, client_id, db)

    # Aggregate stats across the whole brand (no filter), one round-trip
    from sqlalchemy import func
    stats_rows = (
        db.query(
            ClientBrandPage.status,
            ClientBrandPage.source,
            func.count(ClientBrandPage.id).label("n"),
            func.max(ClientBrandPage.last_seen_at).label("last_seen"),
            func.max(ClientBrandPage.last_crawled_at).label("last_crawled"),
        )
        .filter(ClientBrandPage.client_brand_id == brand_id)
        .group_by(ClientBrandPage.status, ClientBrandPage.source)
        .all()
    )
    by_status: dict[str, int] = {}
    by_source: dict[str, int] = {}
    total = 0
    last_seen_max = None
    last_crawled_max = None
    for r in stats_rows:
        total += r.n
        by_status[r.status] = by_status.get(r.status, 0) + r.n
        by_source[r.source] = by_source.get(r.source, 0) + r.n
        if r.last_seen and (last_seen_max is None or r.last_seen > last_seen_max):
            last_seen_max = r.last_seen
        if r.last_crawled and (last_crawled_max is None or r.last_crawled > last_crawled_max):
            last_crawled_max = r.last_crawled

    # Active crawl job — drives a "Refreshing..." pill in the UI
    in_flight_crawl = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type.in_(("crawl_brand_sitemap", "fetch_brand_pages")),
            Job.status.in_(("pending", "running")),
            Job.payload["client_brand_id"].astext == str(brand_id),
        )
        .order_by(Job.created_at.desc())
        .first()
    )

    # Page list — bounded, default to manual rows (the ones the user can
    # delete). Sitemap detail page can come later if needed.
    q = (
        db.query(ClientBrandPage)
        .filter(ClientBrandPage.client_brand_id == brand_id)
        .order_by(ClientBrandPage.first_seen_at.asc())
    )
    if source in ("manual", "sitemap"):
        q = q.filter(ClientBrandPage.source == source)
    elif source is None:
        # Default to manual-only for the UI list (sitemap rows live in stats)
        q = q.filter(ClientBrandPage.source == "manual")
    if status:
        q = q.filter(ClientBrandPage.status == status)
    q = q.limit(max(1, min(int(limit), 500)))
    page_rows = q.all()

    return {
        "ok": True,
        "brand": {"id": str(brand.id), "name": brand.name, "domain": brand.domain},
        "stats": {
            "total": total,
            "by_status": by_status,
            "by_source": by_source,
            "last_seen_at": last_seen_max.isoformat() if last_seen_max else None,
            "last_crawled_at": last_crawled_max.isoformat() if last_crawled_max else None,
        },
        "in_flight_job": {
            "id": str(in_flight_crawl.id),
            "type": in_flight_crawl.job_type,
            "status": in_flight_crawl.status,
        } if in_flight_crawl else None,
        "pages": [
            {
                "id": str(p.id),
                "url": p.url,
                "title": p.title,
                "status": p.status,
                "source": p.source,
                "fetch_error": p.fetch_error,
                "lastmod": p.lastmod.isoformat() if p.lastmod else None,
                "last_crawled_at": p.last_crawled_at.isoformat() if p.last_crawled_at else None,
            }
            for p in page_rows
        ],
    }


class ManualPageUrlBody(BaseModel):
    url: str


@router.post("/{client_id}/brands/{brand_id}/pages/manual")
@limiter.limit("20/minute")
async def add_manual_page(
    request: Request, client_id: str, brand_id: str, body: ManualPageUrlBody,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a URL to the brand's page index. Goes into the fetch -> embed
    pipeline like any sitemap-discovered row, but with source='manual'
    so it survives sitemap diffs.

    Enqueues a fetch_brand_pages job if one isn't already in flight for
    this brand — same in-flight pattern as trust-sources discovery.
    """
    _check_client_access(client_id, user, db)
    brand = _brand_for_client(brand_id, client_id, db)
    canonical_url = _validate_manual_page_url(body.url, brand)

    existing = (
        db.query(ClientBrandPage)
        .filter(
            ClientBrandPage.client_brand_id == brand_id,
            ClientBrandPage.url == canonical_url,
        )
        .first()
    )
    if existing:
        # If it was previously sitemap-discovered then went gone, restore
        # via the user's manual intent. Otherwise 409 — the row already
        # exists and the user can see it in the list.
        if existing.status == "gone":
            existing.status = "pending_fetch"
            existing.gone_since = None
            existing.fetch_error = None
            existing.fetch_retry_count = 0
            existing.source = "manual"
            db.commit()
            _maybe_enqueue_fetch(client_id, brand_id, db)
            return {
                "ok": True, "url": canonical_url, "status": existing.status,
                "source": "manual", "restored": True,
            }
        raise HTTPException(409, {
            "error": "duplicate",
            "message": f"This URL is already indexed (status={existing.status}, source={existing.source}).",
        })

    row = ClientBrandPage(
        client_brand_id=brand_id,
        url=canonical_url,
        status="pending_fetch",
        source="manual",
    )
    db.add(row)
    db.commit()

    _maybe_enqueue_fetch(client_id, brand_id, db)

    return {
        "ok": True, "url": canonical_url, "status": "pending_fetch",
        "source": "manual", "restored": False,
    }


@router.delete("/{client_id}/brands/{brand_id}/pages/manual")
async def delete_manual_page(
    client_id: str, brand_id: str, body: ManualPageUrlBody,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Hard-delete a manual page. Sitemap-discovered rows cannot be
    removed this way (would be re-inserted by the next crawl) — they
    only leave the index via the natural gone -> purge_stale_pages flow.
    """
    _check_client_access(client_id, user, db)
    _brand_for_client(brand_id, client_id, db)
    target_url = (body.url or "").strip()
    if not target_url:
        raise HTTPException(400, "URL is required.")

    row = (
        db.query(ClientBrandPage)
        .filter(
            ClientBrandPage.client_brand_id == brand_id,
            ClientBrandPage.url == target_url,
        )
        .first()
    )
    if not row:
        raise HTTPException(404, "Page not found in this brand's index.")
    if row.source != "manual":
        raise HTTPException(
            409,
            {"error": "not_manual",
             "message": "Only manually-added pages can be deleted. Sitemap-discovered "
                        "pages must be removed from your sitemap.xml or wait for the "
                        "30-day gone-purge cycle."},
        )

    db.delete(row)
    db.commit()
    return {"ok": True, "deleted": target_url}


def _maybe_enqueue_fetch(client_id: str, brand_id: str, db: Session) -> str | None:
    """Enqueue a fetch_brand_pages job iff none is already pending/running.

    Mirrors the discover_trust_sources in-flight protection — a single
    pending fetch will pick up every pending_fetch row at runtime, so
    queueing more is wasted work."""
    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "fetch_brand_pages",
            Job.status.in_(("pending", "running")),
            Job.payload["client_brand_id"].astext == str(brand_id),
        )
        .first()
    )
    if in_flight:
        return str(in_flight.id)
    job = Job(
        client_id=client_id,
        job_type="fetch_brand_pages",
        status="pending",
        payload={"client_brand_id": str(brand_id)},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    return str(job.id)


@router.post("/{client_id}/brands/{brand_id}/sitemap/refresh")
@limiter.limit("3/minute")
async def refresh_brand_sitemap(
    request: Request, client_id: str, brand_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Manually trigger a sitemap re-crawl for one brand. Enqueues
    crawl_brand_sitemap, which chains fetch_brand_pages on completion
    (Day 2). In-flight protection : if a crawl is already pending /
    running for this brand we return its job id."""
    _check_client_access(client_id, user, db)
    brand = _brand_for_client(brand_id, client_id, db)

    in_flight = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "crawl_brand_sitemap",
            Job.status.in_(("pending", "running")),
            Job.payload["client_brand_id"].astext == str(brand_id),
        )
        .first()
    )
    if in_flight:
        return {
            "ok": True, "job_id": str(in_flight.id),
            "status": in_flight.status, "message": "Already in flight",
        }

    if not (brand.domain or "").strip():
        raise HTTPException(
            422,
            {"error": "brand_domain_missing",
             "message": "This brand has no registered domain — cannot crawl. "
                        "Set it in Settings → My primary brands first."},
        )

    job = Job(
        client_id=client_id,
        job_type="crawl_brand_sitemap",
        status="pending",
        payload={"client_brand_id": str(brand_id)},
        max_attempts=2,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"ok": True, "job_id": str(job.id), "status": "pending"}


@router.delete("/{client_id}/trust-sources/extra-domains/{domain}")
async def remove_trust_source_extra(client_id: str, domain: str,
                                     user=Depends(get_current_user),
                                     db: Session = Depends(get_db)):
    """Remove a user-managed domain from the prefer-hint list."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    normalized = _normalize_extra_domain(domain)
    if not normalized:
        raise HTTPException(400, "Invalid domain")

    apps = dict(client.apps or {})
    trust = dict(apps.get("trust_sources") or {})
    extras = list(trust.get("extra_domains") or [])
    if normalized not in extras:
        raise HTTPException(404, f"{normalized} not in extras")
    extras = [d for d in extras if d != normalized]
    trust["extra_domains"] = extras
    apps["trust_sources"] = trust
    client.apps = apps
    flag_modified(client, "apps")
    db.commit()
    return {"ok": True, "extra_domains": extras}
