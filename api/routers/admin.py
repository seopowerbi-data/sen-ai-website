"""M5: Superadmin-only routes for platform operations.

Includes:
  * Client / user listing + feature flag toggling
  * Platform monitoring dashboard (users, scans, credits, modules)
  * LLM API cost monitoring (Anthropic, OpenAI, Gemini)
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import case, cast, func, Date, text
from sqlalchemy.orm import Session

from models import (
    AuditLog,
    AuditRequest,
    Client,
    ClientCredit,
    LlmUsageLog,
    OAuthConnection,
    Scan,
    ScanLLMResult,
    User,
    UserClient,
    get_db,
)
from services.audit import audit_log
from services.auth_service import get_current_user

router = APIRouter()


def require_superadmin(user: User = Depends(get_current_user)) -> User:
    """Dependency that 403s any non-superadmin user.

    Cheap (no extra DB hit — `User` is already loaded by `get_current_user`).
    """
    if not user.is_superadmin:
        raise HTTPException(403, "Superadmin access required")
    return user


@router.get("/clients")
async def admin_list_clients(
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """List every client on the platform with summary stats.

    Returns: id, name, brand, created_at, member_count, scan_count.
    Sorted by created_at desc (newest first).
    """
    rows = (
        db.query(
            Client.id,
            Client.name,
            Client.brand,
            Client.created_at,
            func.count(func.distinct(UserClient.user_id)).label("member_count"),
            func.count(func.distinct(Scan.id)).label("scan_count"),
        )
        .outerjoin(UserClient, UserClient.client_id == Client.id)
        .outerjoin(Scan, Scan.client_id == Client.id)
        .group_by(Client.id)
        .order_by(Client.created_at.desc())
        .all()
    )
    # Fetch apps for each client (not available in the aggregated query)
    client_ids = [r.id for r in rows]
    clients_map = {}
    if client_ids:
        clients_full = db.query(Client).filter(Client.id.in_(client_ids)).all()
        clients_map = {c.id: c for c in clients_full}

    return [
        {
            "id": str(r.id),
            "name": r.name,
            "brand": r.brand,
            "apps": clients_map[r.id].apps if r.id in clients_map else {},
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "member_count": r.member_count,
            "scan_count": r.scan_count,
        }
        for r in rows
    ]


@router.get("/users")
async def admin_list_users(
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """List every user on the platform with their client memberships.

    Returns: id, email, name, is_superadmin, auth methods, created_at,
    list of (client_id, client_name, role) for each membership.
    """
    users = db.query(User).order_by(User.created_at.desc()).all()
    out = []
    for u in users:
        links = (
            db.query(UserClient, Client)
            .join(Client, Client.id == UserClient.client_id)
            .filter(UserClient.user_id == u.id)
            .all()
        )
        out.append({
            "id": str(u.id),
            "email": u.email,
            "name": u.name,
            "is_superadmin": bool(u.is_superadmin),
            "auth": {
                "password": u.password_hash is not None,
                "google_oauth": u.google_id is not None,
            },
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "memberships": [
                {
                    "client_id": str(c.id),
                    "client_name": c.name,
                    "role": link.role,
                }
                for link, c in links
            ],
        })
    return out


VALID_APP_KEYS = {"ai_scan", "local_business"}


class AppToggleRequest(BaseModel):
    app_key: str          # e.g. "ai_scan"
    enabled: bool = True
    config: dict | None = None  # optional app-specific config


@router.patch("/clients/{client_id}/apps")
async def admin_toggle_app(
    client_id: str,
    req: AppToggleRequest,
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Toggle an app module on/off for a client.

    Sets or removes an app key in the client's `apps` JSONB column.
    When enabled=false, the key is removed entirely (clean sidebar).
    """
    if req.app_key not in VALID_APP_KEYS:
        raise HTTPException(400, f"Unknown app: {req.app_key}. Valid: {sorted(VALID_APP_KEYS)}")

    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    apps = dict(client.apps or {})
    if req.enabled:
        app_entry = {"enabled": True}
        if req.config:
            app_entry.update(req.config)
        apps[req.app_key] = app_entry
    else:
        apps.pop(req.app_key, None)

    client.apps = apps
    # Force SQLAlchemy to detect JSONB mutation
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(client, "apps")
    db.commit()

    return {"client_id": str(client.id), "apps": client.apps}


@router.get("/clients/{client_id}/connections")
async def admin_client_connections(
    client_id: str,
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """List OAuth connections for a specific client (superadmin)."""
    rows = (
        db.query(OAuthConnection)
        .filter(OAuthConnection.client_id == client_id)
        .order_by(OAuthConnection.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(c.id),
            "provider": c.provider,
            "product": c.product,
            "account_email": c.account_email,
            "status": c.status,
            "authorized_at": c.authorized_at.isoformat() if c.authorized_at else None,
        }
        for c in rows
    ]


@router.get("/me")
async def admin_whoami(user: User = Depends(require_superadmin)):
    """Confirm the caller is a superadmin. Useful for the admin UI to
    decide whether to render the /app/admin/ navigation entry."""
    return {
        "id": str(user.id),
        "email": user.email,
        "is_superadmin": True,
    }


# ── Platform Monitoring Dashboard ──────────────────────────────────────


@router.get("/dashboard")
async def admin_dashboard(
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Platform overview — single request for the admin monitoring dashboard.

    Returns aggregated stats: users, clients, scans, credits, modules, LLM usage.
    """
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    # ── Users ──────────────────────────────────────────────────────────
    total_users = db.query(func.count(User.id)).scalar() or 0
    verified_users = db.query(func.count(User.id)).filter(User.is_email_verified == True).scalar() or 0
    superadmin_count = db.query(func.count(User.id)).filter(User.is_superadmin == True).scalar() or 0
    users_today = db.query(func.count(User.id)).filter(User.created_at >= today).scalar() or 0
    users_this_week = db.query(func.count(User.id)).filter(User.created_at >= week_ago).scalar() or 0
    users_this_month = db.query(func.count(User.id)).filter(User.created_at >= month_ago).scalar() or 0

    # ── Clients ────────────────────────────────────────────────────────
    total_clients = db.query(func.count(Client.id)).scalar() or 0

    # Count clients with scans
    clients_with_scans = (
        db.query(func.count(func.distinct(Scan.client_id)))
        .filter(Scan.status == "completed")
        .scalar() or 0
    )

    # Count clients per module (from apps JSONB)
    all_clients = db.query(Client.apps).all()
    module_counts = {"ai_scan": 0, "local_business": 0}
    for (apps,) in all_clients:
        if apps:
            for key in module_counts:
                if apps.get(key, {}).get("enabled"):
                    module_counts[key] += 1

    # ── Scans ──────────────────────────────────────────────────────────
    scan_stats = (
        db.query(
            func.count(Scan.id).label("total"),
            func.count(case((Scan.status == "completed", 1))).label("completed"),
            func.count(case((Scan.status == "failed", 1))).label("failed"),
            func.count(case((Scan.status.in_(["scanning", "fetching_keywords", "generating_personas"]), 1))).label("in_progress"),
        )
        .first()
    )
    scans_this_month = (
        db.query(func.count(Scan.id))
        .filter(Scan.created_at >= month_ago)
        .scalar() or 0
    )

    # ── Credits ────────────────────────────────────────────────────────
    credit_agg = (
        db.query(
            func.sum(case((ClientCredit.amount > 0, ClientCredit.amount), else_=0)).label("total_purchased"),
            func.sum(case((ClientCredit.amount < 0, func.abs(ClientCredit.amount)), else_=0)).label("total_consumed"),
        )
        .filter(ClientCredit.credit_type == "scan")
        .first()
    )
    total_purchased = int(credit_agg.total_purchased or 0)
    total_consumed = int(credit_agg.total_consumed or 0)

    # Revenue from Stripe (scan packs only — count distinct stripe_session_ids)
    stripe_purchases = (
        db.query(func.count(func.distinct(ClientCredit.stripe_session_id)))
        .filter(ClientCredit.stripe_session_id.isnot(None))
        .scalar() or 0
    )

    # ── LLM Usage (from llm_usage_log) ─────────────────────────────────
    llm_agg = (
        db.query(
            func.count(LlmUsageLog.id).label("total_calls"),
            func.coalesce(func.sum(LlmUsageLog.input_tokens + LlmUsageLog.output_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(LlmUsageLog.cost_usd), 0).label("total_cost"),
        )
        .first()
    )

    # LLM by provider
    llm_by_provider = (
        db.query(
            LlmUsageLog.provider,
            func.count(LlmUsageLog.id).label("calls"),
            func.coalesce(func.sum(LlmUsageLog.input_tokens + LlmUsageLog.output_tokens), 0).label("tokens"),
            func.coalesce(func.sum(LlmUsageLog.cost_usd), 0).label("cost_usd"),
        )
        .group_by(LlmUsageLog.provider)
        .all()
    )

    # Also aggregate from historical scan_llm_results (pre-migration data)
    legacy_llm = (
        db.query(
            ScanLLMResult.provider,
            func.count(ScanLLMResult.id).label("calls"),
            func.coalesce(func.sum(ScanLLMResult.input_tokens + ScanLLMResult.output_tokens), 0).label("tokens"),
        )
        .group_by(ScanLLMResult.provider)
        .all()
    )

    # ── Recent activity (last 10 audit events) ─────────────────────────
    recent_events = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(10)
        .all()
    )

    # ── Recent registrations (last 15 users) ───────────────────────────
    recent_users = (
        db.query(User.id, User.email, User.name, User.is_email_verified, User.created_at)
        .order_by(User.created_at.desc())
        .limit(15)
        .all()
    )

    return {
        "users": {
            "total": total_users,
            "verified": verified_users,
            "superadmin": superadmin_count,
            "registered_today": users_today,
            "registered_this_week": users_this_week,
            "registered_this_month": users_this_month,
        },
        "clients": {
            "total": total_clients,
            "with_scans": clients_with_scans,
            "modules": module_counts,
        },
        "scans": {
            "total": scan_stats.total if scan_stats else 0,
            "completed": scan_stats.completed if scan_stats else 0,
            "failed": scan_stats.failed if scan_stats else 0,
            "in_progress": scan_stats.in_progress if scan_stats else 0,
            "this_month": scans_this_month,
        },
        "credits": {
            "total_purchased": total_purchased,
            "total_consumed": total_consumed,
            "total_balance": total_purchased - total_consumed,
            "stripe_purchases": stripe_purchases,
        },
        "llm_usage": {
            "total_calls": int(llm_agg.total_calls or 0),
            "total_tokens": int(llm_agg.total_tokens or 0),
            "total_cost_usd": round(float(llm_agg.total_cost or 0), 4),
            "by_provider": {
                row.provider: {
                    "calls": int(row.calls),
                    "tokens": int(row.tokens),
                    "cost_usd": round(float(row.cost_usd), 4),
                }
                for row in llm_by_provider
            },
            "legacy_scan_results": {
                row.provider: {
                    "calls": int(row.calls),
                    "tokens": int(row.tokens),
                }
                for row in legacy_llm
            },
        },
        "recent_events": [
            {
                "action": e.action,
                "target_type": e.target_type,
                "target_id": e.target_id,
                "ip": e.ip_address,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in recent_events
        ],
        "recent_users": [
            {
                "id": str(u.id),
                "email": u.email,
                "name": u.name,
                "verified": bool(u.is_email_verified),
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in recent_users
        ],
    }


@router.get("/llm-usage")
async def admin_llm_usage(
    days: int = Query(30, ge=1, le=365),
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Detailed LLM API usage for cost monitoring.

    Returns daily breakdown, per-model costs, and per-operation costs
    over the requested period (default 30 days).
    """
    since = datetime.utcnow() - timedelta(days=days)

    # ── Daily breakdown ────────────────────────────────────────────────
    daily = (
        db.query(
            cast(LlmUsageLog.created_at, Date).label("date"),
            LlmUsageLog.provider,
            func.count(LlmUsageLog.id).label("calls"),
            func.sum(LlmUsageLog.input_tokens).label("input_tokens"),
            func.sum(LlmUsageLog.output_tokens).label("output_tokens"),
            func.sum(LlmUsageLog.cost_usd).label("cost_usd"),
            func.count(case((LlmUsageLog.error == True, 1))).label("errors"),
        )
        .filter(LlmUsageLog.created_at >= since)
        .group_by(cast(LlmUsageLog.created_at, Date), LlmUsageLog.provider)
        .order_by(cast(LlmUsageLog.created_at, Date))
        .all()
    )

    # ── By model ───────────────────────────────────────────────────────
    by_model = (
        db.query(
            LlmUsageLog.provider,
            LlmUsageLog.model,
            func.count(LlmUsageLog.id).label("calls"),
            func.sum(LlmUsageLog.input_tokens).label("input_tokens"),
            func.sum(LlmUsageLog.output_tokens).label("output_tokens"),
            func.sum(LlmUsageLog.cost_usd).label("cost_usd"),
        )
        .filter(LlmUsageLog.created_at >= since)
        .group_by(LlmUsageLog.provider, LlmUsageLog.model)
        .order_by(func.sum(LlmUsageLog.cost_usd).desc())
        .all()
    )

    # ── By operation ───────────────────────────────────────────────────
    by_operation = (
        db.query(
            LlmUsageLog.operation,
            func.count(LlmUsageLog.id).label("calls"),
            func.sum(LlmUsageLog.input_tokens).label("input_tokens"),
            func.sum(LlmUsageLog.output_tokens).label("output_tokens"),
            func.sum(LlmUsageLog.cost_usd).label("cost_usd"),
        )
        .filter(LlmUsageLog.created_at >= since)
        .group_by(LlmUsageLog.operation)
        .order_by(func.sum(LlmUsageLog.cost_usd).desc())
        .all()
    )

    # ── By client (top 20 spenders) ────────────────────────────────────
    by_client = (
        db.query(
            LlmUsageLog.client_id,
            Client.name.label("client_name"),
            func.count(LlmUsageLog.id).label("calls"),
            func.sum(LlmUsageLog.cost_usd).label("cost_usd"),
        )
        .outerjoin(Client, Client.id == LlmUsageLog.client_id)
        .filter(LlmUsageLog.created_at >= since)
        .filter(LlmUsageLog.client_id.isnot(None))
        .group_by(LlmUsageLog.client_id, Client.name)
        .order_by(func.sum(LlmUsageLog.cost_usd).desc())
        .limit(20)
        .all()
    )

    # ── Totals for period ──────────────────────────────────────────────
    totals = (
        db.query(
            func.count(LlmUsageLog.id).label("calls"),
            func.coalesce(func.sum(LlmUsageLog.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(LlmUsageLog.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(LlmUsageLog.cost_usd), 0).label("cost_usd"),
            func.count(case((LlmUsageLog.error == True, 1))).label("errors"),
        )
        .filter(LlmUsageLog.created_at >= since)
        .first()
    )

    return {
        "period_days": days,
        "totals": {
            "calls": int(totals.calls or 0),
            "input_tokens": int(totals.input_tokens or 0),
            "output_tokens": int(totals.output_tokens or 0),
            "cost_usd": round(float(totals.cost_usd or 0), 4),
            "errors": int(totals.errors or 0),
        },
        "daily": [
            {
                "date": str(row.date),
                "provider": row.provider,
                "calls": int(row.calls),
                "input_tokens": int(row.input_tokens or 0),
                "output_tokens": int(row.output_tokens or 0),
                "cost_usd": round(float(row.cost_usd or 0), 4),
                "errors": int(row.errors or 0),
            }
            for row in daily
        ],
        "by_model": [
            {
                "provider": row.provider,
                "model": row.model,
                "calls": int(row.calls),
                "input_tokens": int(row.input_tokens or 0),
                "output_tokens": int(row.output_tokens or 0),
                "cost_usd": round(float(row.cost_usd or 0), 4),
            }
            for row in by_model
        ],
        "by_operation": [
            {
                "operation": row.operation,
                "calls": int(row.calls),
                "input_tokens": int(row.input_tokens or 0),
                "output_tokens": int(row.output_tokens or 0),
                "cost_usd": round(float(row.cost_usd or 0), 4),
            }
            for row in by_operation
        ],
        "by_client": [
            {
                "client_id": str(row.client_id) if row.client_id else None,
                "client_name": row.client_name,
                "calls": int(row.calls),
                "cost_usd": round(float(row.cost_usd or 0), 4),
            }
            for row in by_client
        ],
    }


@router.get("/users/activity")
async def admin_users_activity(
    days: int = Query(90, ge=1, le=365),
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """User registration timeline — daily counts for trend chart."""
    since = datetime.utcnow() - timedelta(days=days)

    daily = (
        db.query(
            cast(User.created_at, Date).label("date"),
            func.count(User.id).label("count"),
        )
        .filter(User.created_at >= since)
        .group_by(cast(User.created_at, Date))
        .order_by(cast(User.created_at, Date))
        .all()
    )

    return [
        {"date": str(row.date), "count": int(row.count)}
        for row in daily
    ]


@router.get("/credits/overview")
async def admin_credits_overview(
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Platform-wide credit overview — per-client balances + top consumers."""
    # Current balance per client (latest balance_after per credit_type)
    from sqlalchemy import desc

    # Get current scan credit balance per client
    subq = (
        db.query(
            ClientCredit.client_id,
            ClientCredit.credit_type,
            ClientCredit.balance_after,
            func.row_number().over(
                partition_by=[ClientCredit.client_id, ClientCredit.credit_type],
                order_by=desc(ClientCredit.created_at),
            ).label("rn"),
        )
        .subquery()
    )

    balances = (
        db.query(
            subq.c.client_id,
            Client.name.label("client_name"),
            subq.c.credit_type,
            subq.c.balance_after,
        )
        .join(Client, Client.id == subq.c.client_id)
        .filter(subq.c.rn == 1)
        .order_by(subq.c.balance_after.desc())
        .all()
    )

    # Group by client
    clients_map: dict = {}
    for row in balances:
        cid = str(row.client_id)
        if cid not in clients_map:
            clients_map[cid] = {"client_id": cid, "client_name": row.client_name, "scan": 0, "content": 0}
        clients_map[cid][row.credit_type] = int(row.balance_after)

    # Monthly credit consumption trend
    monthly = (
        db.query(
            func.to_char(ClientCredit.created_at, 'YYYY-MM').label("month"),
            func.sum(case((ClientCredit.amount > 0, ClientCredit.amount), else_=0)).label("purchased"),
            func.sum(case((ClientCredit.amount < 0, func.abs(ClientCredit.amount)), else_=0)).label("consumed"),
        )
        .group_by(func.to_char(ClientCredit.created_at, 'YYYY-MM'))
        .order_by(func.to_char(ClientCredit.created_at, 'YYYY-MM'))
        .all()
    )

    return {
        "clients": list(clients_map.values()),
        "monthly": [
            {
                "month": row.month,
                "purchased": int(row.purchased or 0),
                "consumed": int(row.consumed or 0),
            }
            for row in monthly
        ],
    }


# -----------------------------------------------------------------------------
# 018: Audit-gratuit requests admin endpoints
# -----------------------------------------------------------------------------


class AuditRequestStatusUpdate(BaseModel):
    status: str  # one of: pending|confirmed|launched|completed|rejected
    scan_id: str | None = None


@router.get("/coverage/brand-briefs")
async def admin_brand_brief_coverage(
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Phase BB observability — % of primary brands with a generated brief per workspace.

    A primary brand counts as "briefed" when ``client_brands.brief IS NOT NULL`` —
    either generated via LLM (generations_count > 0) OR seeded via the one-shot
    backfill script. Returns one row per client, ordered worst-coverage first
    so partial workspaces surface at the top.

    See project_phase_brand_briefs.md (coverage metric section).
    """
    rows = db.execute(text(
        """
        SELECT
            c.id::text                                                      AS client_id,
            c.name                                                          AS client_name,
            COUNT(*) FILTER (WHERE cb.brief IS NOT NULL)                    AS briefed,
            COUNT(*) FILTER (WHERE cb.brief IS NOT NULL
                              AND cb.brief_generations_count > 0)           AS briefed_via_llm,
            COUNT(*) FILTER (WHERE cb.brief IS NOT NULL
                              AND (cb.brief_generations_count = 0
                                   OR cb.brief_generations_count IS NULL))  AS seeded_only,
            COUNT(*)                                                        AS total_primary,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE cb.brief IS NOT NULL)
                / NULLIF(COUNT(*), 0),
                1
            )                                                               AS pct_briefed
        FROM clients c
        JOIN client_brands cb
          ON cb.client_id = c.id
         AND cb.id = ANY(c.primary_brand_ids)
        WHERE c.primary_brand_ids IS NOT NULL
        GROUP BY c.id, c.name
        ORDER BY pct_briefed ASC NULLS FIRST, total_primary DESC
        """
    )).fetchall()

    return {
        "rows": [
            {
                "client_id": r.client_id,
                "client_name": r.client_name,
                "briefed": int(r.briefed or 0),
                "briefed_via_llm": int(r.briefed_via_llm or 0),
                "seeded_only": int(r.seeded_only or 0),
                "total_primary": int(r.total_primary or 0),
                "pct_briefed": float(r.pct_briefed) if r.pct_briefed is not None else 0.0,
            }
            for r in rows
        ],
        "totals": {
            "workspaces": len(rows),
            "primary_brands_total": sum(int(r.total_primary or 0) for r in rows),
            "primary_brands_briefed": sum(int(r.briefed or 0) for r in rows),
            "primary_brands_briefed_via_llm": sum(int(r.briefed_via_llm or 0) for r in rows),
        },
    }


@router.get("/audit-requests")
async def list_audit_requests(
    status: str | None = Query(default=None, description="Filter by status"),
    _: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """List audit-gratuit requests with optional status filter.

    Sorted by created_at desc — most recent first. No pagination yet (volume
    expected to be low for MVP).
    """
    q = db.query(AuditRequest)
    if status:
        q = q.filter(AuditRequest.status == status)
    rows = q.order_by(AuditRequest.created_at.desc()).limit(500).all()
    return {
        "items": [
            {
                "id": str(r.id),
                "website": r.website,
                "email": r.email,
                "topic_focus": r.topic_focus,
                "first_name": r.first_name,
                "message": r.message,
                "status": r.status,
                "scan_id": str(r.scan_id) if r.scan_id else None,
                "source_ip": r.source_ip,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "confirmed_at": r.confirmed_at.isoformat() if r.confirmed_at else None,
                "processed_at": r.processed_at.isoformat() if r.processed_at else None,
            }
            for r in rows
        ],
    }


@router.patch("/audit-requests/{audit_id}")
async def update_audit_request(
    audit_id: str,
    update: AuditRequestStatusUpdate,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Update an audit_request status (admin workflow).

    Used to mark a request as launched once the scan is created, or rejected
    if it's spam / out of scope. Setting status to 'launched' or 'completed'
    stamps processed_at; setting 'launched' also accepts an optional scan_id.
    """
    valid = {"pending", "confirmed", "launched", "completed", "rejected"}
    if update.status not in valid:
        raise HTTPException(400, f"Invalid status. Must be one of {valid}")

    audit_req = db.query(AuditRequest).filter(AuditRequest.id == audit_id).first()
    if not audit_req:
        raise HTTPException(404, "Audit request not found")

    audit_req.status = update.status
    if update.status in ("launched", "completed", "rejected"):
        audit_req.processed_at = datetime.utcnow()
    if update.scan_id:
        audit_req.scan_id = update.scan_id

    audit_log(
        db,
        action="audit_request.update",
        user_id=str(user.id),
        target_type="audit_request",
        target_id=str(audit_req.id),
        details={"status": update.status, "scan_id": update.scan_id},
    )
    db.commit()
    return {"ok": True, "status": audit_req.status}
