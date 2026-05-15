from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from config import settings
from models import Base, engine
from routers import admin, audit_requests, auth, clients, content_items, invitations, oauth, organizations, reports, stripe, scans, brands
from services.rate_limit import limiter
from services.request_context import current_request_method

app = FastAPI(
    title="sen-ai API",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# H2: rate limiting (slowapi). Limiter instance lives on app.state so per-route
# decorators can find it; the middleware enforces default_limits, and the
# exception handler converts RateLimitExceeded into a 429 with a clean body.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_method_middleware(request: Request, call_next):
    """H6: store the HTTP method in a contextvar so internal helpers
    (notably `_check_scan_access`) can auto-escalate role requirements
    on destructive methods without having to thread `Request` through
    every signature."""
    token = current_request_method.set(request.method)
    try:
        return await call_next(request)
    finally:
        current_request_method.reset(token)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(clients.router, prefix="/api/clients", tags=["clients"])
app.include_router(stripe.router, prefix="/api/stripe", tags=["stripe"])
app.include_router(scans.router, prefix="/api/scans", tags=["scans"])
app.include_router(brands.router, prefix="/api/clients", tags=["brands"])
app.include_router(oauth.router, prefix="/api/oauth", tags=["oauth"])
app.include_router(reports.router, prefix="/api/admin/reports", tags=["admin-reports"])
app.include_router(audit_requests.router, prefix="/api/audit-requests", tags=["audit-requests"])
app.include_router(content_items.router, prefix="/api", tags=["content-items"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(organizations.router, prefix="/api/organizations", tags=["organizations"])
app.include_router(invitations.org_scoped_router, prefix="/api/organizations", tags=["invitations"])
app.include_router(invitations.token_scoped_router, prefix="/api/invitations", tags=["invitations"])


@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
