"""healthchecks.io ping helper for worker liveness monitoring.

We use two separate check URLs :
  - HEALTHCHECK_WORKER_URL  — pinged every HEARTBEAT_INTERVAL_SECONDS from the
    main poll loop. If the worker is dead/hung/crashed without restarting,
    healthchecks.io alerts after the grace period configured on their side.
  - HEALTHCHECK_T14_URL     — pinged at the end of each
    enqueue_post_publish_measurements() sweep (so once per hour in steady
    state). Expected cadence is ~24/day ; configure the check on healthchecks
    to alert if it goes silent for > 4h.

Both URLs are optional — if env var is empty, ping_* is a no-op (mirrors the
SENTRY_DSN / RESEND_API_KEY pattern). Calling either function MUST NEVER
block or raise — these are observability side-effects, not load-bearing.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

# 5 min — matches the user's brief. healthchecks.io free tier configures the
# grace period on their side (e.g. "expect a ping every 5 min, alert after
# 10 min of silence"), so we just need to ping at least that often.
HEARTBEAT_INTERVAL_SECONDS = 300
_LAST_HEARTBEAT_TS = 0.0

# Short timeout — if the request takes > 3s, something is wrong with the
# external service, not with us. Better to skip the ping than block the
# poll loop.
_PING_TIMEOUT_SECONDS = 3.0


def _safe_get(url: str) -> None:
    """Fire-and-forget GET. Swallows every exception (network, DNS, timeout).

    healthchecks.io semantics : any 2xx response means "I'm alive". We don't
    care about the response body and we don't retry — the next interval will
    try again.
    """
    try:
        httpx.get(url, timeout=_PING_TIMEOUT_SECONDS)
    except Exception as e:
        # Debug-level because failures here are expected in dev and on
        # transient network blips ; not actionable.
        logger.debug(f"healthcheck ping failed: {e}")


def ping_heartbeat() -> None:
    """Ping the worker liveness check, throttled to HEARTBEAT_INTERVAL_SECONDS.

    Called every iteration of the main poll loop ; the throttle prevents
    spamming healthchecks.io when the worker has work and iterates rapidly.
    """
    global _LAST_HEARTBEAT_TS
    url = (os.environ.get("HEALTHCHECK_WORKER_URL") or "").strip()
    if not url:
        return
    now = time.time()
    if now - _LAST_HEARTBEAT_TS < HEARTBEAT_INTERVAL_SECONDS:
        return
    _LAST_HEARTBEAT_TS = now
    _safe_get(url)


def ping_t14_sweep() -> None:
    """Ping the T+14 post-publish cron check.

    Called by `enqueue_post_publish_measurements()` after each sweep
    completes (whether or not it found ripe items — the point is "the cron
    ran"). No throttle here : the caller is already throttled to 1h.
    """
    url = (os.environ.get("HEALTHCHECK_T14_URL") or "").strip()
    if not url:
        return
    _safe_get(url)
