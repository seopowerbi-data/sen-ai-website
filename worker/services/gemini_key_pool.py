"""Gemini API key pool — round-robin rotation with cooldown on 429.

Each Gemini API key = 1 GCP project = independent quota (~1500-2000 RPD on free tier).
With N keys, we multiply throughput by N. Rotation is round-robin; on a 429 we put
the offending key in cooldown for `GEMINI_COOLDOWN_SECONDS` (default 30s) so
subsequent calls skip it.

Configuration (from worker env, comma-separated takes precedence) :
    GEMINI_API_KEYS=key_a,key_b,key_c    → pool of 3
    GEMINI_API_KEY=key_a                 → pool of 1 (back-compat)
    (neither set)                        → empty pool, has_keys() returns False

Usage in handlers ::

    from services.gemini_key_pool import get_gemini_pool

    pool = get_gemini_pool()
    if not pool.has_keys():
        return  # caller skips Gemini-dependent step
    key = pool.next_key()
    try:
        client = LLMClient(provider="gemini", api_key=key, model=...)
        client.generate(...)
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            pool.mark_rate_limited(key)
        raise

Ported from worker/seo_llm/src/gemini_key_rotator.py — adapted to return key strings
(rather than pre-built `genai.Client` objects) so it composes with the existing
LLMClient(api_key=...) pattern used in aiscan handlers without a refactor.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


class GeminiKeyPool:
    """Round-robin pool of Gemini API keys with per-key cooldown on rate-limit."""

    def __init__(self, api_keys: list[str] | None = None,
                 cooldown_seconds: int | None = None):
        if api_keys is None:
            keys_csv = os.getenv("GEMINI_API_KEYS", "").strip()
            if keys_csv:
                api_keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
            else:
                single = os.getenv("GEMINI_API_KEY", "").strip()
                api_keys = [single] if single else []

        self._keys: list[str] = api_keys or []
        self._cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else int(os.getenv("GEMINI_COOLDOWN_SECONDS", "30"))
        )
        self._cooldowns: dict[int, float] = {}  # idx -> cooldown_until_ts
        self._next_idx = 0
        self._lock = threading.Lock()

        if self._keys:
            logger.info(
                f"GeminiKeyPool initialized: {len(self._keys)} key(s), "
                f"cooldown={self._cooldown_seconds}s"
            )
        else:
            logger.info("GeminiKeyPool initialized empty (no GEMINI_API_KEY[S] set)")

    @property
    def num_keys(self) -> int:
        return len(self._keys)

    def has_keys(self) -> bool:
        return bool(self._keys)

    def next_key(self) -> str:
        """Return the next available key (round-robin, skipping cooldowns).

        If all keys are in cooldown, returns the one expiring soonest — the caller
        will likely 429 again and that's fine: the alternative (sleep) blocks the
        worker, the alternative (raise) is a worse experience than a soft retry.
        """
        if not self._keys:
            raise RuntimeError("GeminiKeyPool is empty — set GEMINI_API_KEY or GEMINI_API_KEYS")

        with self._lock:
            now = time.time()
            expired = [i for i, t in self._cooldowns.items() if now >= t]
            for i in expired:
                del self._cooldowns[i]
                logger.info(f"Gemini key #{i + 1} out of cooldown")

            for _ in range(len(self._keys)):
                idx = self._next_idx % len(self._keys)
                self._next_idx += 1
                if idx not in self._cooldowns:
                    return self._keys[idx]

            earliest_idx = min(self._cooldowns, key=self._cooldowns.get)
            wait_s = self._cooldowns[earliest_idx] - now
            logger.warning(
                f"All Gemini keys in cooldown — returning key #{earliest_idx + 1} "
                f"(available in {wait_s:.0f}s)"
            )
            return self._keys[earliest_idx]

    def mark_rate_limited(self, key: str) -> None:
        """Put `key` in cooldown for `cooldown_seconds`. No-op if key not in pool."""
        with self._lock:
            try:
                idx = self._keys.index(key)
            except ValueError:
                return
            self._cooldowns[idx] = time.time() + self._cooldown_seconds
            active = len(self._keys) - len(self._cooldowns)
            logger.warning(
                f"Gemini key #{idx + 1}/{len(self._keys)} cooldown "
                f"({self._cooldown_seconds}s) — {active} key(s) still active"
            )


# ── singleton ─────────────────────────────────────────────────────────
_pool: GeminiKeyPool | None = None
_pool_lock = threading.Lock()


def get_gemini_pool() -> GeminiKeyPool:
    """Lazy singleton — first call reads env, subsequent calls reuse."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = GeminiKeyPool()
    return _pool
