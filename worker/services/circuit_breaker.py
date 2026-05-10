"""ProviderCircuitBreaker — kill a provider mid-scan when it's catastrophically failing.

Adapted from seo_llm/src/llm_tester.py:54 for the SaaS scan handler. Stateless to
scan_id (one instance per run_llm_tests invocation), thread-safe so the parallel
test loop can update + read state without coordination.

Trip rules (tunable via env):
    CIRCUIT_BREAKER_FAILURE_RATE   default 0.8   — failure rate that counts as "bad"
    CIRCUIT_BREAKER_MIN_SAMPLE     default 10    — don't trip on the first few tests

When a provider trips, the handler cancels all PENDING futures for that provider
(running ones still complete — ThreadPoolExecutor can't kill threads safely).
Skipped tests are surfaced in scan.summary.provider_status so the user sees why
their result count is below total_tests, and refunded prorata via C.3.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)


class ProviderCircuitBreaker:
    """Per-scan tracker for provider success/failure with auto-trip."""

    def __init__(self, providers: list[str],
                 failure_rate: float | None = None,
                 min_sample: int | None = None):
        self.failure_rate = (
            failure_rate if failure_rate is not None
            else float(os.getenv("CIRCUIT_BREAKER_FAILURE_RATE", "0.8"))
        )
        self.min_sample = (
            min_sample if min_sample is not None
            else int(os.getenv("CIRCUIT_BREAKER_MIN_SAMPLE", "10"))
        )
        self._lock = threading.Lock()
        self._stats: dict[str, dict] = {
            p: {"success": 0, "failure": 0, "skipped": 0, "tripped": False}
            for p in providers
        }

    def record_success(self, provider: str) -> None:
        with self._lock:
            if provider in self._stats:
                self._stats[provider]["success"] += 1

    def record_failure(self, provider: str) -> None:
        with self._lock:
            if provider in self._stats:
                self._stats[provider]["failure"] += 1

    def record_skip(self, provider: str) -> None:
        """Called for each future cancelled when the provider trips."""
        with self._lock:
            if provider in self._stats:
                self._stats[provider]["skipped"] += 1

    def is_tripped(self, provider: str) -> bool:
        with self._lock:
            return self._stats.get(provider, {}).get("tripped", False)

    def maybe_trip(self, provider: str) -> bool:
        """Check if `provider` should trip now. Returns True if NEWLY tripped.

        A provider trips when failure_rate >= threshold AND we have at least
        min_sample tests recorded. Idempotent: subsequent calls return False.
        """
        with self._lock:
            s = self._stats.get(provider)
            if not s or s["tripped"]:
                return False
            total = s["success"] + s["failure"]
            if total < self.min_sample:
                return False
            fr = s["failure"] / total
            if fr < self.failure_rate:
                return False
            s["tripped"] = True
            logger.warning(
                f"Circuit breaker: provider '{provider}' tripped "
                f"({s['failure']}/{total} = {fr:.0%} failure rate, "
                f"threshold={self.failure_rate:.0%})"
            )
            return True

    def to_dict(self) -> dict[str, dict]:
        """Snapshot for persistence in scan.summary.provider_status."""
        with self._lock:
            return {p: dict(s) for p, s in self._stats.items()}

    def any_tripped(self) -> bool:
        with self._lock:
            return any(s["tripped"] for s in self._stats.values())
