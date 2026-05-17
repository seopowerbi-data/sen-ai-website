"""Handler: generate netlinking article (long-form GEO content) for one ScanContentItem.

Wraps `seo_llm.src.geo_content_generator.GEOContentGenerator` — the 12 680-line
Pierre-Fabre-derived pipeline that scores SOSEO/DSEO via YTG, grounds via Serper,
generates section-by-section, validates RAPP 7-criteria, runs rephrase loops on
quality < 8/10, and post-processes Schema.org HTML.

Two responsibilities :

1. **Vertical-agnostic injection layer** — the seo_llm CLI hardcodes 6 PF-specific
   constants (BRAND_MAP, PATHOLOGY_PF_BRANDS, GAMME_TO_SITE, INSTITUTIONAL_URLS,
   BRAND_EXPERT_SECTIONS, PATHOLOGY_KEYWORDS) and 3 helpers
   (_extract_brand_from_source, _get_pf_brands_for_pathology, _build_brand_fallback).
   Per `feedback_no_hardcoded_vertical.md`, our SaaS wrapper must derive everything
   from per-client DB data at runtime. `_PatchedModuleFns` shadows those names at
   the module level for the duration of one call, restoring them on exit. Other
   PF-knowledge constants (consumed by methods we don't override) degrade gracefully
   on non-PF clients via the empty-dict shadow.

2. **SaaS lifecycle integration** — credit-gated execution (3 content_credits debit
   at API, refund on permanent fail), per-client LLM budget cap circuit breaker
   ($0.30 projection), brand promotion via BrandResolver (workspace primary +
   per-item LEAD override), trust sources & competitor denylist injection,
   workspace brief context, audit log, progress field for UI polling (5 phases).

Console silencing : the seo_llm module uses rich.console.Console.print in ~209
places (CLI ancestor). `_silence_rich_console` swaps the module's `console`
object for one writing to devnull during the call — our structured logger is
unaffected.

Output goes to ScanContentItem columns : `content_html` (Schema.org wrapper),
`content_text` (plain), `article_outline` (JSON H2 sections), `content_metadata`
(quality_score, validation_verdict, ytg_soseo, ytg_dseo, sources_used, etc.).
No SharePoint, no GDoc upload, no image — SaaS owns the artifact in DB.

See plan : `~/.claude/plans/twinkling-strolling-turtle.md`
Quality baseline : `project_phase_c1_article_handler.md` (post-smoke memo).
Refund pairing : worker/main.py `_refund_content_item_credit` matches the
`"Article generation: {item_id}"` ledger description with `job_type=
"generate_article"`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import datetime
from urllib.parse import urlparse

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from config import settings

logger = logging.getLogger(__name__)


# ─── Module-level constants ────────────────────────────────────────────

# Defensive LLM budget projection. Real measured cost is ~$0.06-0.07/article
# (Serper 5 calls + YTG + Gemini/Claude writing + validation). We project $0.30
# to fail-fast a near-cap client BEFORE they waste credits on a call that would
# finish over the line. Tune if measured cost diverges in C.1 smoke.
_PROJECTED_COST_USD = 0.30

# Phases broadcast to UI via `content_metadata['in_progress']`. Order matters —
# UI shows `phase_num / phase_total` progress numerator + phase_label as a chip.
_PHASE_TOTAL = 5
_PHASES: dict[str, tuple[int, str]] = {
    "preparing":        (1, "Preparing brand context"),
    "fetching_sources": (2, "Fetching grounding sources"),
    "writing":          (3, "Writing article sections"),
    "validating":       (4, "Validating quality"),
    "finalizing":       (5, "Finalizing HTML & schema.org"),
}

# Estimated total generation time in seconds for the UI ETA display. Article
# gen runs 5-15 min depending on rephrase loops + SERP analysis. 600s (10 min)
# is the rough mean ; the UI caps the displayed pct at 95% so completion
# (status → 'draft' → reload) flips to 100% atomically.
_ESTIMATED_TOTAL_SECONDS = 600


# ─── ImagenClient stub (bypass broken `from src.config` CLI import) ────

def _install_imagen_stub() -> None:
    """Inject a stub `seo_llm.src.imagen_client` module to bypass its broken
    CLI-relative `from src.config import get_task_model_config` import.

    `imagen_client.py` (line 21) uses `from src.config import ...` — an
    absolute import that works when seo_llm/ IS the CWD (CLI mode), but
    breaks when imported as `seo_llm.src.imagen_client` (worker mode, no
    top-level `src` module). The hard fix would be patching the submodule,
    but we never call ImagenClient anyway (generate_image=False), so we
    stub it before `geo_content_generator.py` line 65 `from .imagen_client
    import ImagenClient` resolves.

    Idempotent — safe to call multiple times. Mirrors the
    `_install_geo_stub` pattern in generate_faq.py for the BRAND_MAP case.
    """
    import sys
    import types
    mod_name = "seo_llm.src.imagen_client"
    if mod_name in sys.modules and getattr(sys.modules[mod_name], "_is_stub", False):
        return
    stub = types.ModuleType(mod_name)

    class _StubImagenClient:
        """No-op ImagenClient stub. Article handler always passes
        generate_image=False, so the real class is never instantiated."""

        def __init__(self, *args, **kwargs):
            pass

        def _filter_sensitive_terms(self, *args, **kwargs):
            return args[0] if args else ""

        def _try_generate_image(self, *args, **kwargs):
            return None

    stub.ImagenClient = _StubImagenClient
    stub._is_stub = True
    sys.modules[mod_name] = stub


# ─── Console silencing ────────────────────────────────────────────────

@contextlib.contextmanager
def _silence_rich_console():
    """Redirect seo_llm's `console.print(...)` calls to /dev/null for the
    duration of the article generation.

    `geo_content_generator.py` is a CLI ancestor and uses rich.console.Console
    in ~209 places for progress / warnings / success messages. In a SaaS
    worker container those go to stdout and pollute Docker logs (mostly
    noise, mixed with our structured logger output). We swap the module's
    `console` symbol for one writing to devnull while we run, and restore
    on exit (even on exception). Our subclass's `self.logger.info(...)`
    calls go through the standard `logging` module and are unaffected.

    No-op (yield without patching) if seo_llm imports fail — let the caller
    surface the import error instead of masking it.
    """
    try:
        import seo_llm.src.geo_content_generator as gcg
        from rich.console import Console
    except Exception:
        yield
        return

    original = getattr(gcg, "console", None)
    devnull = open(os.devnull, "w", encoding="utf-8")
    silenced = Console(file=devnull, quiet=True, no_color=True)
    gcg.console = silenced
    try:
        yield
    finally:
        if original is not None:
            gcg.console = original
        try:
            devnull.close()
        except Exception:
            pass


# ─── Module-fn / module-constant shadowing ─────────────────────────────

class _PatchedModuleFns:
    """Context manager : shadow vertical-specific module-level names in
    `seo_llm.src.geo_content_generator` with client-aware closures for the
    duration of one generate_for_opportunity call.

    Names shadowed (saved + restored) :

    Helper functions :
        - _extract_brand_from_source   → returns BrandResolver-derived dict
        - _get_pf_brands_for_pathology → returns the promoted brand list
        - _build_brand_fallback        → returns workspace-brief brand brief

    Constants :
        - BRAND_MAP, PATHOLOGY_PF_BRANDS  → emptied (consumed only by the
                                            helpers above which are shadowed)
        - GAMME_TO_SITE                   → built from ClientBrand.product_lines
        - BRAND_EXPERT_SECTIONS           → built from ClientBrand.expert_section_paths
        - INSTITUTIONAL_URLS              → emptied (we replace via our
                                            _append_pubmed_and_institutional
                                            override below)
        - PATHOLOGY_KEYWORDS              → emptied (parent's heuristic
                                            graceful no-op — slightly less
                                            strict topic match)

    Thread-safety : module-level monkey-patching is process-global. The
    worker runs one `poll_and_execute` at a time today (single-threaded),
    so the patch is safe within one call. If the worker becomes
    multi-threaded, refactor via contextvars OR fork the submodule to
    expose an injection API. See risk R11 in the plan + the post-smoke
    memo for the long-term architectural note.
    """

    def __init__(self, gen: "object"):  # forward ref to _WorkspaceAwareArticleGenerator
        self.gen = gen
        self._saved: dict = {}
        self._gcg = None

    def __enter__(self):
        import seo_llm.src.geo_content_generator as gcg
        self._gcg = gcg

        # Save originals (everything we touch below)
        self._saved = {
            "_extract_brand_from_source":   gcg._extract_brand_from_source,
            "_get_pf_brands_for_pathology": gcg._get_pf_brands_for_pathology,
            "_build_brand_fallback":        gcg._build_brand_fallback,
            "BRAND_MAP":                    gcg.BRAND_MAP,
            "PATHOLOGY_PF_BRANDS":          gcg.PATHOLOGY_PF_BRANDS,
            "GAMME_TO_SITE":                gcg.GAMME_TO_SITE,
            "INSTITUTIONAL_URLS":           gcg.INSTITUTIONAL_URLS,
            "BRAND_EXPERT_SECTIONS":        gcg.BRAND_EXPERT_SECTIONS,
            "PATHOLOGY_KEYWORDS":           gcg.PATHOLOGY_KEYWORDS,
        }

        gen = self.gen

        # ── Helper-fn shadows ─────────────────────────────────────────
        def _fake_extract(_source_name: str) -> dict:
            """Replace `_extract_brand_from_source` : the synthetic
            source_name we hand to the pipeline is never parsed — we
            already know the brand context from BrandResolver."""
            return {
                "name":            gen._ux_promoted_lead_brand_name,
                "site":            gen._ux_promoted_brand_domains[0]
                                    if gen._ux_promoted_brand_domains else "",
                "code":            "",
                "category":        gen._ux_client_industry or "generic",
                "is_own":          True,
                "competitor_name": "",
                "competitor_site": "",
                "pf_brands": [
                    {"code": "", "name": n, "site": d, "is_own": True}
                    for n, d in zip(
                        gen._ux_promoted_brand_names,
                        gen._ux_promoted_brand_domains,
                    )
                ],
            }
        gcg._extract_brand_from_source = _fake_extract

        def _fake_get_pf_brands(_topic: str) -> list[dict]:
            """Replace `_get_pf_brands_for_pathology` : ignore the topic
            param (used by original to lookup PATHOLOGY_PF_BRANDS — empty
            for non-PF clients). Always return our BrandResolver promoted
            brand list."""
            return [
                {"code": "", "name": n, "site": d, "is_own": True}
                for n, d in zip(
                    gen._ux_promoted_brand_names,
                    gen._ux_promoted_brand_domains,
                )
            ]
        gcg._get_pf_brands_for_pathology = _fake_get_pf_brands

        def _fake_brand_fallback(_brand_code: str) -> str:
            """Replace `_build_brand_fallback` : return a workspace-aware
            brand brief built from BrandResolver + workspace_brief, instead
            of the PF-hardcoded BRAND_MAP lookup."""
            parts = [f"# {gen._ux_promoted_lead_brand_name or 'Brand'}", ""]
            if gen._ux_promoted_brand_names:
                parts.append(f"Marques: {', '.join(gen._ux_promoted_brand_names)}")
            if gen._ux_promoted_brand_domains:
                parts.append(f"Sites: {', '.join(gen._ux_promoted_brand_domains)}")
            if gen._ux_client_industry:
                parts.append(f"Industrie: {gen._ux_client_industry}")
            if gen._ux_workspace_brief_text:
                parts.append("")
                parts.append(gen._ux_workspace_brief_text)
            return "\n".join(parts)
        gcg._build_brand_fallback = _fake_brand_fallback

        # ── Constant shadows ──────────────────────────────────────────

        # GAMME_TO_SITE : inverse mapping {product_line_lower: brand_site}.
        # Built from ClientBrand.product_lines (migration 032). Used by
        # parent's _wrap_html post-processing to linkify product names in
        # generated tables.
        gamme_to_site: dict[str, str] = {}
        for domain, plines in zip(
            gen._ux_promoted_brand_domains,
            gen._ux_promoted_brand_product_lines,
        ):
            if not domain:
                continue
            for pl in (plines or []):
                if pl and isinstance(pl, str):
                    gamme_to_site[pl.lower()] = domain
        gcg.GAMME_TO_SITE = gamme_to_site

        # BRAND_EXPERT_SECTIONS : {brand_site: [path_fragments]}. Built from
        # ClientBrand.expert_section_paths (migration 032). Parent's
        # _fetch_brand_content uses this for expert-page scraping strategy.
        # Register both the www. and non-www variant since seo_llm's lookup
        # is inconsistent (sometimes uses one form, sometimes the other).
        expert_sections: dict[str, list[str]] = {}
        for domain, paths in zip(
            gen._ux_promoted_brand_domains,
            gen._ux_promoted_brand_expert_section_paths,
        ):
            if not domain or not paths:
                continue
            expert_sections[domain] = list(paths)
            bare = domain[4:] if domain.startswith("www.") else "www." + domain
            expert_sections[bare] = list(paths)
        gcg.BRAND_EXPERT_SECTIONS = expert_sections

        # INSTITUTIONAL_URLS : emptied. We replace via our
        # _append_pubmed_and_institutional override which injects
        # client.trust_domains instead. Parent's _wrap_html post-processing
        # also reads this — empty dict = no curated sources footer (graceful).
        gcg.INSTITUTIONAL_URLS = {}

        # PATHOLOGY_KEYWORDS : emptied. Parent uses for URL/topic match
        # heuristics and brand-content scoring. Empty = slightly less strict
        # matching (parent's `for patho_key, kw_set in PATHOLOGY_KEYWORDS.items()`
        # just iterates zero times → no boost, no penalty). Multi-vertical
        # NLP-derived variant deferred to C.7.
        gcg.PATHOLOGY_KEYWORDS = {}

        # PATHOLOGY_PF_BRANDS + BRAND_MAP : consumed only by helpers we
        # already shadow. Empty defensively (in case future seo_llm versions
        # add another consumer).
        gcg.PATHOLOGY_PF_BRANDS = {}
        gcg.BRAND_MAP = {}

        return self

    def __exit__(self, exc_type, exc, tb):
        for k, v in self._saved.items():
            setattr(self._gcg, k, v)
        return False  # don't suppress exceptions


# ─── Workspace-aware article generator subclass ────────────────────────

def _make_workspace_aware_class():
    """Lazy subclass factory — defers the heavy `seo_llm.src.geo_content_generator`
    import to first call. Mirrors `worker/handlers/generate_faq.py
    :_make_workspace_aware_class` pattern.

    Why lazy : the module's top-level imports cascade to anthropic / msal /
    google-cloud-aiplatform / Pillow / etc. We want worker boot to stay fast
    and to surface any missing dep on the first article generation (when the
    user clicks Generate) rather than on a happy-path FAQ user's worker boot.

    Stub install order matters : `_install_imagen_stub()` MUST run before the
    `from seo_llm.src.geo_content_generator import GEOContentGenerator` line
    below — geo_content_generator.py line 65 does `from .imagen_client import
    ImagenClient`, which triggers the broken `from src.config import ...`
    inside imagen_client.py. Stub first → real import resolves to our no-op.
    """
    _install_imagen_stub()
    from seo_llm.src.geo_content_generator import GEOContentGenerator

    class _WorkspaceAwareArticleGenerator(GEOContentGenerator):
        """Subclass injecting BrandResolver-derived context into the seo_llm
        article pipeline.

        Overrides the smallest possible surface :
          - generate_for_opportunity     : wrap with _PatchedModuleFns,
                                           fire 'preparing' phase callback.
          - _fetch_brand_content         : phase callback + post-filter URLs
                                           to promoted brand domains.
          - _fetch_scientific_sources    : phase callback + strip URLs on
                                           competitor denylist.
          - _fetch_user_reviews          : phase callback + strip competitor
                                           review entries.
          - _append_pubmed_and_institutional : replace INSTITUTIONAL_URLS
                                           lookup with client trust_domains.
          - _generate_geo_content        : pass-through with 'writing' phase.
          - _validate_content            : pass-through with 'validating' phase.
          - _wrap_html                   : pass-through with 'finalizing' phase.

        All overrides forward args/kwargs unchanged to super() — we don't
        fork any seo_llm logic. The point is to inject our context, fire
        progress callbacks, and let the pipeline run end-to-end as designed.
        """

        def __init__(
            self,
            *,
            workspace_brief_text: str = "",
            promoted_brand_names: list[str] | None = None,
            promoted_lead_brand_name: str = "",
            promoted_brand_domains: list[str] | None = None,
            promoted_brand_aliases: dict[str, list[str]] | None = None,
            promoted_brand_expert_section_paths: list[list[str]] | None = None,
            promoted_brand_product_lines: list[list[str]] | None = None,
            trust_domains: list[str] | None = None,
            competitor_domains: set[str] | None = None,
            excluded_brand_names: list[str] | None = None,
            client_industry: str = "",
            writing_provider: str = "claude",
            phase_callback=None,
            **kwargs,
        ):
            super().__init__(writing_provider=writing_provider, **kwargs)

            # Phase C.1.5 — YTG receives a FAN-OUT query (clean SEO format
            # 30-80c, extracted from real LLM web_search_queries) instead of
            # the conversational long question. The fan-out extraction +
            # selection happens in execute() BEFORE constructing this
            # subclass, and opportunity["question_text"] is set to the
            # primary fan-out (which IS short by construction).
            #
            # We keep a defensive YTG truncate as a SAFETY NET only — fires
            # when the primary fan-out is somehow still > 150c (shouldn't
            # happen, but Haiku synthesis edge cases are possible).
            #
            # See `project_phase_c1_article_handler.md` section C.1.5 for the
            # full architecture (fan_out_extractor B1+B2 hybrid + ranking).
            _orig_create_guide = self.ytg.create_guide

            def _create_guide_safety_truncate(query, *args, **kwargs):
                _YTG_MAX = 150
                if len(query) > _YTG_MAX:
                    truncated = query[:_YTG_MAX]
                    if " " in truncated:
                        truncated = truncated.rsplit(" ", 1)[0]
                    logger.warning(
                        f"YTG safety-net truncate: query {len(query)}→{len(truncated)}c. "
                        f"This shouldn't happen post-C.1.5 (fan-outs are 30-80c by "
                        f"construction). Investigate primary fan-out selection."
                    )
                    query = truncated
                return _orig_create_guide(query, *args, **kwargs)

            self.ytg.create_guide = _create_guide_safety_truncate

            self._ux_workspace_brief_text = workspace_brief_text or ""
            self._ux_promoted_brand_names = list(promoted_brand_names or [])
            self._ux_promoted_lead_brand_name = (
                promoted_lead_brand_name
                or (self._ux_promoted_brand_names[0]
                    if self._ux_promoted_brand_names else "")
            )
            self._ux_promoted_brand_domains = list(promoted_brand_domains or [])
            self._ux_promoted_brand_aliases = dict(promoted_brand_aliases or {})
            self._ux_promoted_brand_expert_section_paths = list(
                promoted_brand_expert_section_paths or []
            )
            self._ux_promoted_brand_product_lines = list(
                promoted_brand_product_lines or []
            )
            self._ux_trust_domains = list(trust_domains or [])
            self._ux_competitor_domains = set(competitor_domains or set())
            self._ux_excluded_brand_names = list(excluded_brand_names or [])
            self._ux_client_industry = client_industry or ""
            self._phase_callback = phase_callback or (lambda *a, **kw: None)

        def _fire_phase(self, key: str) -> None:
            """Best-effort phase notification — never propagates exceptions
            (a phase_callback failure must not crash the article pipeline)."""
            try:
                num, label = _PHASES[key]
                self._phase_callback(key, num, label)
            except Exception:
                logger.exception(
                    "phase_callback failed for phase=%s — continuing generation",
                    key,
                )

        # ─── Pipeline overrides ────────────────────────────────────────

        def generate_for_opportunity(self, opportunity, fanout_queries=None,
                                     faq_file=None, generate_image=True):
            """Wrap parent's generate_for_opportunity with module-fn shadow
            and 'preparing' phase callback.

            The shadow context manager replaces vertical-specific module
            functions/constants for the duration of the call (and restores
            them on exit). All other phase callbacks fire from the inline
            method overrides below (called by super's own internal loop).
            """
            self._fire_phase("preparing")
            with _PatchedModuleFns(self):
                return super().generate_for_opportunity(
                    opportunity,
                    fanout_queries=fanout_queries,
                    faq_file=faq_file,
                    generate_image=generate_image,
                )

        def _fetch_brand_content(self, brand_site: str, topic: str,
                                  question: str) -> str:
            """Phase callback + post-filter content to keep only blocks
            whose URL is on a promoted brand domain. Drops off-brand
            pages that grounding may surface for hot queries (e.g., a
            competitor's blog post showing up on a brand-name search).

            If filtering removes everything, fall back to original content
            — losing brand context entirely would degrade the article
            harder than letting a few off-brand URLs through (the LLM
            still has the workspace_brief + promoted brands prompt block).
            """
            self._fire_phase("fetching_sources")
            content = super()._fetch_brand_content(brand_site, topic, question)
            return _filter_content_to_promoted_domains(
                content, self._ux_promoted_brand_domains,
            )

        def _fetch_scientific_sources(self, topic: str, question: str,
                                       brand_category: str = "") -> str:
            """Phase callback + strip URL blocks on competitor brand
            domains. The parent (Serper / Gemini / OpenAI grounding) does
            its own discovery — we just clean its output through the
            per-scan competitor denylist.
            """
            self._fire_phase("fetching_sources")
            content = super()._fetch_scientific_sources(
                topic, question, brand_category,
            )
            return _strip_competitor_blocks(content, self._ux_competitor_domains)

        def _fetch_user_reviews(self, brand_name: str, brand_site: str,
                                 topic: str):
            """Phase callback + filter (text, urls_by_domain, verified_domains)
            tuple against competitor denylist. Defensive : a competitor's
            review aggregator showing up as a "verified domain" would later
            leak the competitor brand name into the article via citations.
            """
            self._fire_phase("fetching_sources")
            text, urls_by_domain, verified_domains = super()._fetch_user_reviews(
                brand_name, brand_site, topic,
            )
            comp = self._ux_competitor_domains
            urls_by_domain = {
                d: urls for d, urls in (urls_by_domain or {}).items()
                if not _is_competitor_domain(d, comp)
            }
            verified_domains = {
                d for d in (verified_domains or set())
                if not _is_competitor_domain(d, comp)
            }
            text = _strip_competitor_blocks(text, comp)
            return text, urls_by_domain, verified_domains

        def _append_pubmed_and_institutional(self, text: str, topic: str) -> str:
            """Replace the seo_llm INSTITUTIONAL_URLS lookup with the
            client's trust_domains (discovered via OpenAI web_search per
            client industry — see `worker/services/trust_sources.py`).

            Parent's implementation does PubMed fetch + INSTITUTIONAL_URLS
            iteration. We've shadowed INSTITUTIONAL_URLS to {} so that loop
            no-ops. We then append a "TRUSTED REFERENCE DOMAINS" block
            from the per-client list. Top 8 to keep prompt tight.
            """
            text = super()._append_pubmed_and_institutional(text, topic)
            if self._ux_trust_domains:
                trust_block = "\n".join(
                    f"- {d}" for d in self._ux_trust_domains[:8]
                )
                text += (
                    "\n\nTRUSTED REFERENCE DOMAINS "
                    "(client-discovered authoritative sources):\n"
                    + trust_block
                    + "\nWhen citing references, prefer URLs on these domains "
                    "or universal public-sector TLDs "
                    "(.gov, .gouv.fr, .europa.eu, .int).\n"
                )
            return text

        def _generate_geo_content(self, *args, **kwargs):
            """Pass-through with 'writing' phase callback."""
            self._fire_phase("writing")
            return super()._generate_geo_content(*args, **kwargs)

        def _validate_content(self, *args, **kwargs):
            """Pass-through with 'validating' phase callback."""
            self._fire_phase("validating")
            return super()._validate_content(*args, **kwargs)

        def _wrap_html(self, *args, **kwargs):
            """Pass-through with 'finalizing' phase callback.

            Parent's post-processing tries to inject GAMME_TO_SITE links and
            an INSTITUTIONAL_URLS sources footer. We've shadowed both —
            GAMME_TO_SITE is populated from ClientBrand.product_lines if
            the user filled it in (migration 032), INSTITUTIONAL_URLS is
            empty. So post-processing degrades to "no extra product-line
            links in tables, no curated sources footer" for clients that
            haven't seeded product_lines. Article still ships clean
            Schema.org + content — graceful.
            """
            self._fire_phase("finalizing")
            return super()._wrap_html(*args, **kwargs)

    return _WorkspaceAwareArticleGenerator


# Cached after first call (per worker process)
_WorkspaceAwareArticleGenerator_cls: type | None = None


def _get_workspace_aware_class():
    global _WorkspaceAwareArticleGenerator_cls
    if _WorkspaceAwareArticleGenerator_cls is None:
        _WorkspaceAwareArticleGenerator_cls = _make_workspace_aware_class()
    return _WorkspaceAwareArticleGenerator_cls


# ─── URL filtering helpers ─────────────────────────────────────────────

def _normalize_domain(d: str | None) -> str:
    """Strip protocol, www., trailing path. Return lowercase bare domain."""
    if not d:
        return ""
    s = d.lower().strip()
    if s.startswith("http://"):
        s = s[7:]
    elif s.startswith("https://"):
        s = s[8:]
    if s.startswith("www."):
        s = s[4:]
    return s.split("/", 1)[0].rstrip(".")


def _is_competitor_domain(domain: str | None, comp_set: set[str]) -> bool:
    """Match `domain` against the competitor set with subdomain awareness."""
    if not domain or not comp_set:
        return False
    d = _normalize_domain(domain)
    if not d:
        return False
    for c in comp_set:
        cn = _normalize_domain(c)
        if not cn:
            continue
        if d == cn or d.endswith("." + cn):
            return True
    return False


def _filter_content_to_promoted_domains(content: str,
                                         promoted_domains: list[str]) -> str:
    """Keep only `URL: ...\\n<body>` blocks whose URL is on a promoted brand
    domain. Mirrors `generate_faq.py:_filter_brand_context_by_site` extended
    to a list of acceptable domains. Defensive : on parse error returns the
    original content (don't risk losing brand context entirely).
    """
    if not content or not promoted_domains:
        return content or ""
    if "URL:" not in content:
        return content

    promoted_set = {_normalize_domain(d) for d in promoted_domains if d}
    promoted_set.discard("")
    if not promoted_set:
        return content

    def _on_promoted(url: str) -> bool:
        try:
            h = _normalize_domain(urlparse(url or "").netloc or "")
        except Exception:
            return False
        if not h:
            return False
        return any(h == pd or h.endswith("." + pd) for pd in promoted_set)

    try:
        chunks = content.split("URL:")
        kept: list[str] = []
        for chunk in chunks[1:]:  # skip preamble
            first_nl = chunk.find("\n")
            if first_nl < 0:
                continue
            url_line = chunk[:first_nl].strip()
            body = chunk[first_nl:]
            if _on_promoted(url_line):
                kept.append(f"URL: {url_line}{body}")
        # If filtering nuked everything, return original — losing context
        # entirely degrades the article more than keeping a few off-brand URLs
        return "\n\n".join(kept) if kept else content
    except Exception:
        logger.exception(
            "_filter_content_to_promoted_domains parse error — keeping original"
        )
        return content


def _strip_competitor_blocks(content: str, comp_set: set[str]) -> str:
    """Drop `URL: ...\\n<body>` blocks whose URL is on a competitor brand
    domain. Universal e-commerce / social patterns are already filtered
    upstream by the parent's `services.url_filter` (when wired) — we only
    add the per-scan competitor brand domain denylist here.
    """
    if not content or not comp_set:
        return content or ""
    if "URL:" not in content:
        return content

    try:
        chunks = content.split("URL:")
        kept: list[str] = []
        for chunk in chunks[1:]:
            first_nl = chunk.find("\n")
            if first_nl < 0:
                continue
            url_line = chunk[:first_nl].strip()
            body = chunk[first_nl:]
            try:
                netloc = urlparse(url_line).netloc or ""
            except Exception:
                netloc = ""
            if _is_competitor_domain(netloc, comp_set):
                continue
            kept.append(f"URL: {url_line}{body}")
        return "\n\n".join(kept)
    except Exception:
        logger.exception(
            "_strip_competitor_blocks parse error — returning original"
        )
        return content


# ─── Progress field helpers (UI polling backend) ───────────────────────

def _update_progress(item, db: Session, phase: str, phase_num: int,
                      phase_label: str) -> None:
    """Update `item.content_metadata['in_progress']` for UI polling.

    Called by `_WorkspaceAwareArticleGenerator._fire_phase` at the start
    of each macroscopic phase. Best-effort : NEVER raises (a DB blip
    must not crash the pipeline). No-op if the item is no longer in
    'generating' status — defensive against late callbacks after a
    refund-reset path resets the status.
    """
    try:
        db.refresh(item)
    except Exception:
        pass  # refresh failure is fine — we read the stale state

    if item.status != "generating":
        logger.info(
            "phase_callback: item %s no longer 'generating' "
            "(status=%s) — skipping progress update", item.id, item.status,
        )
        return

    meta = dict(item.content_metadata or {})
    in_progress = dict(meta.get("in_progress") or {})
    # Preserve started_at + estimated_total_seconds set by execute() at boot
    in_progress.update({
        "phase":       phase,
        "phase_num":   phase_num,
        "phase_total": _PHASE_TOTAL,
        "phase_label": phase_label,
        "updated_at":  datetime.utcnow().isoformat(),
    })
    meta["in_progress"] = in_progress
    item.content_metadata = meta
    flag_modified(item, "content_metadata")
    try:
        db.commit()
    except Exception:
        logger.exception("Failed to commit progress update for item %s", item.id)
        try:
            db.rollback()
        except Exception:
            pass


def _clear_progress(item) -> None:
    """Remove `in_progress` from content_metadata. Called on abort/error
    path; the success path overwrites content_metadata entirely with the
    final audit payload, which implicitly drops `in_progress`."""
    meta = dict(item.content_metadata or {})
    if "in_progress" in meta:
        meta.pop("in_progress", None)
        item.content_metadata = meta
        flag_modified(item, "content_metadata")


# ─── Public entrypoint ─────────────────────────────────────────────────

def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    """Generate netlinking article content for one ScanContentItem.

    job_payload must contain: ``{"item_id": "<uuid>"}``
    scan_id is the parent scan, passed by the worker for credit accounting.

    Side effects on success:
      - item.status            = 'draft'
      - item.content_html      = generated HTML (Schema.org wrapper)
      - item.content_text      = plain-text variant
      - item.article_outline   = JSON outline (H2 sections)
      - item.content_metadata  = audit payload (quality_score, ytg_soseo,
                                 ytg_dseo, sources_used, duration_ms, …)
      - item.promoted_brand_ids = brands resolved at gen time (audit trail)
      - LlmUsageLog row (coarse — pipeline mixes Gemini/Claude/OpenAI)

    Side effects on failure (caller propagates exception, worker retries
    up to max_attempts before triggering `_refund_content_item_credit`
    with job_type='generate_article'):
      - item.status            = 'identified' (retry-able from Kanban)
      - in_progress key cleared
    """
    from models import ScanContentItem, ClientBrand, Client

    item_id = job_payload.get("item_id")
    if not item_id:
        raise RuntimeError("generate_article requires item_id in job payload")

    item = (
        db.query(ScanContentItem)
        .filter(ScanContentItem.id == item_id)
        .first()
    )
    if not item:
        raise RuntimeError(f"ScanContentItem {item_id} not found")

    if item.content_type != "netlinking_article":
        raise RuntimeError(
            f"ScanContentItem {item_id} is content_type='{item.content_type}', "
            f"not 'netlinking_article' — wrong handler"
        )

    if not item.target_url:
        # API endpoint already gates this with HTTP 400, defensive worker-side guard
        from exceptions import PermanentScanError
        raise PermanentScanError(
            "Article generation needs a target URL. Open the item in the "
            "validation page and pick a URL on your site that should host "
            "this article."
        )

    # ── Budget cap (cap-then-call) ────────────────────────────────────
    from services.llm_budget import assert_within_budget, BudgetExceeded
    client_id_for_budget = item.scan.client_id if item.scan else None
    assert_within_budget(
        client_id_for_budget, db,
        projected_cost_usd=_PROJECTED_COST_USD,
    )

    # ── Mark generating + initial progress meta ──────────────────────
    item.status = "generating"
    meta_init = dict(item.content_metadata or {})
    meta_init["in_progress"] = {
        "started_at":              datetime.utcnow().isoformat(),
        "estimated_total_seconds": _ESTIMATED_TOTAL_SECONDS,
        "phase":                   "preparing",
        "phase_num":               1,
        "phase_total":             _PHASE_TOTAL,
        "phase_label":             _PHASES["preparing"][1],
    }
    item.content_metadata = meta_init
    flag_modified(item, "content_metadata")
    db.commit()

    # ── Workspace + brand + trust context ────────────────────────────
    from adapters.brief_injector import format_workspace_brief
    from services.brand_resolver import resolve_promotion, PromotionUnsetError
    from services.trust_sources import get_trust_sources_for_client
    from services.competitor_domains import get_competitor_domains_for_scan

    client = None
    if item.scan and item.scan.client_id:
        client = db.query(Client).filter(Client.id == item.scan.client_id).first()

    workspace_brief_text = format_workspace_brief(client.apps if client else None)
    client_industry = ""
    if client and client.apps:
        client_brief = client.apps.get("client_brief") or {}
        client_industry = (client_brief.get("industry") or "").strip()

    promoted_brand_ids: list = []
    promoted_brand_names: list[str] = []
    promoted_brand_domains: list[str] = []
    promoted_brand_aliases: dict[str, list[str]] = {}
    excluded_brand_names: list[str] = []
    try:
        promotion = resolve_promotion(item.scan, db)
        promoted_brand_ids = list(promotion.promote_brand_ids)
        promoted_brand_names = [b.name for b in promotion.promote_brands if b.name]
        promoted_brand_domains = [
            _normalize_domain(b.domain) for b in promotion.promote_brands
        ]
        promoted_brand_aliases = {
            b.name: list(b.aliases or []) for b in promotion.promote_brands
        }
        excluded_brand_names = list(promotion.exclude_domain_names or [])
        logger.info(
            f"Article promotion resolved (item {item_id}): "
            f"promote={promoted_brand_names}, "
            f"exclude={excluded_brand_names[:5]}"
            f"{'…' if len(excluded_brand_names) > 5 else ''}, "
            f"via={promotion.resolved_via}"
        )
    except PromotionUnsetError as e:
        logger.warning(
            f"Article for item {item_id} has no resolved brands to promote — "
            f"generating with workspace context only ({e})"
        )

    # ── Per-item LEAD override (mirror generate_faq.py:187-235 pattern) ──
    item_override_ids = [
        str(b) for b in (getattr(item, "promoted_brand_ids", None) or [])
    ]
    if item_override_ids and promoted_brand_ids:
        existing_id_strs = {str(b) for b in promoted_brand_ids}
        unknown_ids = [bid for bid in item_override_ids
                       if bid not in existing_id_strs]
        if unknown_ids:
            extra_rows = (
                db.query(ClientBrand)
                .filter(ClientBrand.id.in_(unknown_ids))
                .all()
            )
            by_id_extra = {str(b.id): b for b in extra_rows}
            for bid in unknown_ids:
                b = by_id_extra.get(bid)
                if b and b.name:
                    promoted_brand_ids.append(b.id)
                    promoted_brand_names.append(b.name)
                    promoted_brand_domains.append(_normalize_domain(b.domain))
                    promoted_brand_aliases[b.name] = list(b.aliases or [])

        by_id_pos = {str(b): i for i, b in enumerate(promoted_brand_ids)}
        ordered_ids, ordered_names, ordered_domains = [], [], []
        seen: set[str] = set()
        for bid in item_override_ids:
            pos = by_id_pos.get(bid)
            if pos is None or bid in seen:
                continue
            ordered_ids.append(promoted_brand_ids[pos])
            ordered_names.append(promoted_brand_names[pos])
            ordered_domains.append(promoted_brand_domains[pos])
            seen.add(bid)
        for i, b in enumerate(promoted_brand_ids):
            if str(b) in seen:
                continue
            ordered_ids.append(b)
            ordered_names.append(promoted_brand_names[i])
            ordered_domains.append(promoted_brand_domains[i])
            seen.add(str(b))
        if ordered_names and (not promoted_brand_names
                              or ordered_names[0] != promoted_brand_names[0]):
            logger.info(
                f"Article promotion: per-item override applied (item {item_id}) "
                f"— LEAD={ordered_names[0]}"
            )
        promoted_brand_ids = ordered_ids
        promoted_brand_names = ordered_names
        promoted_brand_domains = ordered_domains

    # ── Load full ClientBrand rows for migration-032 fields ──────────
    primary_brand_rows: list = []
    if promoted_brand_ids:
        rows = (
            db.query(ClientBrand)
            .filter(ClientBrand.id.in_(promoted_brand_ids))
            .all()
        )
        by_id = {b.id: b for b in rows}
        primary_brand_rows = [by_id[bid] for bid in promoted_brand_ids
                              if bid in by_id]
    promoted_brand_expert_section_paths = [
        list(b.expert_section_paths or []) for b in primary_brand_rows
    ]
    promoted_brand_product_lines = [
        list(b.product_lines or []) for b in primary_brand_rows
    ]

    # ── Trust sources (SOFT prefer-hint) + competitors (HARD denylist) ──
    trust_domains: list[str] = []
    if client:
        try:
            trust_domains = get_trust_sources_for_client(client.id, db)
        except Exception:
            logger.exception(
                f"get_trust_sources_for_client failed for client {client.id} "
                f"— article will run without trust prefer-hint"
            )

    competitor_domains: set[str] = set()
    if item.scan and item.scan.id:
        try:
            competitor_domains = get_competitor_domains_for_scan(
                item.scan.id, db,
            )
        except Exception:
            logger.exception(
                f"get_competitor_domains_for_scan failed for scan {item.scan.id}"
            )

    # ── Fan-out extraction (Phase C.1.5) ─────────────────────────────
    # Replaces the long conversational question with a short Google-search-
    # style fan-out for YTG. The primary fan-out (index [0]) feeds YTG;
    # all fan-outs (incl. primary) are passed to generate_for_opportunity
    # via fanout_queries for content coverage + FAQ Q seeds.
    #
    # Fallback chain (B1 → B2 → truncated question) handled inside
    # extract_or_get_cached. Returns empty list ONLY when there's no
    # scan_llm_results data at all for this question (very rare).
    from services.fan_out_extractor import extract_or_get_cached

    fanouts: list[str] = []
    question_id_for_extract = _resolve_scan_question_id(item, db)
    if question_id_for_extract:
        try:
            fanouts = extract_or_get_cached(
                question_id_for_extract, str(item.scan_id), db,
            )
        except Exception:
            logger.exception(
                f"fan_out_extractor crashed for question {question_id_for_extract} "
                f"— falling back to truncated question_text"
            )

    if fanouts:
        # Use primary fan-out as YTG-bound query. The rest of the pipeline
        # (SERP analysis, brand_content fetch, content gen prompts) ALSO sees
        # this short fan-out via opportunity["question_text"] — that's the
        # tradeoff for not patching all the touch-points. SERP analysis on a
        # clean SEO query is actually BETTER than on a conversational long
        # question (cleaner top-10 SERP, more representative grammes).
        primary_fanout = fanouts[0]
        logger.info(
            f"fan_out_extractor: {len(fanouts)} fan-outs for question "
            f"{question_id_for_extract}, primary='{primary_fanout[:60]}' "
            f"(used for YTG + SERP). Full conversational question preserved "
            f"in item.target_question for FAQ UI + scan_llm_tests."
        )
        opportunity = _build_synthetic_opportunity(item, ytg_query=primary_fanout)
    else:
        logger.warning(
            f"fan_out_extractor returned empty for item {item_id} "
            f"(no scan_llm_results captured yet ?) — falling back to question_text "
            f"with YTG safety-net truncate."
        )
        opportunity = _build_synthetic_opportunity(item)

    logger.info(
        f"Generating article for content_item {item_id} "
        f"(target={item.target_url}, scan={scan_id}, "
        f"workspace_brief={'yes' if workspace_brief_text else 'no'}, "
        f"promote={len(promoted_brand_names)}, "
        f"exclude={len(excluded_brand_names)}, "
        f"trust={len(trust_domains)}, "
        f"competitor={len(competitor_domains)}, "
        f"fanouts={len(fanouts)})"
    )

    # ── Generate ─────────────────────────────────────────────────────
    start = time.time()
    writing_provider = "claude"
    if hasattr(settings, "task_models"):
        writing_provider = settings.task_models.get(
            "generate_article_writing", writing_provider,
        )

    try:
        generator_cls = _get_workspace_aware_class()
        generator = generator_cls(
            workspace_brief_text=workspace_brief_text,
            promoted_brand_names=promoted_brand_names,
            promoted_lead_brand_name=(
                promoted_brand_names[0] if promoted_brand_names else ""
            ),
            promoted_brand_domains=promoted_brand_domains,
            promoted_brand_aliases=promoted_brand_aliases,
            promoted_brand_expert_section_paths=promoted_brand_expert_section_paths,
            promoted_brand_product_lines=promoted_brand_product_lines,
            trust_domains=trust_domains,
            competitor_domains=competitor_domains,
            excluded_brand_names=excluded_brand_names,
            client_industry=client_industry,
            writing_provider=writing_provider,
            phase_callback=lambda key, num, label: _update_progress(
                item, db, key, num, label,
            ),
        )

        with _silence_rich_console():
            result = generator.generate_for_opportunity(
                opportunity=opportunity,
                # Phase C.1.5 — pass ALL fan-outs (incl. primary) to the
                # writer. The seo_llm pipeline uses fanout_queries for :
                #   - content gen prompt injection (_format_fanout_section)
                #     so the article covers each sub-intent explicitly
                #   - post-gen coverage check (fanout_coverage / covered /
                #     missed in result dict)
                #   - FAQ Schema.org Q seeds (first 5 fan-outs become FAQ Qs)
                fanout_queries=fanouts or None,
                faq_file=None,
                generate_image=False,
            )

    except BudgetExceeded:
        item.status = "identified"
        _clear_progress(item)
        db.commit()
        raise  # worker retry chain logs + Sentry capture

    except Exception as e:
        item.status = "identified"
        _clear_progress(item)
        db.commit()
        raise RuntimeError(
            f"Article generation failed for item {item_id}: {e}"
        ) from e

    duration_ms = int((time.time() - start) * 1000)

    # ── Persist result ──────────────────────────────────────────────
    html_content = result.get("html_content") or ""
    validation_verdict = result.get("validation_verdict") or "ERROR"
    validation_score = float(result.get("validation_score") or 0.0)
    ytg_soseo = float(result.get("ytg_soseo") or 0.0)
    ytg_dseo = float(result.get("ytg_dseo") or 0.0)
    target_word_count = int(result.get("target_word_count") or 0)
    fanout_coverage = int(result.get("fanout_coverage") or 0)
    serp_competitors = int(result.get("serp_competitors") or 0)
    guide_id = int(result.get("guide_id") or 0)

    item.content_html = html_content or None
    item.content_text = _html_to_text(html_content) if html_content else None
    item.status = "draft"
    if promoted_brand_ids:
        item.promoted_brand_ids = promoted_brand_ids
        flag_modified(item, "promoted_brand_ids")

    # Lightweight outline = H2 list from generated HTML. Future C.2 may
    # extend with section word counts, ytg per-section scores, etc.
    outline = _extract_outline_from_html(html_content)
    if outline:
        item.article_outline = json.dumps(outline, ensure_ascii=False)

    sources_used = _extract_sources_from_html(html_content)

    # quality_score = 0-100 (UI chip expects 0-100, scales of FAQ generator)
    item.content_metadata = {
        "quality_score":       int(validation_score * 10),
        "validation_verdict":  validation_verdict,
        "validation_score":    validation_score,
        "ytg_soseo":           ytg_soseo,
        "ytg_dseo":            ytg_dseo,
        "target_word_count":   target_word_count,
        "fanout_coverage":     fanout_coverage,
        "fanout_covered":      list(result.get("fanout_covered") or []),
        "fanout_missed":       list(result.get("fanout_missed") or []),
        "serp_competitors":    serp_competitors,
        "sources_used":        sources_used,
        "sources_count":       len(sources_used),
        "duration_ms":         duration_ms,
        "generated_at":        datetime.utcnow().isoformat(),
        "generator_version":   "geo-section-mode-v2-fanout",  # bumped for C.1.5
        "writing_provider":    writing_provider,
        "guide_id":            guide_id,
        # Phase C.1.5 — fan-outs actually used (primary at [0] = sent to YTG,
        # rest passed to pipeline for content coverage + FAQ Q seeds).
        # Empty list = no fan-outs extracted (rare, indicates B1+B2 both failed
        # OR no scan_llm_results data for the question). Persisted for audit
        # + future UI transparency in validation page.
        "fan_outs_used":       list(fanouts or []),
    }
    # 'in_progress' implicitly dropped by overwriting content_metadata
    flag_modified(item, "content_metadata")

    db.commit()

    # ── Log LLM usage (coarse — pipeline mixes providers per phase) ─
    try:
        from adapters.llm_logger import log_llm_usage
        log_llm_usage(
            db,
            provider=writing_provider,
            model="geo-pipeline",
            operation="generate_article",
            input_tokens=0,
            output_tokens=0,
            duration_ms=duration_ms,
            scan_id=scan_id,
            client_id=str(item.scan.client_id) if item.scan else None,
        )
    except Exception:
        logger.exception("log_llm_usage failed for generate_article")

    logger.info(
        f"Article generated for item {item_id}: "
        f"verdict={validation_verdict}, score={validation_score}/10, "
        f"SOSEO={ytg_soseo}, DSEO={ytg_dseo}, "
        f"sources={len(sources_used)}, {duration_ms}ms"
    )

    return {
        "status":             "draft",
        "validation_verdict": validation_verdict,
        "validation_score":   validation_score,
        "ytg_soseo":          ytg_soseo,
        "ytg_dseo":           ytg_dseo,
        "duration_ms":        duration_ms,
        "html_length":        len(html_content),
        "sources_count":      len(sources_used),
    }


# ─── Synthetic-opportunity + HTML extraction helpers ───────────────────

def _build_synthetic_opportunity(item, ytg_query: str | None = None) -> dict:
    """Construct the opportunity dict expected by GEOContentGenerator.

    The `source_name` is synthetic — `_extract_brand_from_source` is
    shadowed via `_PatchedModuleFns` so source_name parsing never runs.

    `ytg_query` (Phase C.1.5) : when provided, replaces the conversational
    long question with the primary fan-out (short SEO query) for YTG
    consumption. The full conversational question stays preserved on
    item.target_question for the UI / FAQ Schema.org / future use cases.
    Fallback to item.target_question when not provided (legacy/empty path).
    """
    question_text = (ytg_query or item.target_question or item.topic_name or "").strip()
    return {
        "source_name":   f"item-{item.id}",
        "media_domain":  _extract_url_domain(item.target_url),
        "question_id":   str(item.id),
        "question_text": question_text,
        "persona_name":  (item.persona_name or "").strip(),
        "site_type":     "",
    }


def _resolve_scan_question_id(item, db: Session) -> str | None:
    """Lookup ScanQuestion.id from item.target_question (case-insensitive match).

    Same pattern as content_items.py:_build_competitor_snapshot. Returns
    None when no match — caller's responsibility (falls back to truncated
    question_text via YTG safety-net).
    """
    if not item.target_question:
        return None
    from sqlalchemy import func
    from models import ScanQuestion
    row = (
        db.query(ScanQuestion.id)
        .filter(
            ScanQuestion.scan_id == item.scan_id,
            func.lower(ScanQuestion.question) == item.target_question.strip().lower(),
        )
        .first()
    )
    return str(row[0]) if row else None


def _extract_url_domain(url: str | None) -> str:
    """Bare lowercase host from a URL (https://www.foo.com/bar → foo.com)."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or parsed.path or "").lower()
        return _normalize_domain(host)
    except Exception:
        return _normalize_domain(url)


def _html_to_text(html: str) -> str:
    """Strip HTML tags for the plain-text variant. Defensive — never raises."""
    if not html:
        return ""
    try:
        import re

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    except Exception:
        logger.exception("_html_to_text failed — returning raw HTML")
        return html


def _extract_outline_from_html(html: str) -> list[dict]:
    """Extract H2 headings as a lightweight outline structure for the
    validation page <details> view. Future C.2 may extend with per-section
    word counts or YTG scores."""
    if not html:
        return []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        outline = []
        for i, h2 in enumerate(soup.find_all("h2"), 1):
            outline.append({
                "level":    2,
                "position": i,
                "text":     h2.get_text(strip=True),
            })
        return outline
    except Exception:
        logger.exception("_extract_outline_from_html failed")
        return []


def _extract_sources_from_html(html: str) -> list[dict]:
    """Extract <a href> URLs from the generated article, deduplicated. Each
    entry carries `url`, `domain`, `anchor`, `org`, `type` — the shape the
    validation page UI expects (mirrors FAQ generator's `sources_used`)."""
    if not html:
        return []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        seen: set[str] = set()
        sources: list[dict] = []
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#") or href in seen:
                continue
            seen.add(href)
            try:
                netloc = urlparse(href).netloc or ""
            except Exception:
                netloc = ""
            netloc = _normalize_domain(netloc)
            anchor = a.get_text(strip=True)[:120]
            sources.append({
                "url":    href,
                "domain": netloc,
                "anchor": anchor,
                "org":    netloc,  # cosmetic — TODO C.2 enrich via trust_sources.details
                "type":   "reference",
            })
        return sources
    except Exception:
        logger.exception("_extract_sources_from_html failed")
        return []
