"""Noise filter for LLM-extracted brand names.

BrandAnalyzer (seo_llm submodule) over-extracts brand-like strings from LLM
responses: ingredients ("acide hyaluronique"), product types ("BB crème"),
publications ("60 millions de consommateurs"), domain names ("aderma.fr"),
random tokens. On the 2026-05-19 Avène scan this produced 2 041 unclassified
rows, of which < 10% were real brands.

This module exposes one helper, `is_noise_brand_name(name)`, used in two
places:
  1. `worker/handlers/run_llm_tests.py` — before INSERT on `client_brands`,
     so the noise never enters the catalog.
  2. `worker/adapters/brand_classifier.py` — before sending to Claude,
     so the cleanup payload stays small even when noise slipped through
     (legacy data, race condition, regex miss).

The filter is intentionally conservative — false negatives (let noise through)
are preferable to false positives (drop a real brand). We rely on cleanup_brands
to catch what the regex misses.
"""

from __future__ import annotations

import re

# ── Hard-coded prefixes / patterns ────────────────────────────────────────

# Generic French product types — when a "brand" name starts with one of these,
# it's overwhelmingly an LLM hallucination ("crème hydratante", "gel moussant"
# detected as brands). A real product line keeps the brand name itself.
_PRODUCT_TYPE_PREFIXES: tuple[str, ...] = (
    "crème ",
    "creme ",
    "gel ",
    "sérum ",
    "serum ",
    "lotion ",
    "spray ",
    "stick ",
    "huile ",
    "mousse ",
    "lait ",
    "fluide ",
    "baume ",
    "shampooing ",
    "shampoing ",
    "soin ",
    "soins ",
    "masque ",
    "eau ",
    "fond de teint ",
    "bb crème ",
    "bb creme ",
    "cc crème ",
    "cc creme ",
    "gelée ",
    "gelee ",
)

# French ingredient prefixes — chemistry / botanicals get caught as "brands"
# when they're cited as components. Same conservative heuristic.
_INGREDIENT_PREFIXES: tuple[str, ...] = (
    "acide ",
    "vitamine ",
    "extrait ",
    "extrait de ",
    "complexe ",
    "huile de ",
    "essence de ",
    "beurre de ",
    "eau de ",
)

# Domain TLDs that signal a URL extracted as brand name.
_DOMAIN_TLD_RE = re.compile(
    r"\.(com|fr|net|org|io|eu|co|tv|me|de|es|it|be|ch|ca|uk)\b", re.IGNORECASE
)

# Looks like a hash / random alphanumeric jumble (8+ mixed-case + digits
# without spaces). LLMs sometimes echo IDs verbatim.
_HASH_LIKE_RE = re.compile(r"^[A-Za-z0-9]{12,}$")

# Pure number or starts with digit + word (ex: "60 millions de consommateurs",
# "4-methylbenzylidene camphor"). Real brands rarely lead with a digit.
_LEADING_DIGIT_PHRASE_RE = re.compile(r"^\d+[\s\-]")

# Common French articles / connectors that shouldn't be a brand on their own
_STOPWORDS: frozenset[str] = frozenset({
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "ce",
    "cette", "ces", "mon", "ma", "mes", "son", "sa", "ses", "votre", "notre",
    "the", "a", "an", "and", "or",
})

# Length thresholds.
_MIN_LEN = 2
_MAX_LEN = 60  # Beyond this it's almost certainly a phrase, not a brand


def is_noise_brand_name(name: str) -> bool:
    """Return True if `name` looks like LLM noise rather than a real brand.

    Caller should SKIP creating a `client_brands` row when this returns True.
    """
    if not name:
        return True
    s = name.strip()
    if len(s) < _MIN_LEN or len(s) > _MAX_LEN:
        return True

    low = s.lower()

    # Single-token stop word
    if low in _STOPWORDS:
        return True

    # Domain / URL
    if _DOMAIN_TLD_RE.search(low):
        return True

    # Hash-like blob
    if _HASH_LIKE_RE.match(s):
        return True

    # Leading digit phrase
    if _LEADING_DIGIT_PHRASE_RE.match(s):
        return True

    # Generic product-type or ingredient prefix
    for prefix in _PRODUCT_TYPE_PREFIXES + _INGREDIENT_PREFIXES:
        if low.startswith(prefix):
            return True

    # Pure punctuation / single chars after strip
    if not re.search(r"[A-Za-zÀ-ÿ]", s):
        return True

    return False


def filter_noise(names: list[str]) -> tuple[list[str], list[str]]:
    """Split a list of candidate names into (real, noise) using `is_noise_brand_name`.

    Returns the lists in the input order so caller can preserve any side metadata.
    """
    real: list[str] = []
    noise: list[str] = []
    for n in names or []:
        (noise if is_noise_brand_name(n) else real).append(n)
    return real, noise
