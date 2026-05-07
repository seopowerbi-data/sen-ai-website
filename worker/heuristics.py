"""Heuristic garde-fous applied to LLM-generated questions.

Two checks ported from seo-llm/src/question_generator.py to harden the SaaS
flow against LLM omissions and quirks :

  1. infer_question_type(text) — fallback type_question via FR keyword matching
     when Claude returns missing/invalid values. Avoids dropping otherwise-valid
     questions in Pydantic Literal validation.

  2. validate_question_coherence(personas) — flags two classes of suspicious
     questions and returns warnings (does NOT delete or reject) :
       - off_topic   : question text shares no stem with ANY of the persona's
                       semantic anchors (topic name, persona name, mots_cles_associes,
                       intentions_recherche, points_douleur — all expanded to
                       individual words and accent-stripped 6-char prefixes).
       - duplicate   : Jaccard similarity >= 0.7 between two questions in the
                       same persona (quasi-doublons that waste scan credits)

Off-topic detection v2 (2026-05-07) : the v1 substring match on mots_cles_associes
alone produced 87% false positives because Claude generates natural questions
("boutons", "peau grasse") while persona keywords are SEO-specific ("cicalfate",
"rétinol"). v2 widens the anchor set + matches stems instead of phrases,
expected false positive rate ~5-10% on real scans.

Warnings are stored in `scan.summary["warnings"]` (JSONB) and surfaced in the
UI Personas page as an orange badge with expandable list. The user can edit /
toggle the flagged questions BEFORE launching the scan — preserving the
SaaS editability invariant.
"""

from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger(__name__)

VALID_QUESTION_TYPES: set[str] = {
    "basique", "validation", "comparative", "technique", "urgente",
}

# FR keyword sets for type inference. Order of evaluation in infer_question_type
# matters: most specific (comparative) wins over more generic (technique).
_COMPARATIVE_KEYWORDS = (
    "compar", "versus", " vs ", "différence", "difference", "meilleur",
    "préférer", "preferer", "choix entre", "ou bien", "plutôt", "plutot",
    "avantage", "inconvénient", "inconvenient", "alternative",
)
_URGENTE_KEYWORDS = (
    "urgent", "rapidement", "vite", "immédiat", "immediat", "secours",
    "crise", "grave", "minuit", "dernière minute", "derniere minute",
)
_TECHNIQUE_KEYWORDS = (
    "comment", "pourquoi", "expliquer", "fonctionn", "mécanisme", "mecanisme",
    "processus", "ingrédient", "ingredient", "formul", "composition",
    "actif", "principe",
)
_VALIDATION_KEYWORDS = (
    "est-ce que", "est-il", "peut-on", "dois-je", "faut-il", "recommand",
    "conseill", "avis", "efficace", "marche", "fonctionne", "j'ai entendu",
    "j ai entendu",
)

# Jaccard threshold above which two same-persona questions are flagged as duplicates.
# 0.7 = 70% word overlap. Empirically ported from seo-llm.
_DUPLICATE_JACCARD_THRESHOLD = 0.7

# Words shorter than this are excluded from Jaccard / coherence checks (FR stopwords noise).
_MIN_WORD_LEN = 3

# Stem length for off-topic anchor matching. 6 chars catches:
#   "boutons" → "bouton" (matches "bouton" anchor)
#   "acné" → "acne" (after accent strip, matches "acne" anchor)
#   "imperfections" → "imperf" (matches "imperf" anchor from "imperfection")
# Shorter (4-5) = too many false negatives on common French roots.
_STEM_LEN = 6


def _strip_accents(text: str) -> str:
    """Remove diacritics (acné → acne) for fuzzy matching."""
    if not isinstance(text, str):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _stems(text: str) -> set[str]:
    """Lowercase + accent-strip + split + filter short words + take 6-char prefix.

    Used to compare question text against persona/topic anchors with tolerance
    for plurals, conjugations, and accent variations.
    """
    if not text:
        return set()
    cleaned = _strip_accents(text.lower())
    # Replace punctuation/separators with spaces
    for ch in "'-/,.;:!?()[]{}\"":
        cleaned = cleaned.replace(ch, " ")
    out: set[str] = set()
    for word in cleaned.split():
        word = "".join(c for c in word if c.isalnum())
        if len(word) >= _MIN_WORD_LEN:
            out.add(word[:_STEM_LEN])
    return out


def infer_question_type(question: str) -> str:
    """Infer one of the 5 canonical types from the question text.

    Returns "basique" as fallback when no specific keyword matches.
    Order : comparative > urgente > technique > validation > basique
    (more-specific patterns checked first).
    """
    if not question or not isinstance(question, str):
        return "basique"
    q = question.lower()
    if any(kw in q for kw in _COMPARATIVE_KEYWORDS):
        return "comparative"
    if any(kw in q for kw in _URGENTE_KEYWORDS):
        return "urgente"
    if any(kw in q for kw in _TECHNIQUE_KEYWORDS):
        return "technique"
    if any(kw in q for kw in _VALIDATION_KEYWORDS):
        return "validation"
    return "basique"


def normalize_question_types(personas: list[dict]) -> list[dict]:
    """For each question in each persona, infer type_question if missing/invalid.

    Mutates personas IN PLACE (replaces invalid type_question values with inferred ones)
    AND returns a list of warning dicts describing each inference.

    Called BEFORE Pydantic validation so that no question gets dropped just because
    Claude omitted or misspelled the type field.
    """
    warnings: list[dict] = []
    for p in personas:
        if not isinstance(p, dict):
            continue
        persona_name = (p.get("nom") or "?").strip() or "?"
        for q in p.get("questions") or []:
            if not isinstance(q, dict):
                continue
            current = q.get("type_question")
            current_norm = current.strip().lower() if isinstance(current, str) else ""
            if current_norm in VALID_QUESTION_TYPES:
                continue
            inferred = infer_question_type(q.get("question") or "")
            q["type_question"] = inferred
            warnings.append({
                "type": "type_inferred",
                "persona": persona_name,
                "question": (q.get("question") or "")[:160],
                "original": current if current is not None else "(missing)",
                "inferred": inferred,
            })
    return warnings


def _word_set(text: str) -> set[str]:
    """Lowercase + split + drop short words. Used by Jaccard comparator."""
    if not text:
        return set()
    return {w for w in text.lower().split() if len(w) >= _MIN_WORD_LEN}


def detect_off_topic_questions(persona: dict, questions: list[dict]) -> list[dict]:
    """Flag questions that share no stem with the persona's semantic context.

    v2 (broad anchors): builds the "anchor stem set" from the union of:
      - topic name (via persona.segment_principal)
      - persona name (often contains topic words like "acnéique")
      - persona.mots_cles_associes (Claude's chosen SEO keywords)
      - persona.intentions_recherche (Claude's semantic intents)
      - persona.points_douleur (pain points often share vocabulary with questions)

    A question is flagged ONLY if its stem set is fully disjoint from this anchor
    set. This catches genuine drifts ("question about cooking in a skincare
    persona") while tolerating natural synonyms ("boutons" for "acné", "rougeurs"
    for "rosacée") that share stems via word boundaries.

    Heuristic only — does NOT drop. Surfaced in UI for user review.
    """
    # Collect all anchor stems
    anchors: set[str] = set()
    anchors |= _stems(persona.get("nom"))
    anchors |= _stems(persona.get("segment_principal"))
    for kw in (persona.get("mots_cles_associes") or []):
        anchors |= _stems(kw)
    for intent in (persona.get("intentions_recherche") or []):
        anchors |= _stems(intent)
    for pain in (persona.get("points_douleur") or []):
        anchors |= _stems(pain)

    if not anchors:
        return []  # Nothing to compare against — skip rather than over-flag

    persona_name = (persona.get("nom") or "?").strip() or "?"
    out: list[dict] = []
    for q in questions or []:
        q_text = q.get("question", "") if isinstance(q, dict) else ""
        if not q_text:
            continue
        q_stems_set = _stems(q_text)
        if not q_stems_set:
            continue
        # Match if there's ANY overlap. Disjoint = off-topic.
        if not (anchors & q_stems_set):
            out.append({
                "type": "off_topic",
                "persona": persona_name,
                "question": q_text[:160],
                "reason": "question shares no semantic anchor with persona/topic context",
            })
    return out


def detect_duplicate_questions(persona: dict, questions: list[dict],
                               threshold: float = _DUPLICATE_JACCARD_THRESHOLD) -> list[dict]:
    """Flag pairs of same-persona questions whose word-set Jaccard >= threshold.

    Pairs are reported once (i, j) with i < j. The user can then decide to
    rewrite or disable one of the two.
    """
    persona_name = (persona.get("nom") or "?").strip() or "?"
    word_sets = []
    texts = []
    for q in questions or []:
        q_text = q.get("question", "") if isinstance(q, dict) else ""
        word_sets.append(_word_set(q_text))
        texts.append(q_text)
    out: list[dict] = []
    n = len(word_sets)
    for i in range(n):
        if not word_sets[i]:
            continue
        for j in range(i + 1, n):
            if not word_sets[j]:
                continue
            union = word_sets[i] | word_sets[j]
            if not union:
                continue
            inter = word_sets[i] & word_sets[j]
            jaccard = len(inter) / len(union)
            if jaccard >= threshold:
                out.append({
                    "type": "duplicate",
                    "persona": persona_name,
                    "question_a": texts[i][:160],
                    "question_b": texts[j][:160],
                    "jaccard": round(jaccard, 2),
                })
    return out


def validate_question_coherence(personas: list[dict]) -> list[dict]:
    """Run all coherence checks across all personas and return a flat warning list.

    Each persona dict must have at minimum :
      { "nom": str,
        "mots_cles_associes": [str, ...],
        "questions": [{"type_question": str, "question": str}, ...] }

    Returns a flat list of warning dicts ready to persist in scan.summary["warnings"].
    """
    out: list[dict] = []
    for p in personas or []:
        if not isinstance(p, dict):
            continue
        questions = p.get("questions") or []
        out.extend(detect_off_topic_questions(p, questions))
        out.extend(detect_duplicate_questions(p, questions))
    return out


def summarize_warnings(warnings: list[dict]) -> dict[str, int]:
    """Count warnings by type — useful for UI badge labels."""
    counts: dict[str, int] = {}
    for w in warnings or []:
        t = w.get("type") or "unknown"
        counts[t] = counts.get(t, 0) + 1
    return counts
