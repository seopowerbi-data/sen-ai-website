"""Handler : Sprint 12 crisis monitoring snapshot.

For each scan, count the negative brand_mentions[] for every
classified brand (my_brand + competitors), categorize each by content
theme (safety / efficacy / ingredients / pricing / service / quality /
other) via multilingual keyword heuristics, cluster by scan_topic, and
flag shared crises where target and a competitor are BOTH negative on
the same topic (= industry-wide signal, not target-only).

v1 is single-scan. No cross-scan trend / 3-sigma anomaly in v1 -
the value is making the negative LLM mentions visible per brand so the
user can triage and reach for the matching playbook. Cross-scan
trend = S12.1 once enough rescans accumulate per brand.

Cost : zero LLM. Pure SQL aggregation + Python regex categorization
over data already in the DB (brand_mentions[].sentiment + .contexte).
"""
from __future__ import annotations

import logging
import math
import re

from sqlalchemy import text as _text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

MAX_TOP_CONTEXTS = 5
MAX_TOPIC_CLUSTERS = 5
MAX_SHARED_WITH = 5

# Sentiment lookup is case-insensitive and accepts FR + EN.
NEGATIVE_SENTIMENTS = {"négatif", "negatif", "negative"}
POSITIVE_SENTIMENTS = {"positif", "positive"}
NEUTRAL_SENTIMENTS = {"neutre", "neutral"}

# Category keyword library. Multi-vertical : these phrases are intended
# to be generic enough to cover pharma / cosmetics / auto / BFSI / food.
# Categories are CHECKED IN ORDER ; first match wins. Higher-consequence
# categories (safety, ingredients) come first so a negative mention that
# touches both safety AND pricing gets safety.
CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("safety", [
        # FR
        "allergie", "allergique", "irritation", "irritant", "réaction",
        "danger", "dangereux", "risque", "risqué", "effet secondaire",
        "effets secondaires", "contre-indication", "contre indication",
        "toxique", "toxicité", "alerte", "rappel", "retrait",
        "intolérance", "anaphylaxie", "œdème", "brûlure",
        # EN
        "allergy", "allergic", "irritation", "irritant", "reaction",
        "danger", "dangerous", "risk", "side effect", "adverse",
        "recall", "warning", "alert", "intolerance", "anaphylaxis",
        "edema", "burn", "harmful", "unsafe",
    ]),
    ("ingredients", [
        "ingrédient", "ingredient", "composition", "formul",
        "perturbateur endocrinien", "endocrine disruptor",
        "paraben", "sulfate", "silicone", "alcool dénaturé",
        "huile minérale", "mineral oil", "fragrance", "parfum",
        "additif", "additive", "petrolatum", "pétrolatum",
        "préservateur", "preservative", "phenoxyethanol",
    ]),
    ("efficacy", [
        "inefficace", "ne marche pas", "marche pas", "sans effet",
        "sans résultat", "n'a pas marché", "decevant", "déçu", "deçoit",
        "pas convaincu", "pas convaincant", "aucun résultat",
        "ineffective", "doesn't work", "does not work", "no result",
        "no effect", "disappointing", "disappointed", "underwhelming",
        "unconvincing", "didn't work", "did not work",
    ]),
    ("pricing", [
        "trop cher", "cher pour ce que c'est", "prix abusif", "prix élevé",
        "rapport qualité prix", "qualité-prix", "coûteux", "onéreux",
        "expensive", "overpriced", "not worth", "poor value",
        "rip off", "ripoff", "too costly", "price gouging",
    ]),
    ("service", [
        "service client", "service après vente", "service apres vente",
        "sav", "livraison", "remboursement", "retour", "support",
        "customer service", "customer support", "delivery", "shipping",
        "refund", "return policy", "warranty",
    ]),
    ("quality", [
        "défaut", "casse", "fuite", "qualité médiocre", "qualité mauvaise",
        "mal fini", "fragile", "se casse", "moisi",
        "defect", "defective", "broken", "leak", "leaking", "poor quality",
        "low quality", "cheap quality", "fragile", "shoddy",
    ]),
]

# Severity bucket boundaries (composite 0-100).
SEVERITY_BUCKETS = [
    (0, "none"),
    (16, "low"),
    (36, "medium"),
    (61, "high"),
    (81, "critical"),
]


def _normalize_sentiment(raw: str | None) -> str | None:
    """Lowercase + map FR/EN to {'negative','positive','neutral',None}."""
    if not raw:
        return None
    s = raw.strip().lower()
    if s in NEGATIVE_SENTIMENTS:
        return "negative"
    if s in POSITIVE_SENTIMENTS:
        return "positive"
    if s in NEUTRAL_SENTIMENTS:
        return "neutral"
    return None


def _categorize(contexte: str | None, justification: str | None) -> str:
    """Pick the first category whose any keyword appears in the joined
    lower-cased text. Falls back to 'other'."""
    haystack = " ".join([contexte or "", justification or ""]).lower()
    if not haystack:
        return "other"
    for cat, kws in CATEGORY_KEYWORDS:
        for kw in kws:
            if kw in haystack:
                return cat
    return "other"


def _classified_brands(db: Session, scan_id: str) -> list[dict]:
    """Return [{brand_id, brand_name, canonical, aliases, classification}, ...]."""
    rows = db.execute(_text(
        """
        SELECT cb.id::text AS brand_id, cb.name, cb.canonical_name,
               cb.aliases, sbc.classification
          FROM scan_brand_classifications sbc
          JOIN client_brands cb ON cb.id = sbc.brand_id
         WHERE sbc.scan_id = :scan_id
           AND sbc.classification IN ('my_brand', 'competitor')
        """
    ), {"scan_id": scan_id}).fetchall()
    out = []
    for r in rows:
        names_lower = set()
        for n in [r.name, r.canonical_name, *(r.aliases or [])]:
            if n:
                names_lower.add(n.strip().lower())
        out.append({
            "brand_id": r.brand_id,
            "brand_name": r.canonical_name or r.name,
            "names_lower": names_lower,
            "classification": r.classification,
        })
    return out


def _brand_mentions_with_context(db: Session, scan_id: str) -> list[dict]:
    """One row per (slr, brand_mention). Returns the mention plus the
    question/topic/provider that triggered it, AND the latest Sentiment
    Judge verdict when present (only for negatives - the judge doesn't
    process positives or neutrals).

    The judge LEFT JOIN matches the latest judgement per (slr_id,
    mention_index) by judge_run_at DESC. Stale judgements (whose
    contexte_hash no longer matches the current mention contexte) are
    silently ignored - they belonged to a different mention that
    happened to share the same array slot before a re-run.
    """
    # Topic lookup goes through scan_personas (which carries the topic_id) :
    # scan_questions has only persona_id, not topic_id, on this DB schema.
    sql = _text(
        """
        SELECT slr.id::text          AS slr_id,
               slr.question_id::text AS question_id,
               sq.question            AS question,
               sp.topic_id::text      AS topic_id,
               st.name                AS topic_name,
               slr.provider           AS provider,
               mention_with_index.bm  AS mention,
               judged.judge_verdict   AS judge_verdict,
               judged.judged_sentiment AS judged_sentiment
          FROM scan_llm_results slr
          JOIN LATERAL jsonb_array_elements(slr.brand_mentions)
               WITH ORDINALITY AS mention_with_index(bm, idx) ON true
          LEFT JOIN scan_questions sq ON sq.id = slr.question_id
          LEFT JOIN scan_personas  sp ON sp.id = sq.persona_id
          LEFT JOIN scan_topics    st ON st.id = sp.topic_id
          LEFT JOIN LATERAL (
            SELECT j.judge_verdict, j.judged_sentiment, j.contexte_hash
              FROM scan_sentiment_judgements j
             WHERE j.slr_id = slr.id
               AND j.mention_index = (mention_with_index.idx - 1)::int
             ORDER BY j.judge_run_at DESC
             LIMIT 1
          ) AS judged ON true
         WHERE slr.scan_id = :scan_id
        """
    )
    return list(db.execute(sql, {"scan_id": scan_id}).fetchall())


def _severity_label(score: int | None) -> str:
    if score is None:
        return "none"
    label = "none"
    for cutoff, name in SEVERITY_BUCKETS:
        if score >= cutoff:
            label = name
    return label


def _compute_severity(
    negative_count: int,
    total_mentions: int,
    category_breakdown: dict[str, int],
    distinct_questions: int,
    distinct_providers: int,
) -> int:
    """0-100 composite. Tuned for sparse data (1-50 negative mentions) :
      40 pts volume    : log10(neg + 1) × 30 capped at 40
      30 pts ratio     : (neg / total) × 30
      15 pts severity  : safety / ingredients each worth +8 (cap 15)
      15 pts dispersion : log10(q_count + 1) × 8 + (providers - 1) × 4
    """
    nc = max(0, negative_count)
    volume = min(40, int(round(math.log10(nc + 1) * 30)))

    ratio = (nc / total_mentions) if total_mentions > 0 else 0.0
    ratio_pts = int(round(min(1.0, ratio) * 30))

    high_consequence = (
        category_breakdown.get("safety", 0) + category_breakdown.get("ingredients", 0)
    )
    consequence_pts = min(15, high_consequence * 8)

    dispersion = int(round(math.log10(max(0, distinct_questions) + 1) * 8))
    dispersion += max(0, distinct_providers - 1) * 4
    dispersion = min(15, dispersion)

    return max(0, min(100, volume + ratio_pts + consequence_pts + dispersion))


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Build the per-brand crisis snapshot for this scan.

    job_payload :
      - reset (bool) : drop existing rows for this scan before re-running
    """
    from models import Scan, ScanCrisisSignal

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    reset = bool(job_payload.get("reset"))
    if reset:
        db.query(ScanCrisisSignal).filter(ScanCrisisSignal.scan_id == scan_id).delete()
        db.commit()

    brands = _classified_brands(db, scan_id)
    if not brands:
        logger.info(f"build_crisis_radar: no classified brands for scan {scan_id}")
        return {"brands": 0, "negative_count_total": 0}

    # Index brand names -> brand_id for fast match.
    name_to_brand: dict[str, dict] = {}
    for b in brands:
        for nl in b["names_lower"]:
            # Earlier wins on collision so we don't overwrite my_brand
            # alias by a competitor sub-product alias.
            name_to_brand.setdefault(nl, b)

    rows = _brand_mentions_with_context(db, scan_id)

    # Accumulator per brand_id.
    buckets: dict[str, dict] = {
        b["brand_id"]: {
            "brand_id": b["brand_id"],
            "brand_name": b["brand_name"],
            "classification": b["classification"],
            "negative_count": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "total_mentions": 0,
            "category_breakdown": {},
            # negative mentions only :
            "_neg_contexts": [],
            "_neg_topics": {},  # topic_id -> {name, count}
            "_neg_questions": set(),
            "_neg_providers": set(),
        }
        for b in brands
    }

    # Resolve the focus brand explicitly so est_marque_cible=true mentions
    # land on it, not on whichever my_brand row sorted first. Without this
    # an arbitrary product-line variant (e.g. "physiolift nuit") swallowed
    # all 5000+ target mentions on the Avène scan while the master row
    # "eau thermale avene" only carried the 5 literal-name matches.
    focus_brand_id = (
        str(scan.focus_brand_id) if getattr(scan, "focus_brand_id", None) else None
    )
    focus_bucket_brand = None
    if focus_brand_id and focus_brand_id in buckets:
        focus_bucket_brand = next(b for b in brands if b["brand_id"] == focus_brand_id)

    # Pass 1 : tally + categorize negatives.
    for r in rows:
        bm = r.mention or {}
        brand_raw = (bm.get("brand_name") or "").strip().lower()
        if not brand_raw:
            continue
        # est_marque_cible=true rows always belong to the focus brand,
        # even when the brand_name string is a product alias. Fall back
        # to name match for competitor mentions OR when focus_brand_id
        # is missing (older scans without focus_brand wired).
        is_target_flag = bool(bm.get("est_marque_cible"))
        matched = None
        if is_target_flag and focus_bucket_brand is not None:
            matched = focus_bucket_brand
        elif is_target_flag:
            # Legacy fallback : first my_brand wins. Acceptable when the
            # scan has only one my_brand row.
            for b in brands:
                if b["classification"] == "my_brand":
                    matched = b
                    break
        if matched is None:
            matched = name_to_brand.get(brand_raw)
        if matched is None:
            continue

        bucket = buckets[matched["brand_id"]]
        bucket["total_mentions"] += 1
        raw_sentiment = _normalize_sentiment(bm.get("sentiment"))
        # Sentiment Judge override : if a judgement exists for this
        # mention and verdict='overturn' OR 'hedge', use the judged
        # sentiment instead of the BrandAnalyzer raw label. 'confirm'
        # keeps the raw label. This fixes the brand_analyzer false-
        # positive class (e.g. "X n'est pas destiné à Y" misclassified
        # as négatif when it's just usage clarification).
        verdict = getattr(r, "judge_verdict", None)
        if verdict in ("overturn", "hedge"):
            sentiment = _normalize_sentiment(getattr(r, "judged_sentiment", None)) or raw_sentiment
        else:
            sentiment = raw_sentiment
        if sentiment == "positive":
            bucket["positive_count"] += 1
        elif sentiment == "neutral":
            bucket["neutral_count"] += 1
        elif sentiment == "negative":
            bucket["negative_count"] += 1
            cat = _categorize(bm.get("contexte"), bm.get("sentiment_justification"))
            bucket["category_breakdown"][cat] = bucket["category_breakdown"].get(cat, 0) + 1
            bucket["_neg_questions"].add(r.question_id)
            if r.provider:
                bucket["_neg_providers"].add(r.provider)
            if r.topic_id:
                t = bucket["_neg_topics"].setdefault(r.topic_id, {
                    "topic_id": r.topic_id,
                    "topic_name": r.topic_name,
                    "negative_count": 0,
                })
                t["negative_count"] += 1
            bucket["_neg_contexts"].append({
                "contexte": (bm.get("contexte") or "")[:400],
                "sentiment_justification": (bm.get("sentiment_justification") or "")[:300],
                "question": r.question,
                "question_id": r.question_id,
                "topic_name": r.topic_name,
                "provider": r.provider,
                "slr_id": r.slr_id,
                "category": cat,
                "_consequence": 2 if cat in ("safety", "ingredients") else 1,
            })

    # Pass 2 : compute shared crises (target negative on same topic as a
    # competitor negative). Only filled for the target brand row.
    target_bucket = next(
        (b for b in buckets.values() if b["classification"] == "my_brand"),
        None,
    )
    shared_with_list: list[dict] = []
    if target_bucket and target_bucket["negative_count"] > 0:
        target_topic_ids = set(target_bucket["_neg_topics"].keys())
        for b in buckets.values():
            if b["classification"] != "competitor":
                continue
            if b["negative_count"] == 0:
                continue
            overlap_topics: list[dict] = []
            for tid, td in b["_neg_topics"].items():
                if tid in target_topic_ids:
                    overlap_topics.append({
                        "topic_id": tid,
                        "topic_name": td["topic_name"],
                        "competitor_negative_count": td["negative_count"],
                        "target_negative_count": target_bucket["_neg_topics"][tid]["negative_count"],
                    })
            if overlap_topics:
                shared_with_list.append({
                    "competitor_brand_name": b["brand_name"],
                    "competitor_brand_id": b["brand_id"],
                    "shared_topics": sorted(overlap_topics,
                                            key=lambda t: -t["competitor_negative_count"]),
                })
        shared_with_list.sort(key=lambda s: -sum(t["competitor_negative_count"] for t in s["shared_topics"]))
        shared_with_list = shared_with_list[:MAX_SHARED_WITH]

    # Pass 3 : materialize. Drop brands with zero mentions entirely.
    inserted = 0
    total_negatives = 0
    for bucket in buckets.values():
        if bucket["total_mentions"] == 0:
            continue
        neg = bucket["negative_count"]
        total = bucket["total_mentions"]
        ratio = (neg / total) if total > 0 else 0.0

        cat_breakdown = bucket["category_breakdown"]
        dominant = None
        if cat_breakdown:
            dominant = sorted(cat_breakdown.items(), key=lambda kv: -kv[1])[0][0]

        severity = _compute_severity(
            neg, total, cat_breakdown,
            len(bucket["_neg_questions"]), len(bucket["_neg_providers"]),
        )
        severity_label = _severity_label(severity)

        # Top contexts : consequence-weighted then by length.
        bucket["_neg_contexts"].sort(
            key=lambda c: (-c["_consequence"], -len(c.get("contexte") or "")),
        )
        top_contexts = []
        for c in bucket["_neg_contexts"][:MAX_TOP_CONTEXTS]:
            top_contexts.append({k: v for k, v in c.items() if not k.startswith("_")})

        # Topic clusters sorted by negative_count desc.
        clusters = sorted(
            bucket["_neg_topics"].values(),
            key=lambda t: -t["negative_count"],
        )[:MAX_TOPIC_CLUSTERS]

        shared = shared_with_list if bucket["classification"] == "my_brand" else []

        db.add(ScanCrisisSignal(
            scan_id=scan_id,
            brand_id=bucket["brand_id"],
            brand_classification=bucket["classification"],
            brand_name=bucket["brand_name"],
            negative_count=neg,
            positive_count=bucket["positive_count"],
            neutral_count=bucket["neutral_count"],
            total_mentions=total,
            negative_ratio=ratio,
            severity=severity,
            severity_label=severity_label,
            dominant_category=dominant,
            category_breakdown=cat_breakdown,
            top_contexts=top_contexts,
            topic_clusters=clusters,
            shared_with=shared,
        ))
        inserted += 1
        total_negatives += neg

    db.commit()
    logger.info(
        f"build_crisis_radar: scan {scan_id} -> {inserted} brand rows, "
        f"{total_negatives} negative mentions total"
    )
    return {
        "brands": inserted,
        "negative_count_total": total_negatives,
        "shared_with_count": len(shared_with_list),
    }
