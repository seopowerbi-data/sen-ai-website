"""Sprint E: broaden brand_analyzer detection to 5 entity types.

The PDF SEO LLM Nov 2025 framework distinguishes 3 axes of LLM-response
observation : citation source / brand mention / entity product-gamme. Pre-Sprint E
sen-ai collapsed these into one BRAND_ANALYSIS_PROMPT that treated products,
gammes and domains as the same "brand_name" bucket, and EXCLUDED authority
sources (Wikipédia / Doctissimo / Ameli) entirely (see seo_llm BRAND_ANALYSIS_PROMPT
lines 599-606 of the legacy prompt).

This module re-introduces all 3 axes with explicit per-entry typing :

    brand          La Roche-Posay, Avène, Pierre Fabre — corporate
    product        Effaclar Duo, Cicaplast B5 — specific item
    range          Effaclar, Anthelios, Anaphase — gamme / line
    domain         laroche-posay.fr, amazon.com, sephora.fr — site cited
    expert_source  Wikipédia, Doctissimo, Ameli.fr — non-commercial authority

The wire format on ScanLLMResult.brand_mentions stays identical (same JSONB
column, same `brand_name` / `est_marque_cible` key names) for backward
compatibility with downstream consumers (api serializer, UI templates,
opportunities scorer). Each entry now CARRIES two additional fields :

    entity_type  : one of the 5 categories above
    parent_brand : populated when entity_type is product or range
                   (e.g. "effaclar duo" -> parent_brand: "la roche-posay")

`est_marque_cible` resolution is now typed : a product matches against the
target_products list (not target_brands), so "Effaclar Duo" without the
parent brand cited still counts as a target hit if the client owns that
product line. This fixes the 30-50% mention loss the memo flagged for
clients pushing specific gammes.

The class deliberately does NOT subclass seo_llm.src.brand_analyzer.BrandAnalyzer
even though it ports several of its helpers — the seo_llm submodule API is
not stable for inheritance (it can change underneath us on a bump) and the
cleanup logic here diverges enough that override-by-method would be a sieve.
Self-contained > tightly-coupled.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any

from seo_llm.src.llm_client import LLMClient

logger = logging.getLogger(__name__)


# Sentinel used in the prompt when a category list is empty — keeps the
# prompt readable instead of dropping the whole line.
_NONE_STR = "aucune"


# Match the seo_llm cap (30K) — same Gemini 2.5 budget, output JSON gets
# longer with the new entity_type / parent_brand fields per entry.
_MAX_OUTPUT_TOKENS = 30000


ENTITY_ANALYSIS_PROMPT = """Analyse le texte suivant pour identifier les ENTITÉS pertinentes citées (marques, produits, gammes, domaines, sources expertes).

# Texte à analyser
{response_text}

# Contexte
Question posée : {question}
{target_block}
{known_block}
{domain_context}

# Types d'entités à classer (entity_type)

- **brand**         : Marque commerciale, laboratoire, entreprise nommée (ex: La Roche-Posay, Avène, Pierre Fabre, Galderma)
- **product**       : Produit nommé spécifique (ex: Effaclar Duo, Cicaplast Baume B5, Anaphase Shampoing). Plus précis qu'une gamme.
- **range**         : Gamme/ligne nommée (ex: Effaclar, Anthelios, Cicaplast — une famille de produits, PAS un produit individuel)
- **domain**        : Domaine ou site web cité (ex: laroche-posay.fr, amazon.com, sephora.fr — commerce, fabricant, distributeur)
- **expert_source** : Source d'information / autorité éditoriale non commerciale (Wikipédia, Doctissimo, Ameli.fr, Santé Magazine, blog médical reconnu, journal scientifique)

# Ce qui N'EST PAS une entité — NE PAS EXTRAIRE
- Ingrédients actifs / molécules : acide hyaluronique, niacinamide, rétinol, zinc, vitamine C, céramides, panthénol, paracétamol, ibuprofène...
- Matières premières : coton, polyester, nylon, soie...
- Catégories génériques sans nom propre : crème, gel, sérum, mascara, shampooing, baume, lait, dentifrice...
- Mots courants : il, bio, nature, expert, eau, miel, citron...
- Acronymes techniques : SPF, AHA, BHA, UV, FPS...

# Pour CHAQUE entité extraite, fournis tous ces champs :

1. **entity_type**           : un des 5 codes ci-dessus
2. **brand_name**            : nom exact tel qu'apparaît, en MINUSCULES (clé de jointure côté code)
3. **parent_brand**          : Si entity_type=product OU range, la marque mère en minuscules (ex: "Effaclar Duo" → "la roche-posay" ; "Anthelios" → "la roche-posay"). null pour brand/domain/expert_source.
4. **position_index**        : Ordre d'apparition (1 pour la première, 2 pour la deuxième...)
5. **position_type**         : debut|milieu|fin (3 tiers de la réponse)
6. **contexte**              : Phrase contenant la mention (≤150 chars). DOIT contenir le brand_name littéralement. Si la troncature couperait le nom, ajuster.
7. **sentiment**             : positif|neutre|negatif (critères stricts)
   - positif : explicitement recommandé, loué, décrit favorablement ("efficace", "excellent", "meilleur", "recommandé")
   - negatif : critiqué, déconseillé, défavorablement ("à éviter", "irritant", "décevant", "ne pas utiliser")
   - neutre : mention purement factuelle SANS jugement ("X propose une gamme", "X existe depuis 1983")
   ATTENTION : si jugement de valeur même léger, ce N'EST PAS neutre.
8. **sentiment_justification** : explication courte (≤50 chars), ex "recommandé peau sensible", "simple liste", "critiqué pour le prix"
9. **est_recommandation**    : true|false — est-ce une recommandation explicite ?
10. **type_recommandation**  : premiere_option|alternative|a_eviter|mention_simple

# Format JSON STRICT (sans markdown, sans prose)

{{
  "entities": [
    {{
      "entity_type": "brand",
      "brand_name": "la roche-posay",
      "parent_brand": null,
      "position_index": 1,
      "position_type": "debut",
      "contexte": "...",
      "sentiment": "positif",
      "sentiment_justification": "recommandé peau sensible",
      "est_recommandation": true,
      "type_recommandation": "premiere_option"
    }},
    {{
      "entity_type": "product",
      "brand_name": "effaclar duo",
      "parent_brand": "la roche-posay",
      "position_index": 2,
      "position_type": "milieu",
      "contexte": "...",
      "sentiment": "positif",
      "sentiment_justification": "cité pour acné",
      "est_recommandation": true,
      "type_recommandation": "premiere_option"
    }}
  ],
  "resume": {{"nb_entities": 5, "nb_positifs": 3, "nb_negatifs": 0}}
}}

# Règles de désambiguïsation cruciales

- Une **range** est une famille de produits. Un **product** est un item nommé spécifique de cette famille. Si l'IA dit simplement "Effaclar" sans suffixe, c'est range. Si elle dit "Effaclar Duo" ou "Effaclar Adapalène Gel", c'est product.
- Si une marque ET son produit sont cités dans la même phrase, sortir DEUX entrées distinctes (une pour entity_type=brand, une pour entity_type=product avec parent_brand pointant vers la marque).
- Wikipédia, Doctissimo, Ameli.fr = expert_source. Amazon, Sephora, Yves Rocher = domain (commerce, pas autorité éditoriale).
- Un nom de domaine en .fr/.com/etc. classé en domain — JAMAIS en brand.
- Chaque brand_name DOIT apparaître LITTÉRALEMENT dans le texte (case-insensitive). NE PAS inventer.
- Si aucune entité, retourner "entities": [].
- Répondre UNIQUEMENT en JSON, sans markdown ni commentaires."""


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )


def _norm(s: str) -> str:
    """Lowercase + strip + accent-fold for fuzzy matching keys."""
    return _strip_accents((s or "").strip().lower())


def _format_target_block(target_entities: dict[str, list[str]]) -> str:
    """Render the target_entities dict as a multi-line prompt block.

    Empty categories are omitted to keep the prompt small ; entities in
    each category get joined comma-separated.
    """
    parts: list[str] = []
    label_map = {
        "brands": "Marques cibles",
        "products": "Produits cibles",
        "ranges": "Gammes cibles",
        "domains": "Domaines cibles",
        "expert_sources": "Sources expertes connues",
    }
    for key, label in label_map.items():
        vals = target_entities.get(key) or []
        vals = [v for v in (s.strip() for s in vals) if v]
        if vals:
            parts.append(f"  {label} : {', '.join(vals)}")
    if not parts:
        return "Entités cibles : aucune"
    return "Entités cibles (à valoriser) :\n" + "\n".join(parts)


def _format_known_block(known_entities: list[str]) -> str:
    """Known entities of the sector — used so the LLM doesn't miss obvious
    competitors. Capped at 30 to bound prompt size.
    """
    vals = [v.strip() for v in (known_entities or []) if v and v.strip()]
    if not vals:
        return ""
    return f"Entités du secteur connues : {', '.join(vals[:30])}"


class EntityAnalyzer:
    """Broader-axis analyzer extending brand detection to 5 entity types.

    Wire-compatible with the existing ScanLLMResult.brand_mentions JSONB
    column — each entry adds `entity_type` and `parent_brand` fields,
    keeps `brand_name` + `est_marque_cible` for downstream consumers.

    Args:
        llm_client     : seo_llm LLMClient — typically Gemini Flash Lite
                         (cheapest of the supported models for this task,
                         matches the legacy BrandAnalyzer choice).
        target_entities: dict with keys brands / products / ranges /
                         domains / expert_sources, each a list of names.
                         The brand_resolver builds this from ClientBrand
                         hierarchy + scan.domain + ScanTrustSource.
        known_entities : flat list of names that exist in the sector but
                         aren't strictly targets — pre-warms the LLM so
                         it doesn't miss the obvious competitors.
        domain_context : structured domain_brief text from generate_domain_brief.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        target_entities: dict[str, list[str]],
        known_entities: list[str] | None = None,
        domain_context: str = "",
    ):
        self.llm = llm_client
        self.target_entities = {
            "brands": [s.strip() for s in target_entities.get("brands") or [] if s],
            "products": [s.strip() for s in target_entities.get("products") or [] if s],
            "ranges": [s.strip() for s in target_entities.get("ranges") or [] if s],
            "domains": [s.strip() for s in target_entities.get("domains") or [] if s],
            "expert_sources": [s.strip() for s in target_entities.get("expert_sources") or [] if s],
        }
        self.known_entities = [s.strip() for s in (known_entities or []) if s and s.strip()]
        self.domain_context = domain_context

        # Pre-build normalized lookup sets for est_cible resolution.
        self._target_norm: dict[str, set[str]] = {
            k: {_norm(v) for v in vals} for k, vals in self.target_entities.items()
        }

        total_targets = sum(len(v) for v in self.target_entities.values())
        logger.info(
            f"EntityAnalyzer initialized: {total_targets} target entities across 5 types, "
            f"{len(self.known_entities)} known"
        )

    # ── est_cible resolution ────────────────────────────────────────────

    def _resolve_est_cible(self, entity_type: str, name: str) -> bool:
        """Typed targeting check.

        A product matches against target_products only (not against
        target_brands — that would conflate gamme/brand axes). Domains
        match by suffix-or-equal so "shop.laroche-posay.fr" hits the
        "laroche-posay.fr" target. Expert sources are never "yours" by
        definition (they're external authorities).
        """
        if not name:
            return False
        n = _norm(name)
        if entity_type == "expert_source":
            return False
        if entity_type == "domain":
            for t in self._target_norm["domains"]:
                if not t:
                    continue
                if n == t or n.endswith("." + t) or t.endswith("." + n):
                    return True
            return False
        # brand / product / range share substring-tolerant matching
        # (legacy BrandAnalyzer behavior). Keep that for fuzzy cases like
        # "Avène thermal water" matching target "Avène".
        bucket_key = {
            "brand": "brands",
            "product": "products",
            "range": "ranges",
        }.get(entity_type)
        if not bucket_key:
            return False
        for t in self._target_norm[bucket_key]:
            if not t:
                continue
            if t in n or n in t:
                return True
        return False

    # ── context validation (anti-hallucination) ─────────────────────────

    def _validate_context(self, mention: dict, response_text: str) -> dict:
        """Same pattern as BrandAnalyzer._validate_context — ensure the
        brand_name appears literally in the contexte slice ; if not,
        relocate the real context from response_text or drop the mention.
        """
        brand_name = (mention.get("brand_name") or "").lower().strip()
        contexte = (mention.get("contexte") or "").lower()

        if not brand_name:
            mention["contexte_valide"] = False
            return mention

        # Accent-fold both sides so "Avène" matches its variant "Avene"
        variants = {brand_name, _strip_accents(brand_name)}
        context_valid = any(v in contexte for v in variants if v)

        if not context_valid:
            response_lower = response_text.lower()
            for v in variants:
                if not v:
                    continue
                match = re.search(re.escape(v), response_lower)
                if match:
                    start = max(0, match.start() - 75)
                    end = min(len(response_text), match.end() + 75)
                    if start > 0:
                        space = response_text.rfind(" ", start - 20, start + 10)
                        if space > 0:
                            start = space + 1
                    if end < len(response_text):
                        space = response_text.find(" ", end - 10, end + 20)
                        if space > 0:
                            end = space
                    new_ctx = response_text[start:end].strip()
                    if len(new_ctx) > 150:
                        new_ctx = new_ctx[:147] + "..."
                    mention["contexte"] = new_ctx
                    mention["contexte_valide"] = True
                    mention["contexte_corrige"] = True
                    return mention

        mention["contexte_valide"] = context_valid
        return mention

    # ── aggregation : dedupe by (entity_type, name) ─────────────────────

    def _aggregate_entities(self, entities: list[dict]) -> list[dict]:
        """Dedupe by (entity_type, brand_name) tuple.

        Pre-Sprint E the seo_llm aggregator dedupes by brand_name alone,
        which would merge a `range="Effaclar"` mention with a separate
        `product="Effaclar Duo"` mention — collapsing the gamme/product
        distinction we just introduced. Keying by the tuple preserves it.
        """
        if not entities:
            return []

        groups: dict[tuple[str, str], list[dict]] = {}
        for m in entities:
            key = ((m.get("entity_type") or "brand").lower(),
                   (m.get("brand_name") or "").lower().strip())
            if not key[1]:
                continue
            groups.setdefault(key, []).append(m)

        aggregated: list[dict] = []
        for (etype, name_key), mentions in groups.items():
            mentions_sorted = sorted(mentions, key=lambda m: m.get("position_index") or 999)
            first = mentions_sorted[0]

            # Dominant sentiment (same priority as legacy: positif > negatif > neutre).
            counts = {"positif": 0, "neutre": 0, "negatif": 0}
            for m in mentions:
                s = m.get("sentiment", "neutre")
                if s in counts:
                    counts[s] += 1
            max_c = max(counts.values())
            if counts["positif"] == max_c:
                dom = "positif"
            elif counts["negatif"] == max_c:
                dom = "negatif"
            else:
                dom = "neutre"

            is_rec = any(m.get("est_recommandation", False) for m in mentions)
            rec_priority = {
                "premiere_option": 4, "alternative": 3,
                "a_eviter": 2, "mention_simple": 1,
            }
            best_rec = max(
                mentions, key=lambda m: rec_priority.get(m.get("type_recommandation", ""), 0)
            ).get("type_recommandation", "mention_simple")

            aggregated.append({
                "brand_name": first.get("brand_name", name_key),
                "entity_type": etype,
                "parent_brand": first.get("parent_brand"),
                "position_index": first.get("position_index", 1),
                "position_type": first.get("position_type", "debut"),
                "contexte": first.get("contexte", ""),
                "contexte_valide": first.get("contexte_valide", True),
                "contexte_corrige": first.get("contexte_corrige", False),
                "sentiment": dom,
                "sentiment_justification": first.get("sentiment_justification", ""),
                "est_recommandation": is_rec,
                "type_recommandation": best_rec,
                "est_marque_cible": first.get("est_marque_cible", False),
                "nb_mentions": len(mentions),
            })

        aggregated.sort(key=lambda m: m.get("position_index", 999))
        return aggregated

    # ── summary stats ───────────────────────────────────────────────────

    def _compute_analysis(self, entities: list[dict], resume: dict) -> dict[str, Any]:
        """Schema-compatible with the legacy brand_analyse dict — same keys,
        but the counts now span all entity_types. Adds a per-type breakdown
        so Sprint M dashboards can split SOV by axis without re-aggregating.
        """
        target_mentions = [m for m in entities if m.get("est_marque_cible")]
        first_target = target_mentions[0] if target_mentions else None
        total_count = sum(m.get("nb_mentions", 1) for m in entities)
        target_count = sum(m.get("nb_mentions", 1) for m in target_mentions)

        # Per-type breakdown — keyed by entity_type, value is uniq-count.
        by_type: dict[str, int] = {}
        target_by_type: dict[str, int] = {}
        for m in entities:
            t = m.get("entity_type", "brand")
            by_type[t] = by_type.get(t, 0) + 1
            if m.get("est_marque_cible"):
                target_by_type[t] = target_by_type.get(t, 0) + 1

        return {
            # Legacy keys (consumed by api/routers/scans.py, content templates,
            # generate_opportunities) — kept identical so this is a drop-in.
            "nb_marques": len(entities),
            "nb_marques_cibles": len(target_mentions),
            "nb_total_mentions": total_count,
            "nb_mentions_marque_cible": target_count,
            "nb_positifs": sum(1 for m in entities if m.get("sentiment") == "positif"),
            "nb_negatifs": sum(1 for m in entities if m.get("sentiment") == "negatif"),
            "marque_cible_mentionnee": len(target_mentions) > 0,
            "position_marque_cible": first_target.get("position_index") if first_target else None,
            "sentiment_marque_cible": first_target.get("sentiment") if first_target else None,
            "recommandation_marque_cible": first_target.get("est_recommandation", False) if first_target else False,
            # Sprint E new keys — per-axis SOV split.
            "nb_by_entity_type": by_type,
            "nb_targets_by_entity_type": target_by_type,
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "brand_mentions": [],
            "brand_analyse": {
                "nb_marques": 0, "nb_marques_cibles": 0,
                "nb_total_mentions": 0, "nb_mentions_marque_cible": 0,
                "nb_positifs": 0, "nb_negatifs": 0,
                "marque_cible_mentionnee": False,
                "position_marque_cible": None,
                "sentiment_marque_cible": None,
                "recommandation_marque_cible": False,
                "nb_by_entity_type": {},
                "nb_targets_by_entity_type": {},
            },
            "llm_usage": {},
            "llm_cost": {},
        }

    # ── public entry point ──────────────────────────────────────────────

    def analyze_response(
        self, response_text: str, question: str
    ) -> dict[str, Any] | None:
        """Wire-compatible with seo_llm BrandAnalyzer.analyze_response.

        Returns the same dict shape (`brand_mentions` + `brand_analyse` +
        `llm_usage` + `llm_cost`) so callers don't change. Each entry in
        `brand_mentions` now carries `entity_type` + `parent_brand`.
        """
        try:
            prompt = ENTITY_ANALYSIS_PROMPT.format(
                response_text=response_text,
                question=question,
                target_block=_format_target_block(self.target_entities),
                known_block=_format_known_block(self.known_entities),
                domain_context=self.domain_context or "",
            )

            response = self.llm.generate(
                prompt,
                temperature=0.0,
                max_tokens=_MAX_OUTPUT_TOKENS,
                json_mode=True,
                agent_name="entity_analysis",
            )

            result = self.llm.extract_json(response["text"])
            raw_entities = result.get("entities") or []
            resume = result.get("resume") or {}

            # 1. est_marque_cible — typed resolution
            # 2. parent_brand — lowercase normalize
            # 3. context validation (drop hallucinations)
            normalized: list[dict] = []
            for m in raw_entities:
                if not isinstance(m, dict):
                    continue
                etype = (m.get("entity_type") or "brand").strip().lower()
                if etype not in ("brand", "product", "range", "domain", "expert_source"):
                    etype = "brand"  # safe default
                m["entity_type"] = etype
                parent = m.get("parent_brand")
                m["parent_brand"] = (parent or "").strip().lower() or None
                m["est_marque_cible"] = self._resolve_est_cible(
                    etype, m.get("brand_name", "")
                )
                self._validate_context(m, response_text)
                normalized.append(m)

            # Drop invalid (hallucinated contexte that we couldn't relocate)
            valid = [m for m in normalized if m.get("contexte_valide", True)]
            invalid_count = len(normalized) - len(valid)
            if invalid_count > 0:
                logger.warning(
                    f"EntityAnalyzer: dropped {invalid_count} hallucinated entities "
                    f"(contexte not found in response)"
                )

            # Heuristic filters (port from BrandAnalyzer)
            valid = [
                m for m in valid
                if len((m.get("brand_name") or "").strip()) > 2
            ]

            aggregated = self._aggregate_entities(valid)
            analyse = self._compute_analysis(aggregated, resume)

            logger.debug(
                f"EntityAnalyzer: {len(aggregated)} entities, "
                f"target_cible_mentionnee={analyse['marque_cible_mentionnee']}, "
                f"by_type={analyse['nb_by_entity_type']}"
            )

            return {
                "brand_mentions": aggregated,
                "brand_analyse": analyse,
                "llm_usage": response.get("usage", {}),
                "llm_cost": response.get("cost", {}),
            }
        except json.JSONDecodeError as e:
            logger.error(f"EntityAnalyzer JSON parse failed: {e}")
            return self._empty_result()
        except Exception as e:
            logger.error(f"EntityAnalyzer error: {e}")
            return self._empty_result()


def build_target_entities_from_scan(
    scan,
    db,
) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve target_entities + known_entities for a scan.

    Centralized so run_llm_tests and refresh_ai_snapshot share one source
    of truth on how each axis is populated :

        brands           = focus_brand + parent chain + children + aliases
        products         = product_lines JSONB of focus_brand + children
        ranges           = direct children of focus_brand (gammes)
        domains          = scan.domain + focus_brand.domain + scan_config.target_domains
        expert_sources   = ScanTrustSource.name where source_type='expert'
                           (when discover_trust_sources has run — empty otherwise)

        known_entities   = competitor ClientBrand.name + their product_lines
                           (caps the prompt by sending the LLM the salient
                           competitors instead of a wide-net list)

    The split between target_entities and known_entities mirrors the
    legacy target_brands / all_brands split inside BrandAnalyzer.
    """
    from models import ClientBrand, ScanBrandClassification
    try:
        from models import ScanTrustSource  # may not exist on older deploys
    except ImportError:
        ScanTrustSource = None  # type: ignore[assignment]

    target_brands: list[str] = []
    target_products: list[str] = []
    target_ranges: list[str] = []
    target_domains: list[str] = []
    expert_sources: list[str] = []
    known: list[str] = []

    scan_config = scan.config or {}
    # Always start with the scanned domain — even if focus_brand is unset
    # the scan covers a website, so its domain is the primary target axis.
    if scan.domain:
        target_domains.append(scan.domain)
    for d in scan_config.get("target_domains") or []:
        if d and d not in target_domains:
            target_domains.append(d)

    if scan.focus_brand_id:
        focus = db.query(ClientBrand).filter(ClientBrand.id == scan.focus_brand_id).first()
        if focus:
            target_brands.append(focus.name)
            target_brands.extend(focus.aliases or [])
            if focus.domain and focus.domain not in target_domains:
                target_domains.append(focus.domain)
            target_products.extend(focus.product_lines or [])

            # Phase BB : enrich products with the focus brand's BrandBrief
            # hero_products + signature_features. Capped at 10 each so brands
            # with sprawling catalogs (Avène has 30+ signature ingredients)
            # don't blow up the prompt token budget — same cap pattern as
            # known_entities below. Foot-gun #10 in project_phase_brand_briefs.
            brand_brief = focus.brief or {}
            if isinstance(brand_brief, dict):
                hero = [p for p in (brand_brief.get("hero_products") or [])
                        if isinstance(p, str) and p.strip()][:10]
                features = [f for f in (brand_brief.get("signature_features") or [])
                            if isinstance(f, str) and f.strip()][:10]
                target_products.extend(hero)
                target_products.extend(features)

            # Children = ranges/gammes (direct descendants in the brand tree)
            children = db.query(ClientBrand).filter(
                ClientBrand.parent_id == focus.id
            ).all()
            for c in children:
                target_ranges.append(c.name)
                target_ranges.extend(c.aliases or [])
                # Children's product_lines join into target_products — same
                # gamme can have its own line items (e.g. Effaclar -> Effaclar Duo).
                target_products.extend(c.product_lines or [])

    # Competitors → known_entities (not targets, just sector-known names
    # we want the LLM to recognize correctly).
    if scan.id is not None:
        competitors = (
            db.query(ClientBrand)
              .join(ScanBrandClassification,
                    ScanBrandClassification.brand_id == ClientBrand.id)
              .filter(ScanBrandClassification.scan_id == scan.id,
                      ScanBrandClassification.classification == "competitor")
              .all()
        )
        for c in competitors:
            if c.name:
                known.append(c.name)
            known.extend(c.aliases or [])
            known.extend(c.product_lines or [])

    # Expert sources — discover_trust_sources may not have run yet, in
    # which case ScanTrustSource is empty and the LLM falls back to its
    # priors (Wikipedia/Doctissimo recognition from training data).
    if ScanTrustSource is not None and scan.id is not None:
        try:
            sources = db.query(ScanTrustSource).filter(
                ScanTrustSource.scan_id == scan.id
            ).all()
            for s in sources:
                # Use name if present, else domain
                label = getattr(s, "name", None) or getattr(s, "domain", None)
                if label:
                    expert_sources.append(label)
        except Exception:
            logger.debug("ScanTrustSource lookup failed (table may not exist yet)")

    # Dedupe each list while preserving order.
    def _uniq(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in seq:
            if not x:
                continue
            k = _norm(x)
            if k and k not in seen:
                seen.add(k)
                out.append(x)
        return out

    target_entities = {
        "brands": _uniq(target_brands),
        "products": _uniq(target_products),
        "ranges": _uniq(target_ranges),
        "domains": _uniq(target_domains),
        "expert_sources": _uniq(expert_sources),
    }
    return target_entities, _uniq(known)[:30]
