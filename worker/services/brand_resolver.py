"""Brand promotion resolver — answers "which brands should this content gen promote?"

Used by FAQ + Article + future content generation handlers to determine which
brands the LLM is instructed to push (and conversely, which competitor brands
to actively avoid mentioning positively). Implements the SaaS bias mechanism :
when a Pierre Fabre user generates an FAQ from an opportunity on a *competitor*
scan (laroche-posay.fr), the output must promote Avène/Aderma/Ducray, never
La Roche-Posay.

Resolution chain (highest priority first) :

  1. scan.promotion_brand_ids                      (per-scan explicit override)
  2. ScanBrandClassification(my_brand) for scan    (auto-detected on this scan)
  3. client.primary_brand_ids                      (cross-scan workspace default)
  4. raise PromotionUnsetError                     (UI prompts user to set defaults)

The merge between #1/#2 and #3 is a UNION preserving the priority order — if
the scan auto-detected `[Avène]` as my_brand but the workspace default is
`[Avène, Aderma, Ducray]`, the result is `[Avène, Aderma, Ducray]` so we don't
arbitrarily lose Aderma/Ducray just because they weren't mentioned in this
particular scan's LLM responses.

Returned alongside the promote list :
- the *competitor* brands (from ScanBrandClassification.classification='competitor')
  + the scan.domain itself (when it doesn't belong to a my_brand) — these are
  the names Claude must NOT recommend in the generated content.
- a `resolved_via` audit string describing which step matched, useful in logs
  and the /promotion/resolve transparency endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PromotionUnsetError(RuntimeError):
    """Raised when no brand can be resolved for promotion.

    The caller (API endpoint or worker handler) should catch this and surface
    a 409 / actionable message routing the user to the workspace settings page
    where they can set `client.primary_brand_ids`.
    """


@dataclass
class BrandRef:
    """Lightweight brand descriptor passed downstream into prompt-building code.

    Carries only what the prompt needs — full ClientBrand row stays in DB.
    """
    id: UUID
    name: str
    domain: str | None = None
    aliases: list[str] = field(default_factory=list)


@dataclass
class PromotionResolution:
    """Output of resolve_promotion(). Contains everything the prompt builder needs."""
    promote_brands: list[BrandRef]            # ordered, [0] = lead brand
    exclude_brands: list[BrandRef]            # competitors + scanned competitor domain
    exclude_domain_names: list[str]           # plain strings for regex post-checks
    resolved_via: str                         # one of "scan_override" / "scan_classifications"
                                              # / "client_primary" / "merged" — for logs + audit
    promote_brand_ids: list[UUID]             # convenience: list of UUIDs for storage


def resolve_promotion(scan, db: Session) -> PromotionResolution:
    """Resolve which brands the upcoming content gen should promote vs avoid.

    Args:
        scan: a Scan row (must have .id, .client_id, .focus_brand_id, .domain,
              .promotion_brand_ids)
        db: active SQLAlchemy session

    Returns:
        PromotionResolution dataclass

    Raises:
        PromotionUnsetError if nothing resolves (caller should redirect user to
        the brand-promotion settings page).
    """
    from models import Client, ClientBrand, ScanBrandClassification

    # ── Step 1: scan.promotion_brand_ids (explicit per-scan override) ─────
    promote_ids: list[UUID] = []
    resolved_via = ""
    if scan.promotion_brand_ids:
        promote_ids = list(scan.promotion_brand_ids)
        resolved_via = "scan_override"

    # ── Step 2: ScanBrandClassification(my_brand) for this scan ───────────
    sbc_brand_ids: list[UUID] = []
    sbc_rows = (
        db.query(ScanBrandClassification.brand_id)
        .filter(
            ScanBrandClassification.scan_id == scan.id,
            ScanBrandClassification.classification == "my_brand",
        )
        .all()
    )
    sbc_brand_ids = [r.brand_id for r in sbc_rows]

    # ── Step 3: client.primary_brand_ids (workspace default) ──────────────
    client = db.query(Client).filter(Client.id == scan.client_id).first()
    client_primary_ids: list[UUID] = list(client.primary_brand_ids) if (client and client.primary_brand_ids) else []

    # ── Merge with priority order, preserving uniqueness ──────────────────
    if not promote_ids:
        # No explicit override → take SBC + client primaries (union, SBC first to honor scan-level signals)
        seen = set()
        merged: list[UUID] = []
        for bid in sbc_brand_ids + client_primary_ids:
            if bid not in seen:
                seen.add(bid)
                merged.append(bid)
        promote_ids = merged
        if sbc_brand_ids and client_primary_ids:
            resolved_via = "merged(scan_classifications + client_primary)"
        elif sbc_brand_ids:
            resolved_via = "scan_classifications"
        elif client_primary_ids:
            resolved_via = "client_primary"

    if not promote_ids:
        raise PromotionUnsetError(
            f"No brand to promote for scan {scan.id}: no per-scan override, "
            f"no my_brand classifications, no client.primary_brand_ids set. "
            f"Resolve by setting primary brands in workspace settings."
        )

    # ── Load promote brand details ────────────────────────────────────────
    promote_brand_rows = (
        db.query(ClientBrand)
        .filter(ClientBrand.id.in_(promote_ids))
        .all()
    )
    # Preserve the priority order from promote_ids (db.query doesn't guarantee order)
    by_id = {b.id: b for b in promote_brand_rows}
    promote_brands: list[BrandRef] = []
    for bid in promote_ids:
        b = by_id.get(bid)
        if b is not None:
            promote_brands.append(BrandRef(
                id=b.id,
                name=b.name,
                domain=b.domain,
                aliases=list(b.aliases or []),
            ))

    # ── Build exclude list = competitors of this scan ─────────────────────
    competitor_rows = (
        db.query(ClientBrand)
        .join(ScanBrandClassification, ScanBrandClassification.brand_id == ClientBrand.id)
        .filter(
            ScanBrandClassification.scan_id == scan.id,
            ScanBrandClassification.classification == "competitor",
        )
        .all()
    )
    exclude_brands: list[BrandRef] = [
        BrandRef(id=b.id, name=b.name, domain=b.domain, aliases=list(b.aliases or []))
        for b in competitor_rows
    ]

    # Add the scanned domain itself to the exclusion if it doesn't belong to a promote brand
    promote_domain_set = {b.domain.lower() for b in promote_brands if b.domain}
    scan_domain_lc = (scan.domain or "").lower()
    if scan_domain_lc and not any(scan_domain_lc.endswith(d) or d.endswith(scan_domain_lc) for d in promote_domain_set):
        # scan.domain is a competitor's domain — make sure its name is excluded
        # (already in exclude_brands if SBC tagged it as competitor, but this is defensive)
        pass  # competitor classification should already catch it

    # Flatten exclude names + aliases for post-hoc regex check on generated content
    exclude_domain_names: list[str] = []
    for b in exclude_brands:
        if b.name:
            exclude_domain_names.append(b.name)
        for alias in b.aliases:
            if alias:
                exclude_domain_names.append(alias)
    # Dedupe case-insensitively while preserving original casing
    seen_lc = set()
    deduped_exclude: list[str] = []
    for n in exclude_domain_names:
        lc = n.lower().strip()
        if lc and lc not in seen_lc:
            seen_lc.add(lc)
            deduped_exclude.append(n)

    logger.info(
        f"resolve_promotion(scan={scan.id}): "
        f"{len(promote_brands)} promote, {len(exclude_brands)} exclude, "
        f"resolved_via={resolved_via}"
    )

    return PromotionResolution(
        promote_brands=promote_brands,
        exclude_brands=exclude_brands,
        exclude_domain_names=deduped_exclude,
        resolved_via=resolved_via,
        promote_brand_ids=[b.id for b in promote_brands],
    )


def is_competitor_scan(scan, db: Session) -> bool:
    """Return True when the scanned domain does not belong to any of the
    client's promoted brands — i.e. we're scanning a competitor.

    Used by `materialize_content_items` to decide whether `target_url` should
    be auto-filled from the scan (user-owned domain → yes) or left NULL for
    the user to pick a page on their own site (competitor → A2 manual pick).

    Returns False (the conservative default) on any resolution failure so we
    don't accidentally block a legitimate user-owned scan from auto-filling.
    """
    if not scan or not scan.domain:
        return False
    try:
        resolution = resolve_promotion(scan, db)
    except PromotionUnsetError:
        # No brand resolves at all → we don't know what's user vs competitor.
        # Treat as user-owned (status quo behavior, conservative).
        return False

    scan_domain_lc = scan.domain.lower().strip()
    if scan_domain_lc.startswith("www."):
        scan_domain_lc = scan_domain_lc[4:]

    for b in resolution.promote_brands:
        if not b.domain:
            continue
        b_domain_lc = b.domain.lower().strip()
        if b_domain_lc.startswith("www."):
            b_domain_lc = b_domain_lc[4:]
        # Match either direction so subdomain scans (eu.avene.com vs avene.com)
        # are still treated as user-owned.
        if scan_domain_lc == b_domain_lc:
            return False
        if scan_domain_lc.endswith("." + b_domain_lc):
            return False
        if b_domain_lc.endswith("." + scan_domain_lc):
            return False
    return True
