"""
Shared utilities to format briefs as prompt context blocks.

Two distinct briefs, two distinct purposes :

1. **Domain brief** (per-scan, stored in `scan.config['domain_brief']`)
   Describes the SCANNED domain — which on a competitor scan is the COMPETITOR's
   site (e.g. PF user scans laroche-posay.fr → domain_brief = LRP info).
   Inject this into ANALYSIS prompts (classify_topics, generate_personas,
   brand_classifier, etc.) so the LLM understands what it's looking at.

2. **Workspace brief** (per-client, stored in `client.apps['client_brief']`)
   Describes the USER's company — their industry, brand voice, positioning,
   audience, products. Inject this into CONTENT GENERATION prompts (FAQ,
   article, newsletter) so the output sounds like the user's brand even when
   the source is a competitor scan.

Both return "" gracefully if no brief exists (backward compatible). Together
they provide the vertical-agnostic specialization the SaaS needs without any
hardcoded brand maps or vertical-specific prompts.
"""


def format_brief_context(scan_config: dict | None) -> str:
    """Extract domain brief from scan config and format as prompt context block."""
    if not scan_config:
        return ""
    brief = scan_config.get("domain_brief")
    if not brief:
        return ""

    lines = ["## Domain Context"]
    if brief.get("company"):
        lines.append(f"Company: {brief['company']}")
    if brief.get("description"):
        lines.append(f"Description: {brief['description']}")
    if brief.get("industry"):
        lines.append(f"Industry: {brief['industry']}")
    if brief.get("country"):
        lines.append(f"Country: {brief['country']}")
    if brief.get("brands"):
        lines.append(f"Own brands: {', '.join(brief['brands'])}")
    if brief.get("product_lines"):
        lines.append(f"Product lines: {', '.join(brief['product_lines'])}")
    if brief.get("services"):
        lines.append(f"Services: {', '.join(brief['services'])}")
    if brief.get("competitors"):
        comp_strs = []
        for c in brief["competitors"]:
            prods = c.get("products", [])
            comp_strs.append(f"{c['name']} ({', '.join(prods)})" if prods else c["name"])
        lines.append(f"Competitors: {'; '.join(comp_strs)}")
    if brief.get("topics"):
        lines.append(f"Key topics: {', '.join(brief['topics'])}")
    if brief.get("target_audience"):
        lines.append(f"Target audience: {brief['target_audience']}")

    return "\n".join(lines)


def _dedup_strings(*sources) -> list[str]:
    """Concat-dedup a sequence of string lists, preserving first-seen order.

    Used by the 2-level brief merge for list fields like competitors and topics
    where brand-specific entries should EXTEND workspace-level entries, not
    replace them.
    """
    seen: set[str] = set()
    out: list[str] = []
    for src in sources:
        for v in (src or []):
            if not isinstance(v, str):
                continue
            v = v.strip()
            if not v:
                continue
            k = v.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(v)
    return out


def format_workspace_brief(client_apps: dict | None,
                           brand_brief: dict | None = None) -> str:
    """Extract workspace brief from client.apps and format as 'Your company' block.

    Distinct from format_brief_context (which describes the scanned domain).
    This describes the USER's company — for content generation handlers that
    need to bias output toward the user's brand voice / industry / audience.

    Phase BB : ``brand_brief`` is an optional per-primary-brand JSONB blob
    (``client_brands.brief``) that surcharges the workspace brief per-field.
    Brand wins on overlapping scalars (editorial_voice, target_audience,
    positioning) ; list fields (competitors, topics) concat-dedup. When
    ``brand_brief`` is None or empty, this falls back to the legacy
    workspace-only rendering — backward compatible with every existing caller.

    Pass `client.apps` directly (the JSONB column on Client). Returns "" if
    no client_brief AND no brand_brief exist (workspace not bootstrapped).
    """
    workspace = (client_apps or {}).get("client_brief") or {}
    brand = brand_brief or {}

    if not workspace and not brand:
        return ""

    # ── Workspace company block (always rendered when present) ──────────
    lines = ["## Your company (the brand voice for this content)"]
    if workspace.get("industry"):
        lines.append(f"Industry: {workspace['industry']}")
    if workspace.get("company_overview"):
        lines.append(f"Overview: {workspace['company_overview']}")
    if workspace.get("brand_positioning"):
        lines.append(f"Positioning (company-wide): {workspace['brand_positioning']}")
    # editorial_voice + target_audience : brand wins per-field. Skip workspace
    # if brand has them (rendered in the brand block below), otherwise emit
    # workspace value as fallback.
    if workspace.get("editorial_voice") and not brand.get("editorial_voice"):
        lines.append(f"Editorial voice: {workspace['editorial_voice']}")
    if workspace.get("target_audience") and not brand.get("target_audience"):
        lines.append(f"Target audience: {workspace['target_audience']}")
    if workspace.get("products_services"):
        lines.append(f"Products / services: {', '.join(workspace['products_services'])}")
    if workspace.get("primary_brands"):
        names = [b.get("name", "") for b in workspace["primary_brands"] if b.get("name")]
        if names:
            lines.append(f"Primary brands (priority order): {', '.join(names)}")

    # Competitors : concat-dedup workspace.key_competitors + brand.direct/indirect
    direct_comp_names = [
        (c.get("name") or "").strip()
        for c in (brand.get("direct_competitors") or [])
        if isinstance(c, dict) and (c.get("name") or "").strip()
    ]
    indirect_comp_names = [
        n for n in (brand.get("indirect_competitors") or [])
        if isinstance(n, str)
    ]
    merged_competitors = _dedup_strings(
        workspace.get("key_competitors") or [],
        direct_comp_names,
        indirect_comp_names,
    )
    if merged_competitors:
        lines.append(
            f"Known competitors (do NOT promote these): {', '.join(merged_competitors)}"
        )

    # ── Focus brand block (only when brand_brief present) ───────────────
    if brand and (brand.get("name") or brand.get("description")
                  or brand.get("editorial_voice") or brand.get("target_audience")):
        brand_name = (brand.get("name") or "").strip() or "(unnamed brand)"
        lines.append("")
        lines.append(f"### Focus brand: {brand_name}")
        if brand.get("parent_group"):
            lines.append(f"Parent group: {brand['parent_group']}")
        if brand.get("description"):
            lines.append(f"Description: {brand['description']}")
        # BB.8 narrative blocks — surfaced right after description so article
        # intros/conclusions can pull from them without scrolling.
        if brand.get("heritage"):
            lines.append(f"Heritage: {brand['heritage']}")
        if brand.get("brand_story"):
            lines.append(f"Brand story: {brand['brand_story']}")
        if brand.get("positioning_statement"):
            lines.append(f"Positioning: {brand['positioning_statement']}")
        if brand.get("editorial_voice"):
            lines.append(f"Editorial voice (override): {brand['editorial_voice']}")
        if brand.get("tonality"):
            lines.append(f"Tonality: {', '.join(brand['tonality'])}")
        # BB.8 tone DO / DON'T : the highest-leverage signal for a copywriter.
        # Surface verbatim so the downstream LLM can lift the exact vocabulary.
        if brand.get("tone_dos"):
            lines.append(f"Tone DOs (use these): {', '.join(brand['tone_dos'])}")
        if brand.get("tone_donts"):
            lines.append(f"Tone DON'Ts (avoid these): {', '.join(brand['tone_donts'])}")
        if brand.get("target_audience"):
            lines.append(f"Target audience (override): {brand['target_audience']}")
        if brand.get("audience_segments"):
            lines.append(f"Audience segments: {', '.join(brand['audience_segments'])}")
        if brand.get("differentiators"):
            lines.append(f"Differentiators: {', '.join(brand['differentiators'])}")
        if brand.get("product_lines"):
            lines.append(f"Product lines: {', '.join(brand['product_lines'])}")
        if brand.get("hero_products"):
            lines.append(f"Hero products: {', '.join(brand['hero_products'])}")
        if brand.get("signature_features"):
            lines.append(f"Signature features: {', '.join(brand['signature_features'])}")
        if brand.get("taglines"):
            lines.append(f"Taglines: {', '.join(brand['taglines'])}")
        if brand.get("expertise_topics"):
            # Cap at 10 — these go directly into bias prompts and we don't want
            # to balloon token count for brands that own 30+ topics.
            topics = list(brand["expertise_topics"])[:10]
            lines.append(f"Expertise topics (bias toward these): {', '.join(topics)}")
        # BB.8 claims guidelines : copywriter-facing rules combining brand
        # voice + regulatory framing. Distinct from regulatory_constraints
        # which is the legal list.
        if brand.get("claims_guidelines"):
            lines.append(
                f"Claims guidelines: {' · '.join(brand['claims_guidelines'])}"
            )
        if brand.get("regulatory_constraints"):
            lines.append(
                f"Regulatory constraints: {', '.join(brand['regulatory_constraints'])}"
            )

    return "\n".join(lines)


def format_vertical_examples(scan_config: dict | None) -> str:
    """Build a vertical-aware examples block for brand analysis prompts.

    Replaces the hardcoded "Cicalfate vs acide hyaluronique" dermo-cosmétique
    examples that used to live inside BRAND_ANALYSIS_PROMPT and
    BRAND_CLEANUP_PROMPT. Now the LLM gets industry-specific examples
    derived from the scan's own brief — multi-vertical by construction:
      - Cosmetics scan → ["Avène", "Cicalfate"] valid, ["crème", "rétinol"] noise
      - Automotive scan → ["Castrol", "GTX"] valid, ["huile moteur", "freins"] noise
      - SaaS scan → ["Stripe", "Salesforce"] valid, ["api", "subscription"] noise

    Returns "" when the brief lacks the data (legacy scans pre Option B);
    callers should be tolerant of an empty string in that case.
    """
    if not scan_config:
        return ""
    brief = scan_config.get("domain_brief") or {}
    if not brief:
        return ""

    valid_examples: list[str] = []
    for b in (brief.get("brands") or []):
        if isinstance(b, str) and b.strip():
            valid_examples.append(b.strip())
    for c in (brief.get("competitors") or []):
        if isinstance(c, dict):
            name = (c.get("name") or "").strip()
            if name:
                valid_examples.append(name)
            for p in (c.get("products") or [])[:3]:  # cap to keep prompt short
                if isinstance(p, str) and p.strip():
                    valid_examples.append(p.strip())
    # Dedup case-insensitive while keeping order
    seen: set[str] = set()
    valid: list[str] = []
    for v in valid_examples:
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        valid.append(v)
    valid = valid[:15]

    noise = [n for n in (brief.get("noise_patterns") or []) if isinstance(n, str) and n.strip()][:15]

    if not valid and not noise:
        return ""

    lines = ["## Vertical examples (use these to calibrate what counts as a brand)"]
    if brief.get("industry"):
        lines.append(f"Industry: {brief['industry']}")
    if valid:
        lines.append(f"✓ Real brand patterns: {', '.join(valid)}")
    if noise:
        lines.append(f"✗ Noise to filter (NOT brands): {', '.join(noise)}")
    return "\n".join(lines)


def format_workspace_brief_for_audience_only(client_apps: dict | None,
                                              brand_brief: dict | None = None) -> str:
    """Render ONLY the audience subset of the merged workspace+brand brief.

    BB.9 separation-of-concerns fix : generate_personas needs to know WHO
    the brand talks to (audience demographics, segments, topics) but NOT
    HOW the brand speaks (editorial_voice, tone_dos/donts, brand_story,
    heritage, taglines, claims_guidelines). Mixing both pollutes the
    persona prompt — the LLM ends up mirroring brand voice traits ("expert
    reassuring") into persona personality descriptions, producing personas
    that read like brand brochures instead of real audience profiles.

    Renders a strictly audience-coloured block :
      - industry             (workspace, for category context)
      - target_audience      (brand wins per-field / workspace fallback)
      - audience_segments    (brand)
      - expertise_topics     (brand, cap 10 — these define what topics
                              the brand wants to own, which biases persona
                              questions toward those topics)

    Voice-only fields are deliberately dropped. Callers that need them
    (article + faq generation) use the full ``format_workspace_brief``.

    Returns "" if no audience signal is available.
    """
    workspace = (client_apps or {}).get("client_brief") or {}
    brand = brand_brief or {}

    industry = (workspace.get("industry") or "").strip()
    # Brand wins per-field on target_audience (mirrors full-render logic).
    brand_audience = (brand.get("target_audience") or "").strip()
    workspace_audience = (workspace.get("target_audience") or "").strip()
    target_audience = brand_audience or workspace_audience

    audience_segments = [
        s for s in (brand.get("audience_segments") or [])
        if isinstance(s, str) and s.strip()
    ]
    expertise_topics = [
        t for t in (brand.get("expertise_topics") or [])
        if isinstance(t, str) and t.strip()
    ][:10]  # cap matches full-render cap

    if not industry and not target_audience and not audience_segments and not expertise_topics:
        return ""

    lines = ["## Audience context (who reads the content)"]
    if industry:
        lines.append(f"Industry: {industry}")
    if target_audience:
        lines.append(f"Target audience: {target_audience}")
    if audience_segments:
        lines.append(f"Audience segments: {', '.join(audience_segments)}")
    if expertise_topics:
        lines.append(
            f"Brand's expertise topics (questions tend to cluster here): "
            f"{', '.join(expertise_topics)}"
        )
    return "\n".join(lines)


def format_analysis_context(scan_config: dict | None, client_apps: dict | None,
                            brand_brief: dict | None = None,
                            audience_only: bool = False) -> str:
    """Combine domain brief + workspace brief + vertical examples for analysis prompts.

    Analysers (classify_topics, generate_personas, brand_analyzer, brand_cleanup,
    generate_editorial, …) benefit from all three layers :
      1. Domain brief — what the scanned site is.
      2. Workspace brief — whose perspective to adopt.
      3. Vertical examples — concrete valid/noise patterns for this industry,
         so the LLM stops over-extracting ingredients/generics as brands.

    Phase BB : optional ``brand_brief`` threads through to format_workspace_brief
    to surcharge the workspace block with focus-brand specifics (voice, audience,
    competitors, expertise topics).

    BB.9 : ``audience_only=True`` routes through
    ``format_workspace_brief_for_audience_only`` instead of the full render.
    Used by generate_personas to avoid leaking brand voice into persona
    descriptions. Article / FAQ generators keep the full render.

    Returns "" if all three are empty. Blocks are separated by blank lines.
    """
    parts: list[str] = []
    db_block = format_brief_context(scan_config)
    if db_block:
        parts.append(db_block)
    if audience_only:
        wb_block = format_workspace_brief_for_audience_only(client_apps, brand_brief)
    else:
        wb_block = format_workspace_brief(client_apps, brand_brief)
    if wb_block:
        parts.append(wb_block)
    ve_block = format_vertical_examples(scan_config)
    if ve_block:
        parts.append(ve_block)
    return "\n\n".join(parts)


def format_promoted_brands_block(promoted_brand_names: list[str]) -> str:
    """Format the brands-to-promote list as a high-priority injection block.

    Used when BrandResolver.resolve_promotion() returns the brands the system
    is instructed to promote in this specific content item. This is the
    runtime-resolved brand bias, distinct from the workspace defaults.
    """
    if not promoted_brand_names:
        return ""
    if len(promoted_brand_names) == 1:
        names = promoted_brand_names[0]
    else:
        lead = promoted_brand_names[0]
        rest = ", ".join(promoted_brand_names[1:])
        names = f"{lead} (lead) — supporting: {rest}"
    return (
        "## Brands to promote in this content (priority order)\n"
        f"{names}\n"
        "When the answer naturally fits, feature these brands and their products. "
        "DO NOT promote competitors. If the prompt later mentions 'produits {brand_name}', "
        "it refers to these brands, not the scanned domain."
    )
