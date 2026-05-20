"""brief_injector 2-level merge (Phase BB).

format_workspace_brief(client_apps, brand_brief=None) renders :
1. Workspace-only block when only client_apps['client_brief'] is set (legacy path)
2. Merged workspace + focus-brand block when brand_brief is also passed
3. "" when neither is present

Brand wins per-field on editorial_voice + target_audience. List fields
(competitors) concat-dedup. Tests pin both behaviours so a future refactor
that breaks the merge contract surfaces here.
"""

from __future__ import annotations

from adapters.brief_injector import (
    format_workspace_brief,
    format_workspace_brief_for_audience_only,
    format_analysis_context,
)


WS_BRIEF = {
    "industry": "Dermo-cosmetics — sensitive skin care",
    "company_overview": "Pierre Fabre is a French pharmaceutical group.",
    "brand_positioning": "Premium dermo-cosmetics, pharmacy-distributed",
    "editorial_voice": "expert, reassuring",
    "target_audience": "Adults 25-65 with sensitive skin",
    "products_services": ["Skincare", "Haircare"],
    "primary_brands": [{"name": "Avène"}, {"name": "Klorane"}],
    "key_competitors": ["L'Oréal", "Bioderma"],
}

BB_AVENE = {
    "name": "Avène",
    "parent_group": "Pierre Fabre",
    "description": "Premium dermo-cosmetic skincare for sensitive skin.",
    "positioning_statement": "Skincare for the most sensitive skin",
    "editorial_voice": "expert, reassuring, science-led — never salesy or alarmist",
    "tonality": ["expert", "warm"],
    "target_audience": "Women 25-55 with sensitive skin",
    "audience_segments": ["sensitive skin", "atopic", "post-procedure"],
    "differentiators": ["Thermal spring water", "Pharmacy distribution"],
    "product_lines": ["Cleanance", "Tolerance"],
    "hero_products": ["Cleanance Comedomed"],
    "signature_features": ["thermal spring water"],
    "taglines": ["Skincare for sensitive skin"],
    "direct_competitors": [{"name": "La Roche-Posay"}, {"name": "Eucerin"}],
    "indirect_competitors": ["Mustela"],
    "expertise_topics": ["sensitive skin routine", "rosacea"],
    "regulatory_constraints": ["EU Cosmetic Regulation 1223/2009"],
}


class TestEmpty:
    def test_no_inputs_returns_empty(self):
        assert format_workspace_brief(None) == ""
        assert format_workspace_brief({}, None) == ""
        assert format_workspace_brief({"client_brief": None}) == ""
        assert format_workspace_brief({}, {}) == ""

    def test_brand_only_no_workspace(self):
        # Brand brief alone still produces the workspace header (degraded mode)
        out = format_workspace_brief(None, BB_AVENE)
        assert "## Your company" in out
        assert "### Focus brand: Avène" in out


class TestWorkspaceOnly:
    def test_legacy_render_unchanged(self):
        out = format_workspace_brief({"client_brief": WS_BRIEF})
        assert "## Your company" in out
        assert "Industry: Dermo-cosmetics" in out
        assert "Editorial voice: expert, reassuring" in out
        assert "Target audience: Adults 25-65" in out
        assert "Bioderma" in out
        # No focus-brand block when brand_brief absent
        assert "### Focus brand" not in out

    def test_legacy_caller_one_arg_still_works(self):
        # Backward compat : callers that pass only client_apps must keep working
        out = format_workspace_brief({"client_brief": WS_BRIEF})
        assert "L'Oréal" in out


class TestMerged:
    def test_brand_wins_editorial_voice(self):
        out = format_workspace_brief({"client_brief": WS_BRIEF}, BB_AVENE)
        # Workspace editorial voice MUST NOT appear (brand override won)
        assert "Editorial voice: expert, reassuring\n" not in out
        # Brand editorial voice MUST appear with override marker
        assert "Editorial voice (override): expert, reassuring, science-led" in out

    def test_brand_wins_target_audience(self):
        out = format_workspace_brief({"client_brief": WS_BRIEF}, BB_AVENE)
        # Workspace audience suppressed when brand has it
        assert "Target audience: Adults 25-65" not in out
        # Brand audience emitted with override marker
        assert "Target audience (override): Women 25-55 with sensitive skin" in out

    def test_brand_inherits_when_field_missing(self):
        # Brand brief missing editorial_voice → workspace value should fill in
        bb_lite = dict(BB_AVENE)
        bb_lite["editorial_voice"] = ""
        out = format_workspace_brief({"client_brief": WS_BRIEF}, bb_lite)
        assert "Editorial voice: expert, reassuring" in out
        assert "Editorial voice (override)" not in out

    def test_competitors_concat_dedup(self):
        # Both workspace + brand list Bioderma-adjacent peers ; merged
        # output should contain workspace + brand minus duplicates
        out = format_workspace_brief({"client_brief": WS_BRIEF}, BB_AVENE)
        line = next(l for l in out.split("\n") if l.startswith("Known competitors"))
        # Must contain entries from BOTH sources
        assert "Bioderma" in line          # workspace
        assert "La Roche-Posay" in line    # brand direct
        assert "Mustela" in line           # brand indirect
        assert "L'Oréal" in line           # workspace
        assert "Eucerin" in line           # brand direct

    def test_competitors_dedup_case_insensitive(self):
        ws = dict(WS_BRIEF, key_competitors=["bioderma"])
        bb = dict(BB_AVENE, direct_competitors=[{"name": "Bioderma"}])
        out = format_workspace_brief({"client_brief": ws}, bb)
        line = next(l for l in out.split("\n") if l.startswith("Known competitors"))
        # Bioderma should appear exactly once (case-insensitive dedup)
        assert line.lower().count("bioderma") == 1

    def test_focus_brand_block_present(self):
        out = format_workspace_brief({"client_brief": WS_BRIEF}, BB_AVENE)
        assert "### Focus brand: Avène" in out
        assert "Parent group: Pierre Fabre" in out
        assert "Differentiators: Thermal spring water, Pharmacy distribution" in out
        assert "Hero products: Cleanance Comedomed" in out
        assert "Regulatory constraints: EU Cosmetic Regulation 1223/2009" in out

    def test_focus_brand_topics_capped(self):
        bb = dict(BB_AVENE, expertise_topics=[f"topic-{i}" for i in range(25)])
        out = format_workspace_brief({"client_brief": WS_BRIEF}, bb)
        topics_line = next(l for l in out.split("\n") if "Expertise topics" in l)
        # Capped at 10 — anti foot-gun against brands with sprawling topic lists
        assert "topic-0" in topics_line
        assert "topic-9" in topics_line
        assert "topic-10" not in topics_line
        assert "topic-24" not in topics_line


class TestBrandingProRendering:
    """BB.8 — 5 new branding-pro fields surface in the merged block."""

    def test_heritage_and_brand_story_render(self):
        bb = dict(BB_AVENE, heritage="Founded 1736 at Avène-les-Bains.",
                  brand_story="Three centuries of thermal-water dermatology.")
        out = format_workspace_brief({"client_brief": WS_BRIEF}, bb)
        assert "Heritage: Founded 1736" in out
        assert "Brand story: Three centuries" in out

    def test_tone_dos_donts_render(self):
        bb = dict(BB_AVENE,
                  tone_dos=["soulager", "apaiser", "protéger"],
                  tone_donts=["miracle", "instantané"])
        out = format_workspace_brief({"client_brief": WS_BRIEF}, bb)
        dos = next(l for l in out.split("\n") if l.startswith("Tone DOs"))
        donts = next(l for l in out.split("\n") if l.startswith("Tone DON'Ts"))
        assert "soulager" in dos and "apaiser" in dos
        assert "miracle" in donts and "instantané" in donts

    def test_claims_guidelines_separator(self):
        # claims_guidelines uses " · " separator (visual marker that each
        # entry is a rule, not a comma-listed entity)
        bb = dict(BB_AVENE,
                  claims_guidelines=["No 'cures' without AMM",
                                      "Cite study for health claim"])
        out = format_workspace_brief({"client_brief": WS_BRIEF}, bb)
        line = next(l for l in out.split("\n") if l.startswith("Claims guidelines"))
        assert " · " in line
        assert "AMM" in line

    def test_empty_branding_fields_skip_lines(self):
        # When the branding-pro fields are empty, no labelled line appears
        out = format_workspace_brief({"client_brief": WS_BRIEF}, BB_AVENE)
        assert "Heritage:" not in out
        assert "Brand story:" not in out
        assert "Tone DOs" not in out
        assert "Tone DON'Ts" not in out
        assert "Claims guidelines" not in out


class TestPartialBrand:
    def test_brand_with_only_name(self):
        # Brand brief with just the name (LLM hallucinated an empty shell)
        out = format_workspace_brief({"client_brief": WS_BRIEF}, {"name": "Avène"})
        assert "### Focus brand: Avène" in out
        # Workspace defaults still rendered
        assert "Editorial voice: expert, reassuring" in out

    def test_brand_without_name_still_renders_if_other_signal(self):
        bb = {"description": "Some brand description", "editorial_voice": "punchy"}
        out = format_workspace_brief({"client_brief": WS_BRIEF}, bb)
        assert "### Focus brand: (unnamed brand)" in out
        assert "Editorial voice (override): punchy" in out


class TestAudienceOnlyRender:
    """BB.9 — audience-only render strips voice fields from persona prompt.

    These tests pin the contract : voice / story / claims fields MUST be
    absent from the audience-only block ; audience fields MUST be present.
    Anyone who adds a new field to BrandBrief MUST classify it explicitly :
    either it joins this filter, or it gets the full render only.
    """

    def _full_brand(self):
        # Build a brand brief touching every voice + audience field, so we
        # can verify the partition is clean.
        return dict(BB_AVENE,
                    heritage="Founded 1736",
                    brand_story="Three centuries of dermatology",
                    tone_dos=["soulager", "apaiser"],
                    tone_donts=["miracle", "instantané"],
                    claims_guidelines=["No cure claims"],
                    taglines=["Skincare for sensitive skin"])

    def test_audience_subset_present(self):
        out = format_workspace_brief_for_audience_only(
            {"client_brief": WS_BRIEF}, self._full_brand()
        )
        assert "## Audience context" in out
        assert "Industry: Dermo-cosmetics" in out
        # Brand wins on target_audience
        assert "Women 25-55 with sensitive skin" in out
        # Audience segments + expertise topics
        assert "sensitive skin" in out and "atopic" in out
        assert "rosacea" in out

    def test_voice_fields_absent(self):
        out = format_workspace_brief_for_audience_only(
            {"client_brief": WS_BRIEF}, self._full_brand()
        )
        # NONE of the voice-coloured fields should leak
        assert "editorial_voice" not in out.lower()
        assert "Editorial voice" not in out
        assert "Tone DOs" not in out
        assert "Tone DON'Ts" not in out
        assert "soulager" not in out
        assert "miracle" not in out
        assert "Heritage" not in out
        assert "Brand story" not in out
        assert "Taglines" not in out
        assert "Claims guidelines" not in out
        assert "differentiators" not in out.lower()
        assert "hero" not in out.lower()
        assert "signature" not in out.lower()
        assert "competitors" not in out.lower()

    def test_workspace_audience_fallback_when_brand_empty(self):
        bb = {"name": "Avène"}  # No target_audience on brand
        out = format_workspace_brief_for_audience_only(
            {"client_brief": WS_BRIEF}, bb
        )
        # Falls back to workspace.target_audience
        assert "Adults 25-65 with sensitive skin" in out

    def test_returns_empty_when_no_signal(self):
        assert format_workspace_brief_for_audience_only(None, None) == ""
        assert format_workspace_brief_for_audience_only({}, {}) == ""

    def test_expertise_topics_capped_at_10(self):
        bb = dict(BB_AVENE,
                  expertise_topics=[f"topic-{i}" for i in range(25)])
        out = format_workspace_brief_for_audience_only(
            {"client_brief": WS_BRIEF}, bb
        )
        line = next(l for l in out.split("\n") if "expertise topics" in l.lower())
        assert "topic-0" in line
        assert "topic-9" in line
        assert "topic-10" not in line
        assert "topic-24" not in line


class TestAnalysisContextAudienceOnly:
    """BB.9 — format_analysis_context(audience_only=True) routes correctly."""

    def test_audience_only_flag_strips_voice(self):
        brand = dict(BB_AVENE,
                     tone_dos=["soulager"],
                     tone_donts=["miracle"],
                     heritage="Founded 1736")
        out = format_analysis_context(
            scan_config={"domain_brief": {"company": "Pierre Fabre",
                                           "industry": "Dermo-cosmetics"}},
            client_apps={"client_brief": WS_BRIEF},
            brand_brief=brand,
            audience_only=True,
        )
        # Domain Context still present (analysers need it)
        assert "Pierre Fabre" in out
        # Audience block present
        assert "## Audience context" in out
        # Voice fields stripped
        assert "soulager" not in out
        assert "miracle" not in out
        assert "1736" not in out
        # Not the full workspace block either
        assert "### Focus brand" not in out

    def test_default_keeps_full_render(self):
        # audience_only defaults to False — backward compatible for
        # article/faq callers that pass the brand brief but want everything.
        brand = dict(BB_AVENE, tone_dos=["soulager"])
        out = format_analysis_context(
            scan_config={"domain_brief": {"company": "PF"}},
            client_apps={"client_brief": WS_BRIEF},
            brand_brief=brand,
        )
        assert "soulager" in out
        assert "### Focus brand" in out
