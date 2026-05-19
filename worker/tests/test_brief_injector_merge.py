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

from adapters.brief_injector import format_workspace_brief


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
