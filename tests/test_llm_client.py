import llm_client


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------


def test_find_json_array_extracts_from_fenced_and_trailing_text():
    text = 'Intro\n```json\n[{"a": 1}, {"a": 2}]\n```\n| table | header |'
    assert llm_client.find_json_array(text) == [{"a": 1}, {"a": 2}]


def test_find_json_array_returns_none_when_absent():
    assert llm_client.find_json_array("no json here") is None


def test_find_json_object_extracts_single_object():
    text = 'prefix {"decision_maker_name": "Bob"} suffix'
    assert llm_client.find_json_object(text) == {"decision_maker_name": "Bob"}


def test_find_json_object_returns_none_when_absent():
    assert llm_client.find_json_object("no json here") is None


def test_parse_leads_from_text_array():
    text = '[{"business_name": "A"}]'
    leads, raw = llm_client.parse_leads_from_text(text)
    assert leads == [{"business_name": "A"}]
    assert raw == text


def test_parse_leads_from_text_falls_back_to_leads_key():
    leads, _ = llm_client.parse_leads_from_text('{"leads": [{"business_name": "B"}]}')
    assert leads == [{"business_name": "B"}]


def test_parse_leads_from_text_falls_back_to_bare_object():
    leads, _ = llm_client.parse_leads_from_text('{"business_name": "C"}')
    assert leads == [{"business_name": "C"}]


def test_parse_leads_from_text_returns_empty_on_garbage():
    leads, _ = llm_client.parse_leads_from_text("not json at all")
    assert leads == []


# --------------------------------------------------------------------------
# flatten_lead / infer_industry
# --------------------------------------------------------------------------


def test_flatten_lead_maps_nested_schema_to_flat_columns():
    raw = {
        "lead_id": "IE-LEAD-001",
        "business_name": "Acme Ltd",
        "decision_maker": {"name": "Jane Doe", "title": "MD", "linkedin_url": "https://linkedin.com/in/jane"},
        "contact_details": {
            "email": "jane@acme.ie",
            "website": "https://acme.ie",
            "phone": "091 123456",
            "eircode": "H91 XXXX",
            "county": "Galway",
        },
        "audit_results": {
            "overall_score": 95,
            "status": "Verified (Active Queue)",
            "directory_source": "Chartered Accountants Ireland",
            "outreach_type": "Email & Phone",
            "disqualification_reason": None,
        },
        "peninsula_pitch_hook": "hook text",
    }
    flat = llm_client.flatten_lead(raw, "ALL")
    assert flat["business_name"] == "Acme Ltd"
    assert flat["decision_maker_name"] == "Jane Doe"
    assert flat["county"] == "Galway"
    assert flat["contact_email"] == "jane@acme.ie"
    assert flat["overall_score"] == 95
    assert flat["industry"] == "Accountants"  # inferred from directory_source


def test_flatten_lead_handles_missing_nested_objects():
    flat = llm_client.flatten_lead({"business_name": "Bare Ltd"}, "Legal")
    assert flat["business_name"] == "Bare Ltd"
    assert flat["decision_maker_name"] is None
    assert flat["industry"] == "Legal"  # requested_industry used since no directory_source


def test_infer_industry_uses_requested_when_not_all():
    assert llm_client.infer_industry("Chartered Accountants Ireland", "Legal") == "Legal"


def test_infer_industry_infers_from_directory_source_when_all():
    assert llm_client.infer_industry("Chartered Accountants Ireland", "ALL") == "Accountants"


def test_infer_industry_unknown_source_returns_none():
    assert llm_client.infer_industry("Some Random Directory", "ALL") is None


def test_infer_industry_no_source_and_all_returns_none():
    assert llm_client.infer_industry(None, "ALL") is None


# --------------------------------------------------------------------------
# Tier 1 pre-filter
# --------------------------------------------------------------------------


def test_pre_filter_lead_scores_complete_lead_highest():
    lead = {
        "decision_maker_name": "Jane Doe",
        "contact_email": "jane@acme.ie",
        "contact_phone": "091 123456",
        "website": "https://acme.ie",
        "industry": "Legal",
    }
    result = llm_client.pre_filter_lead(lead)
    assert result is not None
    assert result["overall_score"] == 95  # 92 + website bonus
    assert result["audit_status"] == "Verified (Active Queue)"
    assert result["_pre_filtered"] is True


def test_pre_filter_lead_partial_contact_scores_lower():
    lead = {"decision_maker_name": "Jane Doe", "contact_email": "jane@acme.ie"}
    result = llm_client.pre_filter_lead(lead)
    assert result["overall_score"] == 80


def test_pre_filter_lead_returns_none_when_name_missing():
    assert llm_client.pre_filter_lead({"contact_email": "x@y.ie", "contact_phone": "091 1"}) is None


def test_pre_filter_lead_returns_none_when_no_contact_method():
    assert llm_client.pre_filter_lead({"decision_maker_name": "Jane Doe"}) is None


def test_pre_filter_lead_returns_none_for_empty_lead():
    assert llm_client.pre_filter_lead({}) is None


def test_pre_filter_lead_treats_placeholder_strings_as_empty():
    lead = {"decision_maker_name": "Jane Doe", "contact_email": "nan", "contact_phone": "N/A"}
    assert llm_client.pre_filter_lead(lead) is None


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------


def test_build_enrichment_user_prompt_includes_scraped_block_when_present():
    prompt = llm_client.build_enrichment_user_prompt({"business_name": "Acme"}, "scraped ground truth")
    assert "scraped ground truth" in prompt
    assert "Acme" in prompt


def test_build_enrichment_user_prompt_omits_scraped_block_when_absent():
    prompt = llm_client.build_enrichment_user_prompt({"business_name": "Acme"}, "")
    assert "Ground Truth" not in prompt


def test_get_sector_prompt_known_industry_uses_precompiled_prompt():
    prompt = llm_client.get_sector_prompt("Legal")
    assert prompt == llm_client.SECTOR_PROMPTS["Legal"]
    assert prompt != llm_client.ENRICHMENT_SYSTEM_PROMPT


def test_get_sector_prompt_unknown_industry_falls_back_to_base_prompt():
    assert llm_client.get_sector_prompt("Unclassified") == llm_client.ENRICHMENT_SYSTEM_PROMPT
    assert llm_client.get_sector_prompt(None) == llm_client.ENRICHMENT_SYSTEM_PROMPT


# --------------------------------------------------------------------------
# Usage / cost tracking (no network — synthetic response objects)
# --------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, input_tokens, output_tokens):
        self.usage = _FakeUsage(input_tokens, output_tokens)


def test_usage_tracking_accumulates_across_calls_and_computes_cost():
    llm_client.reset_usage_totals()
    llm_client._record_usage(_FakeResponse(1000, 500))
    llm_client._record_usage(_FakeResponse(2000, 1000))

    totals = llm_client.get_usage_totals()
    assert totals["input_tokens"] == 3000
    assert totals["output_tokens"] == 1500

    expected_cost = round(
        3000 / 1_000_000 * llm_client.ANTHROPIC_INPUT_PRICE_PER_MTOK
        + 1500 / 1_000_000 * llm_client.ANTHROPIC_OUTPUT_PRICE_PER_MTOK,
        4,
    )
    assert totals["estimated_cost_usd"] == expected_cost


def test_reset_usage_totals_zeroes_counters():
    llm_client.reset_usage_totals()
    llm_client._record_usage(_FakeResponse(100, 100))
    llm_client.reset_usage_totals()

    totals = llm_client.get_usage_totals()
    assert totals["input_tokens"] == 0
    assert totals["output_tokens"] == 0
    assert totals["estimated_cost_usd"] == 0


def test_record_usage_ignores_response_with_no_usage_attribute():
    llm_client.reset_usage_totals()

    class NoUsage:
        pass

    llm_client._record_usage(NoUsage())  # must not raise
    assert llm_client.get_usage_totals()["input_tokens"] == 0
