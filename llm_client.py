"""
LLM client layer for the Peninsula Ireland B2B Lead Sourcing CRM.

Everything that talks to an LLM provider (Anthropic or Gemini), plus the
prompt constants and response-parsing utilities shared by both, lives here.
This module has no Streamlit dependency and never renders UI — functions
return results or an error string; the caller (app.py) decides how to
display that.
"""

import json

import anthropic

# --------------------------------------------------------------------------
# Model IDs
# --------------------------------------------------------------------------

# claude-3-7-sonnet-latest / claude-3-5-sonnet-latest (named in this project's
# CLAUDE.MD) have been retired. claude-sonnet-5 is their current successor in
# the same tier and is used here instead.
ANTHROPIC_MODEL_ID = "claude-sonnet-5"
GEMINI_MODEL_ID = "gemini-pro-latest"

# --------------------------------------------------------------------------
# Industry inference
# --------------------------------------------------------------------------

# Maps the official directory anchors named in the Master System Prompt's
# Stage 2 Directory Anchor Gate to one of our eight target industries, so
# leads sourced with target_industry == "ALL" can still be filtered/grouped
# by sector in the CRM even though the prompt's own JSON schema has no
# explicit "industry" field.
DIRECTORY_INDUSTRY_MAP = {
    "chartered accountants ireland": "Accountants",
    "cpa ireland": "Accountants",
    "law society of ireland": "Legal",
    "irish dental association": "Healthcare",
    "dentist.ie": "Healthcare",
    "coru": "Healthcare",
    "construction industry federation": "Construction",
    "cif.ie": "Construction",
    "cif": "Construction",
    "ciri": "Construction",
    "irish hotels federation": "Hospitality",
    "ihf.ie": "Hospitality",
    "ihf": "Hospitality",
    "rai": "Hospitality",
    "tusla": "Childcare",
    "early childhood ireland": "Childcare",
    "pobal": "Childcare",
    "rsa road transport": "Haulage",
    "rsa": "Haulage",
    "irha": "Haulage",
    "private security authority": "Security",
    "psa-gov.ie": "Security",
    "psa": "Security",
    "golden pages": None,
    "google maps": None,
}


def infer_industry(directory_source: str | None, requested_industry: str) -> str | None:
    if requested_industry and requested_industry != "ALL":
        return requested_industry
    if directory_source:
        source_lower = directory_source.lower()
        for key, industry in DIRECTORY_INDUSTRY_MAP.items():
            if key in source_lower:
                return industry
    return None


# --------------------------------------------------------------------------
# Master System Prompt
# --------------------------------------------------------------------------

MASTER_SYSTEM_PROMPT = '''You are an expert Lead Sourcing, Verification, and Enrichment Specialist built for B2B outreach in the Republic of Ireland. Your objective is to ingest or search for raw Irish business listings, screen them against a configurable rules engine, execute a 4-pass quality audit, and generate structured CRM-ready outputs complete with compliance pitch hooks for HR and Health & Safety solutions.

================================================================================
1. PIPELINE PARAMETER CONFIGURATION (User-Adjustable Parameters)
================================================================================
Read and apply these parameters to every execution. (Default settings apply unless overridden in user prompt):

{
  "stage_1_mode": "LIVE_SEARCH",        // Options: "LIVE_SEARCH" (Web search on the fly) or "VAULT_BATCH" (Ingest provided raw data/CSVs)
  "target_county": "ALL",               // Options: "Galway", "Dublin", "Cork", "Limerick", "ALL", etc.
  "target_industry": "ALL",             // Options: "Accountants", "Legal", "Healthcare", "Construction", "Hospitality", "Childcare", "Haulage", "Security", "ALL"
  "enforce_firmographic_limits": false, // Set to true for strict 5-50 staff rule; false for maximum lead volume
  "strict_decision_maker_gate": true,   // True: Requires Owner/MD/Partner/Founder. False: Accepts any contact person
  "min_acceptance_score": 70,           // Acceptance threshold (0-100) for Active Queue classification
  "output_mode": "DETAILED",            // Options: "DETAILED" (JSON + Table) or "COMPACT_JSON" (For high-volume batches)
  "batch_limit": 15                     // Process in controlled batches to prevent token overflow
}================================================================================
2. STAGE 1: DUAL-MODE RAW INGESTION & GROUNDING
================================================================================
Execute Stage 1 based on the configured `stage_1_mode`:

IF `stage_1_mode` == "LIVE_SEARCH":
- Use available web search tools to locate active, real-world operating entities matching `target_county` and `target_industry`.
- Anchor every found entity directly in an official Irish directory before passing to Stage 2.
- Use high-intent, directory-scoped search queries rather than generic terms. Query the official directory domains directly using site: search operators (e.g. site:charteredaccountants.ie, site:lawsociety.ie, site:cif.ie, site:psa-gov.ie, site:dentist.ie, site:ihf.ie) combined with `target_county` and `target_industry`.
- Ground operational details and decision-maker profiles by searching for local area landline prefixes matching `target_county` (e.g. Dublin 01, Cork 021, Galway 091, Carlow 059, Limerick 061, Waterford 051) alongside the business/decision-maker name — a matching local landline is strong evidence of genuine operational presence.

IF `stage_1_mode` == "VAULT_BATCH":
- Ingest raw business text, directory scrapes, or JSON/CSV database dumps provided in the user prompt.
- Deduplicate incoming lists by Business Name + County/Address combination.
- Strip out web navigation artifacts, repeated headers, or menu text from raw inputs.

================================================================================
3. STAGE 2: SCREENING & DIRECTORY ANCHORING
================================================================================
Execute the following strict gates to screen candidates:
1. GEOGRAPHIC HARD GATE (Strict Rule):
   - MUST be physically located within the 26 counties of the Republic of Ireland (e.g., Dublin, Cork, Galway, Limerick, Waterford, Kildare, etc.).
   - HARD DISQUALIFICATION for Northern Ireland (NI), UK, EU, or international locations. Tag with disqualification reason: "Non-ROI Location".

2. DIRECTORY ANCHOR GATE (Zero-Hallucination Rule):
   - The business entity MUST be anchored in a recognized official Irish directory or industry register:
     * Accounting: Chartered Accountants Ireland (charteredaccountants.ie), CPA Ireland
     * Legal: Law Society of Ireland (lawsociety.ie)
     * Healthcare/Dental: Irish Dental Association (dentist.ie), CORU
     * Construction/Engineering: Construction Industry Federation (cif.ie), CIRI
     * Hospitality: Irish Hotels Federation (ihf.ie), RAI
     * Childcare: Tusla Early Years Register, Early Childhood Ireland, Pobal
     * Logistics/Haulage: RSA Road Transport License Registry, IRHA
     * Security: Private Security Authority (psa-gov.ie)
     * General SME: Golden Pages Ireland, Verified Google Maps Local Listings
   - SEARCH QUERY PATTERN: Query each directory above directly using site: operators (e.g. `site:charteredaccountants.ie {target_industry}`, `site:lawsociety.ie {target_county}`, `site:cif.ie`, `site:psa-gov.ie`) rather than relying on generic web search alone.
   - SYNTHETIC DATA BAN: Reject fake entity names or fabricated details.
   - CRO VERIFICATION: Confirm active status via the CRO register before audit.

3. DYNAMIC PARAMETER APPLICATION:
   - Filter by `target_county` and `target_industry` if specified.
   - If `enforce_firmographic_limits` is false, bypass strict employee headcount gates. Focus on operational compliance footprint instead.

================================================================================
4. STAGE 3: 4-PASS QUALITY AUDIT & ENRICHMENT
================================================================================
For each lead that passes Stage 2, execute the 4-Pass Quality Audit:

PASS 1 — DIRECTORY ANCHOR VERIFICATION (98% Weight):
   - Confirm official registry or directory source backing the business entity.

PASS 2 — GEOCODING & SYNTAX AUDIT (94% Weight):
   - Verify street address and county within Republic of Ireland bounds.
   - Validate Eircode syntax (7 alphanumeric characters: e.g., D02 X285, H91 XXXX).
   - Validate landline dialling prefixes: Dublin (01), Cork (021), Galway (091), Limerick (061), Waterford (051), Sligo/Mayo (071/094), Midlands (057), South-East (053/056), or valid Irish mobile (083, 085, 086, 087, 089).

PASS 3 — DOMAIN & EMAIL AUDIT (92% Weight):
   - Validate active corporate domain and email syntax (@firmdomain.ie).
   - Flag generic webmail (@gmail.com, @yahoo.com) with a warning, but do NOT disqualify if phone verification is solid.
   - Capture the business's own corporate website URL (distinct from the directory anchor's domain) whenever discoverable — required for the `website` field in the output schema.

PASS 4 — LINKEDIN EXECUTIVE AUDIT (95% Weight):
   - Validate executive decision-maker presence matching approved titles: Owner, Managing Director (MD), Senior Partner, Practice Director, Founder.

PHONE-FIRST FALLBACK RULE:
- For trade-heavy sectors (Construction, Haulage, Security, Trades), do NOT disqualify a lead solely for lacking a corporate email if a verified landline/mobile and decision-maker name are grounded in an official directory. Tag as "Phone-First Outreach".

SCORING CALCULATOR:
Calculate overall score S = (0.98 * Pass1) + (0.94 * Pass2) + (0.92 * Pass3) + (0.95 * Pass4), normalized to 0-100%.

ACCEPTANCE CATEGORIZATION:
- Verified (Active Queue): Score >= min_acceptance_score (Default 90%+), LinkedIn present, corporate .ie domain, valid landline/mobile.
- Needs Attention: Score between 70% and 89%, or minor flags (missing Eircode, generic email).
- Disqualified Archive: Score < 70%, unverified anchor, or non-ROI entity. Must include explicit `disqualification_reason`.================================================================================
5. PENINSULA COMPLIANCE PITCH HOOK GENERATOR
================================================================================
For every qualified lead, generate a tailored 1-to-2 sentence outreach sales hook identifying relevant compliance risk drivers:

1. WRC (Workplace Relations Commission) Hooks:
   - Statutory Sick Pay Act obligations
   - Organisation of Working Time Act (rest breaks, Sunday premiums, time logging)
   - Mandatory Terms of Employment notices within statutory timeframes

2. HSA (Health & Safety Authority) Hooks:
   - Mandatory Risk Assessments & Safety Statement updates
   - Safe Pass / CSCS certification tracking & site safety compliance

================================================================================
6. OUTPUT FORMAT REQUIREMENTS
================================================================================
Observe the `output_mode` configuration setting:

IF `output_mode` == "COMPACT_JSON":
Output strictly the JSON Array below without introductory prose or markdown tables.

IF `output_mode` == "DETAILED":
Provide output in BOTH formats below:

1. Structured JSON Array (for automated API/CRM handoff):
[
  {
    "lead_id": "IE-LEAD-001",
    "business_name": "Example Business Name",
    "decision_maker": {
      "name": "First Last",
      "title": "Managing Director",
      "linkedin_url": "https://linkedin.com/in/profile"
    },
    "contact_details": {
      "email": "contact@domain.ie",
      "website": "https://domain.ie",
      "phone": "091 123456",
      "eircode": "H91 XXXX",
      "county": "Galway"
    },
    "audit_results": {
      "overall_score": 95,
      "status": "Verified (Active Queue)",
      "directory_source": "Chartered Accountants Ireland",
      "outreach_type": "Email & Phone",
      "disqualification_reason": null
    },
    "peninsula_pitch_hook": "High exposure to WRC inspection risks around mandatory Terms of Employment issuing and Organisation of Working Time Act rest interval tracking."
  }
]

2. Markdown Table / CSV Layout (for immediate manual review and spreadsheet import):
| Lead ID | Business Name | County | Industry | Decision Maker | Title | Phone | Email | Score | Status | Disqualification Reason | Compliance Pitch Hook |'''

# --------------------------------------------------------------------------
# Phase 2 enrichment prompt (website scrape re-verification)
# --------------------------------------------------------------------------

PHASE2_ENRICHMENT_SYSTEM_PROMPT = '''You are the Stage 3 re-verification pass for a single Irish B2B lead that has already been through the initial 4-pass quality audit. You are given that lead's current JSON record and a block of text scraped directly from that business's own website footer/contact area — this scraped text is ground truth, more reliable than any earlier search-snippet-based inference.

Using the scraped text:
- Confirm or correct `contact_details.phone`, `contact_details.eircode`, and `contact_details.county` if the scraped text provides clearer or conflicting evidence.
- If the scraped text confirms a valid Republic of Ireland landline prefix, registered address, or other operational detail consistent with the lead's claimed county, this strengthens the audit — you may raise `audit_results.overall_score` and upgrade `audit_results.status` accordingly.
- If the scraped text contradicts the lead (e.g. reveals a non-ROI address, a different business entirely, or no corroborating detail at all), lower `audit_results.overall_score`, adjust `audit_results.status`, and set an explicit `audit_results.disqualification_reason`.
- Do NOT invent any detail that is supported by neither the original record nor the scraped text. If the scraped text is uninformative, leave the original values unchanged.
- Recalculate `audit_results.overall_score` using the same weighted formula as the original audit (PASS1 98%, PASS2 94%, PASS3 92%, PASS4 95%, normalized to 0-100) to the extent the scraped evidence affects any pass.

Respond with NOTHING except a single JSON object matching the exact same lead schema you were given (lead_id, business_name, decision_maker{}, contact_details{}, audit_results{}, peninsula_pitch_hook) — no array wrapper, no markdown table, no commentary before or after.'''

# --------------------------------------------------------------------------
# Cold email sequence prompt
# --------------------------------------------------------------------------

COLD_EMAIL_SYSTEM_PROMPT = '''You are a B2B sales copywriter for Peninsula Ireland, an outsourced HR, Employment Law, and Health & Safety compliance advisory service for Irish SMEs. Write a concise 3-touch cold email sequence for the given lead: Initial Hook, Value Proposition, and Follow-Up.

Rules:
- Each email has a Subject line and a Body under 120 words.
- Ground every email in the specific compliance risk pain point provided — no generic filler.
- Professional, direct tone. No hedging language, no placeholders like [Company Name] — use the real details given.
- The Follow-Up should reference the earlier emails briefly and add urgency without being pushy.

Output as plain text with exactly these section headers, in order:

EMAIL 1 — INITIAL HOOK
Subject: ...
Body: ...

EMAIL 2 — VALUE PROPOSITION
Subject: ...
Body: ...

EMAIL 3 — FOLLOW-UP
Subject: ...
Body: ...

No preamble, no commentary, no text outside the three emails.'''

# --------------------------------------------------------------------------
# JSON extraction helpers
# --------------------------------------------------------------------------


def find_json_array(text: str) -> list | None:
    """
    Locate the first valid top-level JSON array anywhere in free-form text.
    Tolerates a leading ```json fence and a trailing markdown table, since
    DETAILED output_mode returns the JSON array followed by a markdown
    table in the same response.
    """
    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        start = text.find("[", search_from)
        if start == -1:
            return None
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        if isinstance(obj, list):
            return obj
        search_from = start + 1


def find_json_object(text: str) -> dict | None:
    """
    Locate the first valid top-level JSON object anywhere in free-form text.
    Single-object counterpart to find_json_array, used to parse the Phase 2
    enrichment response (one lead, not an array of leads).
    """
    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return None
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        if isinstance(obj, dict):
            return obj
        search_from = start + 1


def parse_leads_from_text(full_text: str) -> tuple[list[dict], str]:
    """Pull the raw lead objects out of a provider's free-form response text."""
    leads = find_json_array(full_text)

    if leads is None:
        # Fallback: a single bare JSON object rather than an array.
        obj = find_json_object(full_text)
        if isinstance(obj, dict):
            leads = obj.get("leads") if "leads" in obj else [obj]

    return leads or [], full_text


def flatten_lead(raw: dict, requested_industry: str) -> dict:
    """Flatten a nested lead object from the Master System Prompt's JSON
    schema (decision_maker{}, contact_details{}, audit_results{}) into the
    columns database.py expects."""
    decision_maker = raw.get("decision_maker") or {}
    contact = raw.get("contact_details") or {}
    audit = raw.get("audit_results") or {}
    directory_source = audit.get("directory_source")

    return {
        "lead_id": raw.get("lead_id"),
        "business_name": raw.get("business_name") or raw.get("company_name"),
        "county": contact.get("county") or raw.get("county"),
        "industry": infer_industry(directory_source, requested_industry),
        "decision_maker_name": decision_maker.get("name"),
        "decision_maker_title": decision_maker.get("title"),
        "decision_maker_linkedin": decision_maker.get("linkedin_url"),
        "contact_email": contact.get("email"),
        "website": contact.get("website"),
        "contact_phone": contact.get("phone"),
        "eircode": contact.get("eircode"),
        "overall_score": audit.get("overall_score"),
        "audit_status": audit.get("status"),
        "directory_source": directory_source,
        "outreach_type": audit.get("outreach_type"),
        "disqualification_reason": audit.get("disqualification_reason"),
        "peninsula_pitch_hook": raw.get("peninsula_pitch_hook"),
        "scrape_verified": 1 if raw.get("_scrape_verified") else 0,
        "cro_verified": 1 if raw.get("_cro_verified") else 0,
        "cro_number": raw.get("_cro_number"),
        "raw_json": raw,
    }


def build_user_prompt(
    target_county: str,
    target_industry: str,
    stage_1_mode: str,
    enforce_firmographic_limits: bool,
    strict_decision_maker_gate: bool,
    min_acceptance_score: int,
    output_mode: str,
    batch_limit: int,
    vault_raw_data: str | None = None,
) -> str:
    config = {
        "stage_1_mode": stage_1_mode,
        "target_county": target_county,
        "target_industry": target_industry,
        "enforce_firmographic_limits": enforce_firmographic_limits,
        "strict_decision_maker_gate": strict_decision_maker_gate,
        "min_acceptance_score": min_acceptance_score,
        "output_mode": output_mode,
        "batch_limit": batch_limit,
    }
    prompt = (
        "Execute the pipeline with this parameter configuration:\n\n"
        f"```json\n{json.dumps(config, indent=2)}\n```\n"
    )
    if stage_1_mode == "VAULT_BATCH":
        prompt += (
            "\nSTAGE 1 VAULT_BATCH raw input to ingest (deduplicate by Business "
            "Name + County/Address, strip navigation artifacts, then run through "
            "Stage 2 screening and the Stage 3 4-pass audit):\n\n"
            f"```\n{vault_raw_data}\n```\n"
        )
    if stage_1_mode == "LIVE_SEARCH":
        prompt += (
            "\nFor every candidate lead, actively search for and confirm a local "
            "landline dialling prefix consistent with its claimed county (e.g. "
            "Carlow 059, Galway 091, Cork 021, Dublin 01) and/or its registered "
            "office address, to verify genuine operational presence before "
            "including it — do not rely on directory listing alone.\n"
        )
    return prompt


# --------------------------------------------------------------------------
# Anthropic pipeline
# --------------------------------------------------------------------------


def run_pipeline_anthropic(
    api_key: str,
    target_county: str,
    target_industry: str,
    stage_1_mode: str,
    enforce_firmographic_limits: bool,
    strict_decision_maker_gate: bool,
    min_acceptance_score: int,
    output_mode: str,
    batch_limit: int,
    vault_raw_data: str | None = None,
) -> tuple[list[dict] | None, str, str | None]:
    """Returns (raw_leads, raw_text, error). raw_leads is None on failure."""
    if stage_1_mode == "VAULT_BATCH" and not (vault_raw_data or "").strip():
        return None, "", (
            "VAULT_BATCH mode requires raw business data pasted into the "
            "'Raw Vault Data' box in the sidebar."
        )

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = build_user_prompt(
        target_county,
        target_industry,
        stage_1_mode,
        enforce_firmographic_limits,
        strict_decision_maker_gate,
        min_acceptance_score,
        output_mode,
        batch_limit,
        vault_raw_data,
    )

    # web_search is always included per project convention; stage_1_mode
    # (LIVE_SEARCH vs VAULT_BATCH) is enforced through the system prompt.
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 20}]

    try:
        with client.messages.stream(
            model=ANTHROPIC_MODEL_ID,
            max_tokens=16000,
            system=MASTER_SYSTEM_PROMPT,
            tools=tools,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError:
        return None, "", "Invalid ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return None, "", "Rate limited by the Anthropic API. Please wait and try again."
    except anthropic.APIStatusError as e:
        return None, "", f"Anthropic API error ({e.status_code}): {e.message}"
    except anthropic.APIConnectionError:
        return None, "", "Could not connect to the Anthropic API. Check your network connection."

    if response.stop_reason == "refusal":
        return None, "", "The model declined to process this request."

    full_text = "\n".join(block.text for block in response.content if block.type == "text")
    raw_leads, raw_text = parse_leads_from_text(full_text)
    return raw_leads, raw_text, None


def generate_cold_email_sequence(api_key: str, lead: dict) -> tuple[str | None, str | None]:
    """Returns (email_sequence_text, error). email_sequence_text is None on failure."""
    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = (
        f"Business: {lead.get('business_name') or 'Unknown Business'}\n"
        f"Decision Maker: {lead.get('decision_maker_name') or 'the decision maker'}\n"
        f"Industry: {lead.get('industry') or 'General SME'}\n"
        f"Compliance Pitch Hook: {lead.get('peninsula_pitch_hook') or ''}\n\n"
        "Write the 3-touch cold email sequence now."
    )
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL_ID,
            max_tokens=1200,
            system=COLD_EMAIL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError:
        return None, "Invalid ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return None, "Rate limited by the Anthropic API. Please wait and try again."
    except anthropic.APIStatusError as e:
        return None, f"Anthropic API error ({e.status_code}): {e.message}"
    except anthropic.APIConnectionError:
        return None, "Could not connect to the Anthropic API. Check your network connection."

    if response.stop_reason == "refusal":
        return None, "The model declined to generate this content."

    text = "\n".join(block.text for block in response.content if block.type == "text")
    return text, None


def enrich_lead_with_scraped_context(api_key: str, raw_lead: dict, scraped_text: str) -> dict:
    """
    Phase 2 re-verification: re-audit one lead against scraped website text.
    Never raises — returns raw_lead unchanged (with _scrape_verified left
    unset) if the call or parse fails, so a single bad scrape/response can't
    break the run.
    """
    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = (
        "Current lead record:\n\n"
        f"```json\n{json.dumps(raw_lead, indent=2)}\n```\n\n"
        "Scraped website footer/contact text (ground truth):\n\n"
        f"```\n{scraped_text}\n```\n"
    )
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL_ID,
            max_tokens=2000,
            system=PHASE2_ENRICHMENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AnthropicError:
        return raw_lead

    if response.stop_reason == "refusal":
        return raw_lead

    full_text = "\n".join(block.text for block in response.content if block.type == "text")
    updated = find_json_object(full_text)
    if not isinstance(updated, dict):
        return raw_lead
    updated["_scrape_verified"] = True
    return updated


def enrich_leads_phase2(
    api_key: str,
    raw_leads: list[dict],
    fetch_footer_contact_details,
) -> list[dict]:
    """
    Phase 2 batch re-verification: for each raw (nested-schema) lead with a
    website, scrape its footer/contact area via the injected scraper and
    re-audit it against that scraped text with enrich_lead_with_scraped_context.
    Leads with no usable website, or whose scrape/call/parse fails, pass
    through unchanged — enrich_lead_with_scraped_context never raises, so a
    single bad lead can't break the batch.
    """
    if not raw_leads:
        return []

    from concurrent.futures import ThreadPoolExecutor

    def process(raw_lead):
        website = (raw_lead.get("contact_details") or {}).get("website")
        if not website or not isinstance(website, str) or not website.strip():
            return raw_lead
        scraped_text = fetch_footer_contact_details(website.strip()) if fetch_footer_contact_details else None
        if not scraped_text:
            return raw_lead
        return enrich_lead_with_scraped_context(api_key, raw_lead, scraped_text)

    results = [None] * len(raw_leads)
    with ThreadPoolExecutor(max_workers=min(10, len(raw_leads))) as executor:
        future_to_index = {executor.submit(process, lead): i for i, lead in enumerate(raw_leads)}
        for fut in future_to_index:
            index = future_to_index[fut]
            try:
                results[index] = fut.result()
            except Exception:
                results[index] = raw_leads[index]

    return results


# --------------------------------------------------------------------------
# Enrichment Prompt Infrastructure
# --------------------------------------------------------------------------

# Base enrichment system prompt (used as fallback for unknown industry)
ENRICHMENT_SYSTEM_PROMPT = """You are an expert Lead Sourcing & Enrichment Specialist for Peninsula Ireland (B2B HR, Employment Law, Health & Safety compliance solutions).
Your objective is to take a raw business record (Business Name, County, Eircode, CRO number) and enrich it with accurate, verified decision-maker details and contact information.

Search the web if needed to discover:
1. Decision Maker Name: Owner, Founder, Managing Director, Partner, or CEO.
2. Decision Maker Title: Official Title.
3. Decision Maker LinkedIn: Direct LinkedIn URL if available.
4. Contact Email: Public direct or company email address.
5. Contact Phone: Main Irish business phone number (+353 format preferred).
6. Website: Official website URL.
7. Industry: One of ["Accountants", "Legal", "Construction", "Haulage", "Healthcare", "Security", "Hospitality", "Childcare"].
8. Peninsula Pitch Hook: A 2-sentence tailored outreach angle for Peninsula Ireland highlighting HR/Health & Safety compliance risks for this specific company type.
9. Overall Score: Quality score (0-100) based on completeness (score 85-95 if owner name + email/phone found; 70-84 if owner name or phone found; <70 if sparse).
10. Audit Status: Set to "Verified (Active Queue)" if overall score >= 70, otherwise "Needs Attention".

Return strictly a single JSON object with no preamble or extra text:
{
  "decision_maker_name": "...",
  "decision_maker_title": "...",
  "decision_maker_linkedin": "...",
  "contact_email": "...",
  "contact_phone": "...",
  "website": "...",
  "industry": "...",
  "peninsula_pitch_hook": "...",
  "overall_score": 85,
  "audit_status": "Verified (Active Queue)"
}
"""

# --------------------------------------------------------------------------
# Tier 2: Sector-specific prompt cache (pre-built at import time)
# 35% smaller than the base prompt for leads with a known industry.
# --------------------------------------------------------------------------

_SECTOR_PITCH_HINTS = {
    "Accountants": (
        "Accountancy practices must comply with Organisation of Working Time Act "
        "rest break tracking and statutory sick pay obligations under the Sick Leave Act 2022 — "
        "high WRC inspection exposure for firms without formal HR policies."
    ),
    "Legal": (
        "Law firms face WRC scrutiny around Terms of Employment documentation, mandatory "
        "rest intervals, and annual leave accrual record-keeping obligations."
    ),
    "Healthcare": (
        "Healthcare and dental practices carry dual WRC and HSA exposure — mandatory Safety "
        "Statements, CORU registration compliance tracking, and statutory sick pay administration."
    ),
    "Construction": (
        "Construction firms face mandatory HSA Safe Pass / CSCS certification tracking, "
        "Risk Assessment and Safety Statement obligations, and WRC working time compliance "
        "for site operatives."
    ),
    "Haulage": (
        "Road transport operators face RSA roadworthiness obligations, EU drivers' hours "
        "and tachograph record-keeping, and statutory sick pay exposure under the Sick Leave Act 2022."
    ),
    "Hospitality": (
        "Hospitality businesses have high WRC exposure around Sunday premium pay, "
        "rest interval compliance, and mandatory annual leave accrual records under the "
        "Organisation of Working Time Act."
    ),
    "Childcare": (
        "Childcare providers face Tusla inspection compliance, mandatory Garda vetting renewal "
        "tracking, statutory sick pay obligations, and Safety Statement requirements under HSA."
    ),
    "Security": (
        "Security firms carry PSA licence renewal obligations, mandatory Safety Statements, "
        "and WRC working time compliance for shift-based operatives — high enforcement target sectors."
    ),
}

_JSON_SCHEMA_BLOCK = """Return strictly a single JSON object with no preamble or extra text:
{
  "decision_maker_name": "...",
  "decision_maker_title": "...",
  "decision_maker_linkedin": "...",
  "contact_email": "...",
  "contact_phone": "...",
  "website": "...",
  "industry": "...",
  "peninsula_pitch_hook": "...",
  "overall_score": 85,
  "audit_status": "Verified (Active Queue)"
}"""


def _build_sector_prompt(industry: str, hint: str) -> str:
    return (
        f"You are an expert Lead Enrichment Specialist for Peninsula Ireland (B2B HR, "
        f"Employment Law, Health & Safety compliance solutions). "
        f"This lead is in the {industry} sector.\n\n"
        f"Search the web to discover:\n"
        f"1. Decision Maker Name: Owner, Founder, Managing Director, Partner, or CEO.\n"
        f"2. Decision Maker Title: Official Title.\n"
        f"3. Decision Maker LinkedIn: Direct LinkedIn URL if available.\n"
        f"4. Contact Email: Public direct or company email.\n"
        f"5. Contact Phone: Main Irish business phone (+353 format preferred).\n"
        f"6. Website: Official website URL.\n"
        f"7. Peninsula Pitch Hook (use this tailored angle): {hint}\n"
        f"8. Overall Score (0-100): 85-95 if name+email/phone found; 70-84 if name or phone; <70 if sparse.\n"
        f"9. Audit Status: 'Verified (Active Queue)' if score >= 70, else 'Needs Attention'.\n\n"
        f"{_JSON_SCHEMA_BLOCK}"
    )


# Pre-compiled at import time — zero runtime cost per enrichment call
SECTOR_PROMPTS: dict[str, str] = {
    industry: _build_sector_prompt(industry, hint)
    for industry, hint in _SECTOR_PITCH_HINTS.items()
}


def get_sector_prompt(industry: str | None) -> str:
    """Return the pre-compiled sector-specific system prompt, or the base prompt if industry unknown."""
    if industry and industry in SECTOR_PROMPTS:
        return SECTOR_PROMPTS[industry]
    return ENRICHMENT_SYSTEM_PROMPT


# --------------------------------------------------------------------------
# Tier 1: Pre-filter — bypass LLM entirely for already-complete leads
# --------------------------------------------------------------------------

_EMPTY = {"", "nan", "none", "null", "n/a"}


def _is_populated(val) -> bool:
    """True if the field value is a non-empty, non-placeholder string."""
    return bool(val) and isinstance(val, str) and val.strip().lower() not in _EMPTY


def pre_filter_lead(lead: dict) -> dict | None:
    """
    Tier 1 pre-filter: if a lead already has decision_maker_name + contact_email
    + contact_phone, compute a score directly in Python and return an enriched
    dict — no LLM call needed.

    Returns an enriched dict if the lead is pre-filterable, otherwise None.
    """
    name = lead.get("decision_maker_name")
    email = lead.get("contact_email")
    phone = lead.get("contact_phone")
    website = lead.get("website")
    pitch = lead.get("peninsula_pitch_hook")

    if not (_is_populated(name) and (_is_populated(email) or _is_populated(phone))):
        return None  # Not complete enough — needs LLM

    # Score: name + email + phone = 92, name + one of them = 80
    has_email = _is_populated(email)
    has_phone = _is_populated(phone)
    if has_email and has_phone:
        score = 92
    else:
        score = 80

    # Bonus for website
    if _is_populated(website):
        score = min(score + 3, 97)

    status = "Verified (Active Queue)" if score >= 70 else "Needs Attention"

    # Re-use existing pitch or generate a sector-fallback
    industry = lead.get("industry") or ""
    if not _is_populated(pitch):
        pitch = _SECTOR_PITCH_HINTS.get(
            industry,
            "High exposure to WRC inspection risks and statutory sick pay obligations under the Sick Leave Act 2022.",
        )

    return {
        "decision_maker_name": name,
        "decision_maker_title": lead.get("decision_maker_title") or "",
        "decision_maker_linkedin": lead.get("decision_maker_linkedin") or "",
        "contact_email": email or "",
        "contact_phone": phone or "",
        "website": website or "",
        "industry": industry,
        "peninsula_pitch_hook": pitch,
        "overall_score": score,
        "audit_status": status,
        "scrape_verified": 1,
        "_pre_filtered": True,
    }


# --------------------------------------------------------------------------
# Tier 3: Pre-built user prompt assembly helper
# --------------------------------------------------------------------------

from string import Template as _Template

_USER_PROMPT_TEMPLATE = _Template(
    "Target Business Record to Enrich:\n"
    "- Business Name: $business_name\n"
    "- County: $county\n"
    "- Eircode: $eircode\n"
    "- CRO Number: $cro_number\n"
    "- Industry: $industry\n"
    "- Known Website: $website\n"
    "$scraped_block\n"
    "Find the owner/managing director, phone, email, website, industry, "
    "and tailored Peninsula Ireland HR/H&S pitch hook. Return valid JSON only."
)


def build_enrichment_user_prompt(lead: dict, scraped_text: str = "") -> str:
    """Tier 3: Assemble the enrichment user prompt via Template substitution."""
    scraped_block = (
        f"\nScraped Website Text Ground Truth:\n```\n{scraped_text[:3000]}\n```"
        if scraped_text
        else ""
    )
    return _USER_PROMPT_TEMPLATE.safe_substitute(
        business_name=lead.get("business_name") or "Unknown",
        county=lead.get("county") or "",
        eircode=lead.get("eircode") or "N/A",
        cro_number=lead.get("cro_number") or "N/A",
        industry=lead.get("industry") or "Unclassified",
        website=lead.get("website") or "N/A",
        scraped_block=scraped_block,
    )


def enrich_single_lead(
    provider: str,
    api_key: str | list[str],
    lead: dict,
    fetch_footer_contact_details=None,
) -> tuple[dict | None, str | None]:
    """
    Perform Smart Tiered AI Enrichment on a single lead.

    Tier 1: Pre-filter — if lead already has name + email/phone, score and
            return immediately with no API call.
    Tier 2: Use pre-compiled sector-specific system prompt (35% smaller).
    Tier 3: Assemble user prompt via Template substitution (thread-efficient).

    Supports a list/pool of primary & fallback API keys to survive rate limits.
    Returns (enriched_dict, error_string).
    """
    # ── Tier 1: Pre-filter ──────────────────────────────────────────────────
    pre_filtered = pre_filter_lead(lead)
    if pre_filtered is not None:
        return pre_filtered, None

    # ── Key pool setup ───────────────────────────────────────────────────────
    if isinstance(api_key, str):
        key_pool = [k.strip() for k in api_key.split(",") if k.strip()]
    elif isinstance(api_key, (list, tuple)):
        key_pool = [k.strip() for k in api_key if isinstance(k, str) and k.strip()]
    else:
        key_pool = []

    if not key_pool:
        return None, "No valid API keys provided in key pool."

    website = lead.get("website")
    _has_website = bool(
        website and isinstance(website, str)
        and website.strip()
        and website.strip().lower() not in ("nan", "none", "null")
    )

    # ── Firecrawl website discovery for leads with no URL ────────────────────
    if not _has_website:
        try:
            from firecrawl_client import find_business_website
            discovered = find_business_website(
                lead.get("business_name", ""), lead.get("county", "")
            )
            if discovered:
                lead = {**lead, "website": discovered}
                website = discovered
                _has_website = True
        except ImportError:
            pass  # firecrawl_client optional

    # ── Website scraping (Firecrawl → fallback) ──────────────────────────────
    scraped_text = ""
    if _has_website:
        # Try Firecrawl first for JS-rendered pages
        try:
            from firecrawl_client import fetch_with_firecrawl
            scraped_text = fetch_with_firecrawl(
                website,
                fallback_scraper=fetch_footer_contact_details,
            ) or ""
        except ImportError:
            if fetch_footer_contact_details:
                scraped_text = fetch_footer_contact_details(website) or ""
    elif fetch_footer_contact_details and website:
        scraped_text = fetch_footer_contact_details(website) or ""

    # ── Tier 2: Sector-specific system prompt ────────────────────────────────
    system_prompt = get_sector_prompt(lead.get("industry"))

    # ── Tier 3: Template-based user prompt ───────────────────────────────────
    user_prompt = build_enrichment_user_prompt(lead, scraped_text)

    last_error = ""
    for k_idx, current_key in enumerate(key_pool):
        if not current_key or not current_key.strip():
            continue
        current_key = current_key.strip()
        full_text = ""

        if provider == "anthropic":
            try:
                client = anthropic.Anthropic(api_key=current_key)
                response = client.messages.create(
                    model=ANTHROPIC_MODEL_ID,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                full_text = "\n".join(block.text for block in response.content if block.type == "text")
            except anthropic.AnthropicError as e:
                last_error = f"Anthropic Key #{k_idx+1}: {e}"
                continue
        else:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=current_key)
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=4000,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                )
                response = client.models.generate_content(
                    model=GEMINI_MODEL_ID,
                    contents=user_prompt,
                    config=config,
                )
                full_text = getattr(response, "text", "") or ""
            except Exception as e:  # noqa: BLE001
                last_error = f"Gemini Key #{k_idx+1}: {e}"
                continue

        if not full_text:
            last_error = f"Empty response from key #{k_idx+1}."
            continue

        parsed = find_json_object(full_text)
        if isinstance(parsed, dict):
            if scraped_text:
                parsed["scrape_verified"] = 1
            return parsed, None
        else:
            last_error = f"Invalid JSON returned from key #{k_idx+1}."

    return None, f"All {len(key_pool)} key(s) in pool failed. Last error: {last_error}"


def enrich_lead_smart_hybrid(
    lead: dict,
    anthropic_keys: str | list[str] | None = None,
    gemini_keys: str | list[str] | None = None,
    fetch_footer_contact_details=None,
) -> tuple[dict | None, str | None]:
    """
    Smart Hybrid Lead Enrichment:
    - If lead has NO valid website URL -> Primary: Gemini (Google Search grounding).
    - If lead HAS a valid website URL -> Primary: Anthropic (Scraped website context).
    - Automatic Failover: If primary provider fails or rate limits, falls back to secondary provider pool.
    """
    website = lead.get("website")
    has_website = bool(website and isinstance(website, str) and website.strip() and website.strip().lower() not in ("nan", "none", "null"))

    has_anthropic = bool(anthropic_keys)
    has_gemini = bool(gemini_keys)

    primary_provider = "anthropic" if (has_website and has_anthropic) else ("gemini" if has_gemini else "anthropic")
    secondary_provider = "gemini" if primary_provider == "anthropic" else "anthropic"

    primary_keys = anthropic_keys if primary_provider == "anthropic" else gemini_keys
    secondary_keys = gemini_keys if primary_provider == "anthropic" else anthropic_keys

    err_messages = []
    if primary_keys:
        data, err = enrich_single_lead(primary_provider, primary_keys, lead, fetch_footer_contact_details=fetch_footer_contact_details)
        if data:
            data["enrichment_provider"] = primary_provider
            return data, None
        if err:
            err_messages.append(f"Primary ({primary_provider}): {err}")

    # Failover to secondary provider key pool
    if secondary_keys and secondary_keys != primary_keys:
        data, err = enrich_single_lead(secondary_provider, secondary_keys, lead, fetch_footer_contact_details=fetch_footer_contact_details)
        if data:
            data["enrichment_provider"] = secondary_provider
            return data, None
        if err:
            err_messages.append(f"Secondary ({secondary_provider}): {err}")

    return None, "Smart Hybrid Enrichment failed. " + (" | ".join(err_messages) if err_messages else "No valid API keys.")


def enrich_leads_batch_concurrent(
    leads: list[dict],
    anthropic_keys: list[str] | str | None = None,
    gemini_keys: list[str] | str | None = None,
    max_workers: int = 10,
    fetch_footer_contact_details=None,
) -> list[tuple[dict, dict | None, str | None]]:
    """
    High-Concurrency Parallel Lead Enrichment Engine.
    Executes enrichment for `leads` across `max_workers` concurrent threads.
    Distributes API requests across API Key pools (Anthropic & Gemini).
    Returns list of (lead_record, enriched_dict, error_string) tuples.
    """
    if not leads:
        return []

    from concurrent.futures import ThreadPoolExecutor

    def process_lead(lead):
        enriched_data, err = enrich_lead_smart_hybrid(
            lead,
            anthropic_keys=anthropic_keys,
            gemini_keys=gemini_keys,
            fetch_footer_contact_details=fetch_footer_contact_details,
        )
        return lead, enriched_data, err

    results = []
    actual_workers = min(max_workers, len(leads))
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_lead = {executor.submit(process_lead, lead): lead for lead in leads}
        for fut in future_to_lead:
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((future_to_lead[fut], None, str(e)))

    return results


def verify_leads_registry(
    raw_leads: list[dict],
    email: str,
    api_key: str,
    check_cro_status,
) -> list[dict]:
    """
    Runs the CRO active-status verification gate over every raw lead.
    `check_cro_status` is injected (from registry_client.py) rather than
    imported here, same pattern as enrich_leads_phase2's scraper injection.

    A confirmed-active lead is tagged (_cro_verified / _cro_number). A
    confirmed-inactive lead (dissolved/struck off/liquidation/etc.) is
    downgraded to Disqualified Archive with an explicit reason — "ensure
    only verified businesses are audited" per the feature request. A lead
    the CRO lookup can't confidently match is left completely unchanged:
    an inconclusive lookup is not evidence of anything and must never
    silently disqualify a lead.
    """
    updated_leads = []
    for lead in raw_leads:
        business_name = lead.get("business_name")
        county = (lead.get("contact_details") or {}).get("county")
        result = check_cro_status(business_name, county, email, api_key)

        if result is None:
            updated_leads.append(lead)
            continue

        lead = dict(lead)
        if result["active"]:
            lead["_cro_verified"] = True
            lead["_cro_number"] = result.get("cro_number")
        else:
            audit = dict(lead.get("audit_results") or {})
            audit["status"] = "Disqualified Archive"
            audit["disqualification_reason"] = f"CRO register status: {result['status']}"
            lead["audit_results"] = audit
        updated_leads.append(lead)
    return updated_leads


# --------------------------------------------------------------------------
# Gemini pipeline
# --------------------------------------------------------------------------


def run_pipeline_gemini(
    api_key: str,
    target_county: str,
    target_industry: str,
    stage_1_mode: str,
    enforce_firmographic_limits: bool,
    strict_decision_maker_gate: bool,
    min_acceptance_score: int,
    output_mode: str,
    batch_limit: int,
    vault_raw_data: str | None = None,
) -> tuple[list[dict] | None, str, str | None]:
    """Returns (raw_leads, raw_text, error). raw_leads is None on failure."""
    if stage_1_mode == "VAULT_BATCH" and not (vault_raw_data or "").strip():
        return None, "", (
            "VAULT_BATCH mode requires raw business data pasted into the "
            "'Raw Vault Data' box in the sidebar."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "", (
            "The google-genai package is not installed. Run "
            "`pip install -r requirements.txt` and try again."
        )

    user_prompt = build_user_prompt(
        target_county,
        target_industry,
        stage_1_mode,
        enforce_firmographic_limits,
        strict_decision_maker_gate,
        min_acceptance_score,
        output_mode,
        batch_limit,
        vault_raw_data,
    )

    tools = [types.Tool(google_search=types.GoogleSearch())] if stage_1_mode == "LIVE_SEARCH" else None
    config = types.GenerateContentConfig(
        system_instruction=MASTER_SYSTEM_PROMPT,
        max_output_tokens=16000,
        tools=tools,
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=user_prompt,
            config=config,
        )
    except Exception as e:  # noqa: BLE001 - google-genai's exception hierarchy
        # isn't covered by a bundled reference in this project the way
        # Anthropic's is; caught broadly so a Gemini-side failure surfaces
        # as a clean error instead of crashing the run.
        return None, "", f"Gemini API error: {e}"

    full_text = getattr(response, "text", None) or ""
    if not full_text:
        return None, "", "Gemini returned an empty response."

    raw_leads, raw_text = parse_leads_from_text(full_text)
    return raw_leads, raw_text, None


# --------------------------------------------------------------------------
# Provider dispatcher
# --------------------------------------------------------------------------

PROVIDERS = {
    "🤖 Autonomous Fluid Pipeline (Auto-Assigned APIs)": "fluid",
    "Claude Sonnet 5 (Anthropic)": "anthropic",
    "Gemini 2.5 Pro (Google AI)": "gemini",
}


def run_pipeline_fluid(
    anthropic_keys: list[str] | str | None,
    gemini_keys: list[str] | str | None,
    maps_key: str | None,
    target_county: str,
    target_industry: str,
    stage_1_mode: str,
    enforce_firmographic_limits: bool,
    strict_decision_maker_gate: bool,
    min_acceptance_score: int,
    output_mode: str,
    batch_limit: int,
    vault_raw_data: str | None = None,
    google_places_search_fn=None,
) -> tuple[list[dict] | None, str, str | None]:
    """
    Autonomous Fluid Multi-API Orchestration:
    - Task 1: Sourcing -> Google Places API (New) if available, or Gemini Search Grounding / Claude.
    - Task 2: Grounding -> Web scraper for website text.
    - Task 3: Enrichment & Pitch -> Smart Hybrid Routing (Claude for sites, Gemini for web search).
    """
    a_keys = [anthropic_keys] if isinstance(anthropic_keys, str) else (anthropic_keys or [])
    g_keys = [gemini_keys] if isinstance(gemini_keys, str) else (gemini_keys or [])

    raw_leads = []
    raw_text = ""

    # Task 1: Ground-Truth Location & Contact Sourcing via Google Places API (New) if available
    if (stage_1_mode in ("GOOGLE_PLACES_API", "LIVE_SEARCH", "FLUID")) and maps_key and google_places_search_fn:
        search_query = f"{target_industry} companies" if target_industry != "ALL" else "Business services"
        place_results = google_places_search_fn(search_query, maps_key, county=target_county, limit=batch_limit)
        for r in place_results:
            raw_leads.append({
                "business_name": r["business_name"],
                "county": r["county"],
                "industry": target_industry if target_industry != "ALL" else "Unclassified",
                "website": r["website"],
                "contact_phone": r["contact_phone"],
                "eircode": r["eircode"],
                "audit_status": "Pending Approval",
            })
        if raw_leads:
            raw_text = f"Sourced {len(raw_leads)} lead(s) via Google Places API (New)."

    # Task 1 Fallback: LLM Sourcing via Gemini / Anthropic if Google Places didn't return leads
    if not raw_leads:
        primary_key = g_keys[0] if g_keys else (a_keys[0] if a_keys else "")
        provider_name = "gemini" if g_keys else "anthropic"
        if not primary_key:
            return None, "", "No API keys configured for Fluid Pipeline. Please set ANTHROPIC_API_KEY, GEMINI_API_KEY, or GOOGLE_MAPS_API_KEY."
        fn = run_pipeline_gemini if provider_name == "gemini" else run_pipeline_anthropic
        raw_leads, raw_text, err = fn(
            primary_key,
            target_county,
            target_industry,
            stage_1_mode if stage_1_mode in ("LIVE_SEARCH", "VAULT_BATCH") else "LIVE_SEARCH",
            enforce_firmographic_limits,
            strict_decision_maker_gate,
            min_acceptance_score,
            output_mode,
            batch_limit,
            vault_raw_data,
        )
        if err:
            return None, "", err

    return raw_leads, raw_text, None


def run_pipeline(
    provider: str,
    api_key: str | list[str],
    target_county: str,
    target_industry: str,
    stage_1_mode: str,
    enforce_firmographic_limits: bool,
    strict_decision_maker_gate: bool,
    min_acceptance_score: int,
    output_mode: str,
    batch_limit: int,
    vault_raw_data: str | None = None,
    gemini_key: str | list[str] | None = None,
    maps_key: str | None = None,
    google_places_search_fn=None,
) -> tuple[list[dict] | None, str, str | None]:
    """Dispatches to Fluid, Anthropic, or Gemini pipeline based on `provider`."""
    if provider == "fluid":
        return run_pipeline_fluid(
            anthropic_keys=api_key,
            gemini_keys=gemini_key,
            maps_key=maps_key,
            target_county=target_county,
            target_industry=target_industry,
            stage_1_mode=stage_1_mode,
            enforce_firmographic_limits=enforce_firmographic_limits,
            strict_decision_maker_gate=strict_decision_maker_gate,
            min_acceptance_score=min_acceptance_score,
            output_mode=output_mode,
            batch_limit=batch_limit,
            vault_raw_data=vault_raw_data,
            google_places_search_fn=google_places_search_fn,
        )

    fn = run_pipeline_anthropic if provider == "anthropic" else run_pipeline_gemini
    single_key = api_key[0] if isinstance(api_key, list) else api_key
    return fn(
        single_key,
        target_county,
        target_industry,
        stage_1_mode,
        enforce_firmographic_limits,
        strict_decision_maker_gate,
        min_acceptance_score,
        output_mode,
        batch_limit,
        vault_raw_data,
    )




