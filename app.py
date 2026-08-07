"""
Peninsula Ireland — B2B Lead Sourcing CRM

A Streamlit front end that drives an LLM provider (Anthropic or Gemini) to
source, screen, audit, and persist qualified B2B leads for Peninsula
Ireland's HR / Employment Law / Health & Safety outsourcing service.

All LLM-calling logic lives in llm_client.py; website scraping lives in
web_scraper.py; CRO registry verification lives in registry_client.py;
persistence lives in database.py. This file is UI-only.
"""

import importlib
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import database as db
import firecrawl_client
import google_places_client
import llm_client
import registry_client
import web_scraper

# Force reload of local modules on Streamlit rerun to avoid stale module cache errors
importlib.reload(db)
importlib.reload(firecrawl_client)
importlib.reload(google_places_client)
importlib.reload(llm_client)
importlib.reload(registry_client)
importlib.reload(web_scraper)

load_dotenv(override=True)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ROI_COUNTIES = [
    "Carlow", "Cavan", "Clare", "Cork", "Donegal", "Dublin", "Galway",
    "Kerry", "Kildare", "Kilkenny", "Laois", "Leitrim", "Limerick",
    "Longford", "Louth", "Mayo", "Meath", "Monaghan", "Offaly",
    "Roscommon", "Sligo", "Tipperary", "Waterford", "Westmeath",
    "Wexford", "Wicklow",
]

TARGET_INDUSTRIES = [
    "Accountants", "Legal", "Construction", "Haulage", "Healthcare",
    "Security", "Hospitality", "Childcare",
]

STATUS_OPTIONS = ["New", "Contacted", "Qualified", "Rejected", "Won"]
AUDIT_STATUS_OPTIONS = ["Verified (Active Queue)", "Needs Attention", "Disqualified Archive"]

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def get_api_key(env_var_name: str) -> str | None:
    pool = get_api_key_pool(env_var_name)
    return pool[0] if pool else None


def get_api_key_pool(env_var_name: str) -> list[str]:
    """
    Returns a list of all configured API keys for a provider (primary + fallback keys).
    Supports:
    - Comma-separated keys in env_var_name: e.g. ANTHROPIC_API_KEY="key1, key2, key3"
    - Sequential numbered env vars: ANTHROPIC_API_KEY_2, ANTHROPIC_API_KEY_3, etc.
    - Session state UI overrides: override_ANTHROPIC_API_KEY, override_ANTHROPIC_API_KEY_2
    """
    load_dotenv(override=True)
    keys = []
    
    # 1. Check UI Session State Overrides
    override = st.session_state.get(f"override_{env_var_name}")
    if override and override.strip():
        for k in override.split(","):
            if k.strip():
                keys.append(k.strip())

    override_2 = st.session_state.get(f"override_{env_var_name}_2")
    if override_2 and override_2.strip():
        for k in override_2.split(","):
            if k.strip() and k.strip() not in keys:
                keys.append(k.strip())

    # 2. Check main env var (supports comma-separated list)
    main_env = os.getenv(env_var_name)
    if main_env and main_env.strip():
        for k in main_env.split(","):
            if k.strip() and k.strip() not in keys:
                keys.append(k.strip())

    # 3. Check fallback env vars (e.g. ANTHROPIC_API_KEY_2, ANTHROPIC_API_KEY_3, ...)
    for i in range(2, 10):
        alt_key = os.getenv(f"{env_var_name}_{i}")
        if alt_key and alt_key.strip() and alt_key.strip() not in keys:
            keys.append(alt_key.strip())

    # 4. Check Streamlit secrets
    try:
        val = st.secrets[env_var_name]
        if val and str(val).strip():
            for k in str(val).split(","):
                if k.strip() and k.strip() not in keys:
                    keys.append(k.strip())
    except Exception:
        pass

    return keys


def audit_status_indicator(audit_status: str | None) -> str:
    if not audit_status:
        return "⚪"
    status_lower = audit_status.lower()
    if "verified" in status_lower or "active queue" in status_lower:
        return "🟢"
    if "attention" in status_lower:
        return "🟡"
    if "disqualified" in status_lower:
        return "🔴"
    return "⚪"


def score_indicator(score) -> str:
    try:
        score = int(score)
    except (TypeError, ValueError):
        return "⚪"
    if score >= 90:
        return "🟢"
    if score >= 70:
        return "🟡"
    return "🔴"


def render_lead_card(lead: dict, key_prefix: str):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Business Name:** {lead.get('business_name') or '—'}")
        st.markdown(
            f"**County:** {lead.get('county') or '—'}  |  "
            f"**Industry:** {lead.get('industry') or 'Unclassified'}"
        )
        st.markdown(f"**Eircode:** {lead.get('eircode') or '—'}")
        st.markdown(f"**Directory Source:** {lead.get('directory_source') or '—'}")
        st.markdown(f"**Outreach Type:** {lead.get('outreach_type') or '—'}")
        if lead.get("website"):
            st.markdown(f"🔗 [Website]({lead['website']})")
        if lead.get("lead_id"):
            st.caption(f"Lead ID: {lead['lead_id']}")
        if lead.get("scrape_verified"):
            st.caption("🔍 Site-verified (Phase 2)")
        if lead.get("cro_verified"):
            cro_number = lead.get("cro_number")
            st.caption(f"🏛️ CRO active{f' — No. {cro_number}' if cro_number else ''}")

    with col2:
        st.markdown("**Direct Decision Maker**")
        st.markdown(f"👤 {lead.get('decision_maker_name') or 'Not identified'}")
        st.markdown(f"💼 {lead.get('decision_maker_title') or '—'}")
        st.markdown(f"📧 {lead.get('contact_email') or 'Not publicly available'}")
        st.markdown(f"📞 {lead.get('contact_phone') or 'Not publicly available'}")
        if lead.get("decision_maker_linkedin"):
            st.markdown(f"🔗 [LinkedIn Profile]({lead['decision_maker_linkedin']})")

    score = lead.get("overall_score")
    audit_status = lead.get("audit_status")
    st.markdown(
        f"**Audit Result:** {audit_status_indicator(audit_status)} "
        f"{audit_status or 'Unknown'} — Score: {score if score is not None else '—'}"
    )
    if lead.get("disqualification_reason"):
        st.caption(f"⚠️ Disqualification reason: {lead['disqualification_reason']}")

    hook_text = lead.get("peninsula_pitch_hook") or "No pitch hook generated yet."
    st.markdown(
        f'<div class="peninsula-pitch-card">'
        f'<div class="peninsula-pitch-title">🎯 Peninsula Compliance Pitch Hook</div>'
        f'<div>{hook_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    email_state_key = f"cold_email_{key_prefix}"
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("✉️ Generate Cold Email", key=f"email_btn_{key_prefix}"):
            anthropic_key = get_api_key("ANTHROPIC_API_KEY")
            if not anthropic_key:
                st.error("No ANTHROPIC_API_KEY found — the email generator always uses Anthropic.")
            else:
                with st.spinner("Writing cold email sequence..."):
                    sequence, error = llm_client.generate_cold_email_sequence(anthropic_key, lead)
                if error:
                    st.error(error)
                else:
                    st.session_state[email_state_key] = sequence

    with btn_col2:
        if st.button("🔍 AI Enrich Lead", key=f"enrich_btn_{key_prefix}"):
            api_key = get_api_key("ANTHROPIC_API_KEY") or get_api_key("GEMINI_API_KEY")
            provider_type = "anthropic" if get_api_key("ANTHROPIC_API_KEY") else "gemini"
            if not api_key:
                st.error("No API key configured for enrichment.")
            elif not lead.get("id"):
                st.error("Cannot enrich unsaved lead.")
            else:
                with st.spinner(f"Enriching {lead.get('business_name')} via AI..."):
                    enriched_data, err = llm_client.enrich_single_lead(
                        provider_type,
                        api_key,
                        lead,
                        fetch_footer_contact_details=web_scraper.fetch_footer_contact_details,
                    )
                if err:
                    st.error(err)
                else:
                    db.save_enriched_lead(lead["id"], enriched_data)
                    st.success(f"Enriched {lead.get('business_name')}!")
                    st.rerun()

    if st.session_state.get(email_state_key):
        st.markdown("**Cold Email Sequence**:")
        st.text_area(
            "Generated Cold Email Sequence",
            value=st.session_state[email_state_key],
            height=240,
            key=f"txt_{email_state_key}",
            disabled=True,
            label_visibility="collapsed",
        )



# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Peninsula Ireland — Lead Sourcing CRM",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <meta name="description" content="Peninsula Ireland B2B Lead Sourcing CRM — Sourcing, verification, and 4-pass quality audit pipeline for HR, Employment Law, and Health & Safety compliance outreach in Ireland.">
    <style>
    /* Font & Icon Layout Stability (Eliminates CLS from Material Symbols & Source Sans VF) */
    @font-face {
        font-family: 'Material Symbols Rounded';
        font-display: block;
    }

    @font-face {
        font-family: 'Source Sans VF';
        font-display: swap;
    }

    /* Icon dimension reservation to prevent icon-loading shifts */
    .material-symbols-rounded, [data-testid="stIcon"], span[role="img"] {
        display: inline-block !important;
        width: 1.25em !important;
        height: 1.25em !important;
        contain: size layout !important;
    }

    /* Reduce styled-components recalculate style overhead via containment */
    div[data-testid="stVerticalBlock"] > div {
        contain: layout style;
    }

    /* App Containment & CLS Prevention */
    .stApp {
        contain: layout style;
    }

    /* Reserve min-height for tab panels to stabilize viewport and eliminate CLS (0.67 -> ~0.0) */
    div[data-baseweb="tab-panel"] {
        min-height: 520px;
        contain: layout style;
    }

    /* Pre-allocate tab navigation list height */
    div[data-baseweb="tab-list"] {
        min-height: 48px;
        contain: layout;
    }

    /* High-performance header rendering (eliminates 851ms LCP render delay) */
    h1, h2, h3 {
        contain: layout;
        content-visibility: auto;
        contain-intrinsic-size: 1px 38px;
        margin-bottom: 0.5rem !important;
    }

    /* Expander & container stability */
    .stExpander {
        contain: layout;
        min-height: 48px;
    }

    /* Custom lightweight card for Peninsula Pitch Hooks (zero Prism.js main-thread overhead) */
    .peninsula-pitch-card {
        background-color: rgba(28, 131, 225, 0.08);
        border-left: 4px solid #1c83e1;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 0.95rem;
        color: inherit;
        margin: 8px 0 12px 0;
        line-height: 1.5;
        contain: content;
    }

    .peninsula-pitch-title {
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #1c83e1;
        margin-bottom: 4px;
    }

    /* WCAG 2.1 AA Accessibility & Color Contrast Fixes (min 4.5:1 ratio) */
    /* Muted / Secondary / Caption text: upgraded from #83858c to #4a4c52 (6.5:1 ratio) */
    .stCaption, [data-testid="stCaptionContainer"], p.caption, .stMarkdown small {
        color: #4a4c52 !important;
    }

    /* Alert & Status Red text: upgraded from #ff4b4b to #d92626 (5.2:1 ratio) */
    .stAlert p, [data-baseweb="notification"] p, .st-emotion-cache-15okssx p {
        color: #d92626 !important;
    }

    /* Sidebar text contrast enhancement */
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {
        color: #2b2c30 !important;
    }

    /* Clear focus indicator ring for keyboard navigation */
    :focus-visible, button:focus-visible, input:focus-visible, select:focus-visible {
        outline: 2.5px solid #1c83e1 !important;
        outline-offset: 2px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

db.init_db()

if "last_run_leads" not in st.session_state:
    st.session_state.last_run_leads = []
if "last_run_meta" not in st.session_state:
    st.session_state.last_run_meta = {}
if "last_run_raw_text" not in st.session_state:
    st.session_state.last_run_raw_text = ""

st.title("🎯 Peninsula Ireland — B2B Lead Sourcing CRM")
st.caption(
    "Sourcing, verification, and 4-pass quality audit pipeline for HR / "
    "Employment Law / Health & Safety compliance outreach prospects."
)

with st.sidebar:
    st.header("Pipeline Controls")

    provider_label = st.selectbox("LLM Provider", list(llm_client.PROVIDERS.keys()), index=0)
    provider = llm_client.PROVIDERS[provider_label]
    provider_key_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "GEMINI_API_KEY"

    with st.expander("🔑 API Key & Fallback Pool Configuration", expanded=not bool(get_api_key(provider_key_name))):
        st.text_input(
            "Anthropic Primary Key",
            type="password",
            value=st.session_state.get("override_ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY") or ""),
            key="override_ANTHROPIC_API_KEY",
            help="Enter primary Anthropic key (or comma-separated list of keys)",
        )
        st.text_input(
            "Anthropic Fallback Key #2 (Optional)",
            type="password",
            value=st.session_state.get("override_ANTHROPIC_API_KEY_2", os.getenv("ANTHROPIC_API_KEY_2") or ""),
            key="override_ANTHROPIC_API_KEY_2",
            help="Backup key used automatically if Primary gets rate-limited",
        )
        st.text_input(
            "Gemini Primary Key",
            type="password",
            value=st.session_state.get("override_GEMINI_API_KEY", os.getenv("GEMINI_API_KEY") or ""),
            key="override_GEMINI_API_KEY",
            help="Enter primary Google AI Studio key (or comma-separated list of keys)",
        )
        st.text_input(
            "Google Maps / Places API Key",
            type="password",
            value=st.session_state.get("override_GOOGLE_MAPS_API_KEY", os.getenv("GOOGLE_MAPS_API_KEY") or ""),
            key="override_GOOGLE_MAPS_API_KEY",
            help="Enter Google Maps Platform API Key for Places API (New) search & enrichment",
        )

    target_county = st.selectbox("Target County", ["ALL"] + ROI_COUNTIES, index=0)
    target_industry = st.selectbox("Target Industry", ["ALL"] + TARGET_INDUSTRIES, index=0)

    stage_1_mode = st.radio(
        "Stage 1 Sourcing Mode",
        ["LIVE_SEARCH", "GOOGLE_PLACES_API", "VAULT_BATCH"],
        help=(
            "LIVE_SEARCH: uses live web search to locate companies. "
            "GOOGLE_PLACES_API: uses Google Places API (New) to discover real verified Irish businesses. "
            "VAULT_BATCH: ingests raw business data you paste below."
        ),
    )

    vault_raw_data = None
    if stage_1_mode == "VAULT_BATCH":
        vault_raw_data = st.text_area(
            "Raw Vault Data (paste CSV / JSON / directory scrape text)",
            height=180,
            placeholder="Paste raw business listings, a CSV dump, or scraped directory text here...",
        )

    enable_phase2 = False
    enable_cro_check = False
    if stage_1_mode == "LIVE_SEARCH":
        enable_cro_check = st.checkbox(
            "Verify active status via CRO register",
            value=False,
            help=(
                "Looks up each lead's business name in Ireland's CRO company "
                "register. Confirmed-dissolved/struck-off/liquidated companies "
                "are downgraded to Disqualified Archive. Requires CRO_API_EMAIL "
                "and CRO_API_KEY in .env (sign up at services.cro.ie) — an "
                "inconclusive lookup never disqualifies a lead."
            ),
        )
        enable_phase2 = st.checkbox(
            "Enable deep website verification (Phase 2)",
            value=False,
            help=(
                "For each lead with a website, scrapes its footer/contact area "
                "and issues a second Anthropic call to re-verify and re-score "
                "against that real page content. Adds latency and cost — one "
                "extra scrape + API call per lead with a website. Always uses "
                "Anthropic for this step regardless of the LLM Provider above."
            ),
        )

    enforce_firmographic_limits = st.checkbox(
        "Enforce firmographic limits (strict 5-50 staff rule)",
        value=False,
    )
    strict_decision_maker_gate = st.checkbox(
        "Strict decision-maker gate (Owner / MD / Partner / Founder only)",
        value=True,
    )

    min_acceptance_score = st.slider(
        "Minimum Acceptance Score",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        help="Acceptance threshold (0-100) for Active Queue classification.",
    )

    output_mode = st.radio(
        "Output Mode",
        ["DETAILED", "COMPACT_JSON"],
        help="DETAILED returns JSON + a markdown table. COMPACT_JSON returns the JSON array only.",
    )

    batch_limit = st.number_input(
        "Batch Limit (leads per run)",
        min_value=1,
        max_value=30,
        value=15,
        step=1,
        help="Caps how many leads are processed in one run to control token usage.",
    )

    st.divider()
    run_clicked = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    st.divider()
    st.metric("Leads in Database", db.get_lead_count())

if not get_api_key(provider_key_name):
    st.warning(
        f"⚠️ No `{provider_key_name}` detected. Add it to a `.env` file in this "
        "directory before running the pipeline."
    )

if run_clicked:
    if stage_1_mode == "GOOGLE_PLACES_API":
        maps_key = get_api_key("GOOGLE_MAPS_API_KEY")
        if not maps_key:
            st.error("No `GOOGLE_MAPS_API_KEY` detected. Please set it in `.env` or sidebar.")
        else:
            with st.spinner(f"Sourcing real {target_industry} listings in {target_county} via Google Places API (New)..."):
                search_query = f"{target_industry} companies" if target_industry != "ALL" else "Business services"
                results = google_places_client.search_google_places(
                    query=search_query,
                    api_key=maps_key,
                    county=target_county,
                    limit=int(batch_limit),
                )
                if results:
                    ingested_count = 0
                    for r in results:
                        lead_dict = {
                            "business_name": r["business_name"],
                            "county": r["county"],
                            "industry": target_industry if target_industry != "ALL" else "Unclassified",
                            "website": r["website"],
                            "contact_phone": r["contact_phone"],
                            "eircode": r["eircode"],
                            "audit_status": "Pending Approval",
                        }
                        res = db.insert_lead(lead_dict)
                        if res:
                            ingested_count += 1
                    st.success(f"🎉 Discovered & Ingested {ingested_count} verified lead(s) from Google Places API into Pending Vault!")
                    st.rerun()
                else:
                    st.warning("No listings returned from Google Places API for these criteria.")
    else:
        anthropic_keys = get_api_key_pool("ANTHROPIC_API_KEY")
        gemini_keys = get_api_key_pool("GEMINI_API_KEY")
        maps_key = get_api_key("GOOGLE_MAPS_API_KEY")

        if provider == "fluid" and not (anthropic_keys or gemini_keys or maps_key):
            st.error("No API keys detected for Autonomous Fluid Pipeline. Please set ANTHROPIC_API_KEY, GEMINI_API_KEY, or GOOGLE_MAPS_API_KEY in `.env` or sidebar.")
        elif provider != "fluid" and not get_api_key(provider_key_name):
            st.error(f"No {provider_key_name} found. Set it in a `.env` file or in Streamlit secrets.")
        else:
            with st.spinner(f"Sourcing {target_industry} leads in {target_county} via {provider_label}..."):
                raw_leads, raw_text, error = llm_client.run_pipeline(
                    provider,
                    anthropic_keys if provider == "fluid" else get_api_key(provider_key_name),
                    target_county,
                    target_industry,
                    stage_1_mode,
                    enforce_firmographic_limits,
                    strict_decision_maker_gate,
                    min_acceptance_score,
                    output_mode,
                    int(batch_limit),
                    vault_raw_data,
                    gemini_key=gemini_keys if provider == "fluid" else get_api_key("GEMINI_API_KEY"),
                    maps_key=maps_key,
                    google_places_search_fn=google_places_client.search_google_places,
                )

            if error:
                st.error(error)
                raw_leads = None
            else:
                st.session_state.last_run_raw_text = raw_text

            if raw_leads and enable_cro_check:
                cro_email = get_api_key("CRO_API_EMAIL")
                cro_key = get_api_key("CRO_API_KEY")
                if not cro_email or not cro_key:
                    st.warning(
                        "CRO verification requires CRO_API_EMAIL and CRO_API_KEY — "
                        "skipping registry check for this run."
                    )
                else:
                    with st.spinner(
                        f"Verifying active status for up to {len(raw_leads)} lead(s) via CRO..."
                    ):
                        raw_leads = llm_client.verify_leads_registry(
                            raw_leads, cro_email, cro_key, registry_client.check_cro_status
                        )

            if enable_phase2 and raw_leads:
                anthropic_key = get_api_key("ANTHROPIC_API_KEY")
                if not anthropic_key:
                    st.warning(
                        "Phase 2 website verification requires ANTHROPIC_API_KEY — "
                        "skipping deep verification for this run."
                    )
                else:
                    with st.spinner(
                        f"Phase 2: scraping and re-verifying up to {len(raw_leads)} lead site(s)..."
                    ):
                        raw_leads = llm_client.enrich_leads_phase2(
                            anthropic_key, raw_leads, web_scraper.fetch_footer_contact_details
                        )

            leads = [llm_client.flatten_lead(lead, target_industry) for lead in (raw_leads or [])]

            if not leads:
                st.warning("No leads could be parsed from the model's response.")
                with st.expander("Raw model output (for debugging)"):
                    st.code(raw_text or "(empty response)")
            else:
                for lead in leads:
                    lead["source_mode"] = stage_1_mode
                    lead["output_mode"] = output_mode
                    lead.setdefault("status", "New")

                inserted, skipped = db.insert_leads(leads)

                st.session_state.last_run_leads = leads
                st.session_state.last_run_meta = {
                    "county": target_county,
                    "industry": target_industry,
                    "mode": stage_1_mode,
                    "provider": provider_label,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "inserted": inserted,
                    "skipped": skipped,
                }
                st.success(
                    f"Processed {len(leads)} leads for {target_industry} in {target_county} — "
                    f"{inserted} new, {skipped} duplicate(s) skipped."
                )

def render_analytics_and_pipeline():
    st.header("📊 Pipeline Analytics & Sales Kanban")

    metrics = db.get_pipeline_metrics()

    # --------------------------------------------------------------------------
    # 1. Executive KPI Summary Cards
    # --------------------------------------------------------------------------
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.metric("Total Leads Sourced", metrics["total_leads"])
    with mcol2:
        st.metric("Active Pipeline", metrics["active_pipeline"])
    with mcol3:
        st.metric("Won Deals", metrics["won_leads"])
    with mcol4:
        st.metric("Conversion Rate", f"{metrics['conversion_rate']}%")
    with mcol5:
        st.metric("Avg Lead Score", metrics["avg_score"])

    st.divider()

    # --------------------------------------------------------------------------
    # 2. Executive Funnel & Analytics Charts
    # --------------------------------------------------------------------------
    with st.expander("📈 Analytics Charts & Yield Funnel", expanded=True):
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Sales Pipeline Funnel")
            funnel_data = {
                "Stage": STATUS_OPTIONS,
                "Lead Count": [metrics["status_counts"].get(s, 0) for s in STATUS_OPTIONS],
            }
            funnel_df = pd.DataFrame(funnel_data)
            st.bar_chart(funnel_df.set_index("Stage"), color="#0055ff")

        with chart_col2:
            st.subheader("Lead Quality Score Distribution")
            dist_data = {
                "Quality Bracket": list(metrics["score_distribution"].keys()),
                "Leads": list(metrics["score_distribution"].values()),
            }
            dist_df = pd.DataFrame(dist_data)
            st.bar_chart(dist_df.set_index("Quality Bracket"), color="#00aa66")

        breakdown_col1, breakdown_col2 = st.columns(2)

        with breakdown_col1:
            st.subheader("Leads by Target Industry")
            ind_counts = metrics["industry_counts"]
            if ind_counts:
                ind_df = pd.DataFrame({"Industry": list(ind_counts.keys()), "Count": list(ind_counts.values())})
                st.bar_chart(ind_df.set_index("Industry"))
            else:
                st.caption("No industry data available yet.")

        with breakdown_col2:
            st.subheader("Leads by County (Top 10)")
            county_counts = metrics["county_counts"]
            if county_counts:
                c_df = pd.DataFrame({"County": list(county_counts.keys()), "Count": list(county_counts.values())})
                c_df = c_df.sort_values(by="Count", ascending=False).head(10)
                st.bar_chart(c_df.set_index("County"))
            else:
                st.caption("No county data available yet.")

    st.divider()

    # --------------------------------------------------------------------------
    # 3. Interactive Pipeline Kanban Board
    # --------------------------------------------------------------------------
    st.subheader("📋 Interactive Kanban Board")
    st.caption("Filter and manage your leads visually across pipeline stages.")

    kb_col1, kb_col2, kb_col3 = st.columns(3)
    with kb_col1:
        kb_county = st.selectbox("Kanban Filter: County", ["All"] + ROI_COUNTIES, key="kb_county")
    with kb_col2:
        kb_industry = st.selectbox("Kanban Filter: Industry", ["All"] + TARGET_INDUSTRIES, key="kb_industry")
    with kb_col3:
        kb_min_score = st.number_input("Kanban Filter: Min Score", min_value=0, max_value=100, value=0, step=5, key="kb_min_score")

    all_leads_df = db.get_all_leads(
        county=kb_county,
        industry=kb_industry,
        min_score=kb_min_score if kb_min_score > 0 else None,
    )

    columns = st.columns(len(STATUS_OPTIONS))

    for idx, status in enumerate(STATUS_OPTIONS):
        with columns[idx]:
            status_icon = {"New": "🆕", "Contacted": "📩", "Qualified": "⭐", "Rejected": "🚫", "Won": "🎉"}.get(status, "📌")
            matching_leads = (
                all_leads_df[all_leads_df["status"] == status].to_dict("records")
                if not all_leads_df.empty
                else []
            )
            st.markdown(f"### {status_icon} {status} ({len(matching_leads)})")
            st.divider()

            if not matching_leads:
                st.caption("No leads in stage")
            elif len(matching_leads) > 15:
                st.caption(f"Showing top 15 of {len(matching_leads)} leads")

            for lead in matching_leads[:15]:
                lead_id = lead["id"]
                score = lead.get("overall_score")
                with st.container(border=True):
                    st.markdown(f"**{score_indicator(score)} {lead.get('business_name') or 'Unnamed'}**")
                    st.caption(f"📍 {lead.get('county') or '—'} | Score: {score if score is not None else '—'}")
                    if lead.get("decision_maker_name"):
                        st.caption(f"👤 {lead['decision_maker_name']}")

                    # Quick status transition buttons
                    btn_cols = st.columns(2)
                    if status == "New":
                        with btn_cols[0]:
                            if st.button("📩 Contact", key=f"move_cont_{lead_id}"):
                                db.update_lead_status(lead_id, "Contacted")
                                st.rerun()
                        with btn_cols[1]:
                            if st.button("❌ Reject", key=f"move_rej_{lead_id}"):
                                db.update_lead_status(lead_id, "Rejected")
                                st.rerun()

                    elif status == "Contacted":
                        with btn_cols[0]:
                            if st.button("⭐ Qualify", key=f"move_qual_{lead_id}"):
                                db.update_lead_status(lead_id, "Qualified")
                                st.rerun()
                        with btn_cols[1]:
                            if st.button("❌ Reject", key=f"move_rej_c_{lead_id}"):
                                db.update_lead_status(lead_id, "Rejected")
                                st.rerun()

                    elif status == "Qualified":
                        with btn_cols[0]:
                            if st.button("🎉 Win!", key=f"move_won_{lead_id}"):
                                db.update_lead_status(lead_id, "Won")
                                st.rerun()
                        with btn_cols[1]:
                            if st.button("❌ Reject", key=f"move_rej_q_{lead_id}"):
                                db.update_lead_status(lead_id, "Rejected")
                                st.rerun()

                    elif status == "Rejected":
                        with btn_cols[0]:
                            if st.button("🔄 Reopen", key=f"move_reopen_{lead_id}"):
                                db.update_lead_status(lead_id, "New")
                                st.rerun()

                    elif status == "Won":
                        with btn_cols[0]:
                            if st.button("⭐ Qualify", key=f"move_back_qual_{lead_id}"):
                                db.update_lead_status(lead_id, "Qualified")
                                st.rerun()

    st.divider()

    # --------------------------------------------------------------------------
    # 4. Bulk Operations Toolbar
    # --------------------------------------------------------------------------
    st.subheader("🛠️ Bulk Operations Toolbar")
    st.caption("Select multiple leads to perform batch status updates, email sequence generation, exports, or deletions.")

    if all_leads_df.empty:
        st.info("No leads available in database for bulk operations.")
        return

    lead_options = {
        row["id"]: f"ID {row['id']} — {row['business_name']} ({row['county']}, Score: {row.get('overall_score') or '—'}, Status: {row['status']})"
        for _, row in all_leads_df.iterrows()
    }

    selected_ids = st.multiselect(
        "Select Leads for Batch Action",
        options=list(lead_options.keys()),
        format_func=lambda x: lead_options[x],
        key="bulk_selected_ids",
    )

    if selected_ids:
        st.write(f"**{len(selected_ids)} lead(s) selected.**")

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)

        with bcol1:
            new_bulk_status = st.selectbox("Bulk New Status", STATUS_OPTIONS, key="bulk_status_choice")
            if st.button("🔄 Update Status for Selected"):
                count = db.bulk_update_status(selected_ids, new_bulk_status)
                st.success(f"Updated status to '{new_bulk_status}' for {count} lead(s).")
                st.rerun()

        with bcol2:
            selected_df = all_leads_df[all_leads_df["id"].isin(selected_ids)]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Export Selected (CSV)",
                data=selected_df.to_csv(index=False).encode("utf-8"),
                file_name=f"peninsula_bulk_selected_{timestamp}.csv",
                mime="text/csv",
                key="bulk_export_csv",
            )

        with bcol3:
            if st.button("✉️ Batch Generate Emails"):
                anthropic_key = get_api_key("ANTHROPIC_API_KEY")
                if not anthropic_key:
                    st.error("No ANTHROPIC_API_KEY found.")
                else:
                    sequences = []
                    progress_bar = st.progress(0)
                    for i, lid in enumerate(selected_ids):
                        lead_dict = db.get_lead_by_id(lid)
                        if lead_dict:
                            with st.spinner(f"Writing email for {lead_dict.get('business_name')}..."):
                                seq, err = llm_client.generate_cold_email_sequence(anthropic_key, lead_dict)
                                if not err:
                                    sequences.append(f"=== {lead_dict.get('business_name')} (Lead ID: {lid}) ===\n\n{seq}\n")
                        progress_bar.progress((i + 1) / len(selected_ids))
                    
                    batch_text = "\n\n".join(sequences)
                    st.session_state["bulk_email_output"] = batch_text
                    st.success(f"Generated email sequences for {len(sequences)} lead(s)!")

        with bcol4:
            confirm_del = st.checkbox("Confirm deletion", key="confirm_bulk_del")
            if st.button("🗑️ Delete Selected Leads", type="secondary"):
                if confirm_del:
                    del_count = db.bulk_delete_leads(selected_ids)
                    st.success(f"Deleted {del_count} lead(s).")
                    st.rerun()
                else:
                    st.warning("Please check 'Confirm deletion' before deleting.")

        if st.session_state.get("bulk_email_output"):
            st.subheader("Batch Cold Email Sequences")
            st.code(st.session_state["bulk_email_output"], language=None)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download Batch Emails (.txt)",
                data=st.session_state["bulk_email_output"].encode("utf-8"),
                file_name=f"peninsula_batch_emails_{timestamp}.txt",
                mime="text/plain",
                key="download_batch_emails",
            )


def render_enrichment_hub(provider_label: str):
    st.header("🔍 Automated AI Lead Enrichment Hub")
    st.caption(
        "Discover decision-makers, direct emails, phone numbers, websites, and "
        "compliance pitch hooks for imported or sparse company records using Smart Tiered Web Intelligence."
    )

    run_mode = st.radio(
        "Enrichment Execution Mode",
        ["Single Batch Run", "⚡ Continuous Rolling Auto-Batch Mode"],
        horizontal=True,
        key="enh_run_mode"
    )

    parallel_workers = st.slider("Parallel Worker Threads", min_value=1, max_value=20, value=10, step=1, help="Higher worker threads increase leads/minute speed by executing concurrent LLM calls across your API key pool.")

    if run_mode == "Single Batch Run":
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        with ecol1:
            e_county = st.selectbox("Target County", ["All"] + ROI_COUNTIES, key="enh_county")
        with ecol2:
            e_industry = st.selectbox("Target Industry", ["All"] + TARGET_INDUSTRIES, key="enh_industry")
        with ecol3:
            e_subset = st.selectbox("Target Lead Subset", ["Only Un-enriched Leads", "Low Score Leads (<70)", "All Leads"], key="enh_subset")
        with ecol4:
            e_batch_limit = st.number_input("Batch Size (leads per run)", min_value=1, max_value=100, value=25, step=5, key="enh_batch_limit")

        target_leads = db.get_unenriched_leads(
            county=e_county,
            industry=e_industry,
            subset_type=e_subset,
            limit=int(e_batch_limit),
        )

        st.markdown(f"**Found {len(target_leads)} lead(s) matching current enrichment criteria.**")

        if not target_leads:
            st.info("No leads matching selected criteria require enrichment.")
            return

        if st.button("🚀 Launch High-Capacity Single Batch", type="primary", use_container_width=True):
            anthropic_keys = get_api_key_pool("ANTHROPIC_API_KEY")
            gemini_keys = get_api_key_pool("GEMINI_API_KEY")

            if not anthropic_keys and not gemini_keys:
                st.error("No `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` detected. Please set at least one in `.env` or sidebar.")
                return

            progress_bar = st.progress(0.0)
            status_ticker = st.empty()
            metrics_container = st.container()

            status_ticker.markdown("🌐 **Step 1/2:** Multi-threaded parallel fetching of company website contacts...")
            
            from concurrent.futures import ThreadPoolExecutor
            import time

            start_time = time.time()

            def fetch_site(lead_record):
                w = lead_record.get("website")
                if w and isinstance(w, str) and w.strip() and w.strip().lower() not in ("nan", "none", "null"):
                    return lead_record["id"], web_scraper.fetch_footer_contact_details(w)
                return lead_record["id"], None

            scraped_map = {}
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_site, l) for l in target_leads]
                for fut in futures:
                    try:
                        lid, text = fut.result()
                        if text:
                            scraped_map[lid] = text
                    except Exception:
                        pass

            status_ticker.markdown(f"🤖 **Step 2/2:** Executing Parallel AI Lead Enrichment ({parallel_workers} concurrent threads)...")

            def custom_scraper(url):
                return web_scraper.fetch_footer_contact_details(url)

            batch_results = llm_client.enrich_leads_batch_concurrent(
                target_leads,
                anthropic_keys=anthropic_keys,
                gemini_keys=gemini_keys,
                max_workers=int(parallel_workers),
                fetch_footer_contact_details=custom_scraper,
            )

            to_save = []
            enriched_count = 0
            auto_approved_count = 0
            dm_found = 0
            email_found = 0
            phone_found = 0
            approved_leads_list = []
            errors = []

            for lead_rec, enriched_data, err in batch_results:
                if enriched_data:
                    to_save.append((lead_rec["id"], enriched_data))
                    enriched_count += 1
                    if enriched_data.get("decision_maker_name"):
                        dm_found += 1
                    if enriched_data.get("contact_email"):
                        email_found += 1
                    if enriched_data.get("contact_phone"):
                        phone_found += 1
                else:
                    errors.append(f"**{lead_rec.get('business_name')}**: {err or 'Unknown error'}")

            if to_save:
                db.save_enriched_leads_batch(to_save)
                for lid, _ in to_save:
                    saved_lead = db.get_lead_by_id(lid)
                    if saved_lead and saved_lead.get("audit_status") == "Verified (Active Queue)":
                        auto_approved_count += 1
                        approved_leads_list.append(saved_lead)

            elapsed = max(time.time() - start_time, 0.1)
            leads_per_min = round((enriched_count / elapsed) * 60)

            progress_bar.progress(1.0)
            status_ticker.success(f"🎉 Batch finished! Processed {len(target_leads)} leads in {round(elapsed, 1)}s (⚡ {leads_per_min} leads/min).")

            m1, m2, m3, m4, m5 = metrics_container.columns(5)
            with m1:
                st.metric("Total Enriched", enriched_count)
            with m2:
                st.metric("Auto-Approved (Score≥85)", auto_approved_count)
            with m3:
                st.metric("Decision Makers", f"{dm_found} ({round(dm_found/max(1, len(target_leads))*100)}%)")
            with m4:
                st.metric("Emails Sourced", f"{email_found} ({round(email_found/max(1, len(target_leads))*100)}%)")
            with m5:
                st.metric("Speedometer", f"⚡ {leads_per_min} leads/min")

            if approved_leads_list:
                import pandas as pd
                app_df = pd.DataFrame(approved_leads_list)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "⬇️ Export Auto-Approved Leads (CSV)",
                    data=app_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"peninsula_auto_approved_{timestamp}.csv",
                    mime="text/csv",
                    key="dl_auto_approved_single",
                )

            if errors:
                st.error(f"⚠️ {len(errors)} lead(s) encountered enrichment errors:\n" + "\n".join(f"- {e}" for e in errors[:10]))

        st.divider()
        st.subheader("📋 Inspect Enriched Leads & Approve")
        st.caption("Review newly enriched decision-maker details below, edit any fields if needed, and approve leads directly into the Active Sales Pipeline.")

        for lead_raw in target_leads:
            lead = db.get_lead_by_id(lead_raw["id"]) or lead_raw
            lid = lead["id"]
            biz_name = lead.get("business_name") or "Unnamed Business"
            county_str = lead.get("county") or "—"
            ind_str = lead.get("industry") or "Unclassified"
            dm_str = lead.get("decision_maker_name") or "Not Identified"
            score = lead.get("overall_score")
            score_str = f"Score: {score}" if score is not None else "Unscored"
            status_str = lead.get("audit_status") or "Pending Approval"

            card_title = f"🏢 {biz_name} ({county_str} | {ind_str}) — DM: {dm_str} ({score_str}) · {status_str}"
            with st.expander(card_title, expanded=False):
                st.markdown("#### ✏️ Review Details & Approve Lead")

                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    e_name = st.text_input("Business Name", value=lead.get("business_name") or "", key=f"eh_name_{lid}")
                    county_idx = ROI_COUNTIES.index(lead["county"]) if lead.get("county") in ROI_COUNTIES else 0
                    e_county = st.selectbox("County", ROI_COUNTIES, index=county_idx, key=f"eh_county_{lid}")
                    ind_list = TARGET_INDUSTRIES + ["Unclassified"]
                    ind_idx = ind_list.index(lead["industry"]) if lead.get("industry") in ind_list else len(ind_list) - 1
                    e_industry = st.selectbox("Industry", ind_list, index=ind_idx, key=f"eh_ind_{lid}")

                with ec2:
                    e_website = st.text_input("Website URL", value=lead.get("website") or "", key=f"eh_web_{lid}")
                    e_phone = st.text_input("Phone Number", value=lead.get("contact_phone") or "", key=f"eh_phone_{lid}")
                    e_email = st.text_input("Contact Email", value=lead.get("contact_email") or "", key=f"eh_email_{lid}")

                with ec3:
                    e_dm_name = st.text_input("Decision Maker Name", value=lead.get("decision_maker_name") or "", key=f"eh_dm_{lid}")
                    e_dm_title = st.text_input("Decision Maker Title", value=lead.get("decision_maker_title") or "", key=f"eh_title_{lid}")
                    e_hook = st.text_area("Peninsula Pitch Hook", value=lead.get("peninsula_pitch_hook") or "", key=f"eh_hook_{lid}", height=68)

                act_col1, act_col2, act_col3 = st.columns(3)
                with act_col1:
                    if st.button("💾 Save Manual Edits", key=f"eh_save_{lid}"):
                        db.update_lead(lid, {
                            "business_name": e_name,
                            "county": e_county,
                            "industry": e_industry,
                            "website": e_website,
                            "contact_phone": e_phone,
                            "contact_email": e_email,
                            "decision_maker_name": e_dm_name,
                            "decision_maker_title": e_dm_title,
                            "peninsula_pitch_hook": e_hook,
                        })
                        st.success(f"Saved changes for '{e_name}'!")
                        st.rerun()

                with act_col2:
                    if st.button("✅ Approve & Move to Active Pipeline", type="primary", key=f"eh_approve_{lid}"):
                        db.approve_lead(lid)
                        st.success(f"Approved '{biz_name}' and moved to Active Sales Pipeline!")
                        st.rerun()

                with act_col3:
                    if st.button("❌ Reject / Archive", key=f"eh_reject_{lid}"):
                        db.update_lead(lid, {"audit_status": "Disqualified Archive", "status": "Rejected"})
                        st.success(f"Archived '{biz_name}'.")
                        st.rerun()
    else:
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        with rcol1:
            e_county = st.selectbox("Target County", ["All"] + ROI_COUNTIES, key="enh_r_county")
        with rcol2:
            e_industry = st.selectbox("Target Industry", ["All"] + TARGET_INDUSTRIES, key="enh_r_industry")
        with rcol3:
            r_max_leads = st.number_input("Total Leads Goal to Enrich", min_value=10, max_value=1000, value=100, step=10, key="enh_r_goal")
        with rcol4:
            r_batch_size = st.number_input("Batch Size (leads/batch)", min_value=5, max_value=50, value=25, step=5, key="enh_r_batch")

        st.info(f"⚡ **Continuous Rolling Engine Configured:** Will enrich in automated rolling batches of **{r_batch_size} leads** ({parallel_workers} parallel workers) until reaching **{r_max_leads} leads** or completing all matching records.")

        if st.button("⚡ Launch Continuous Rolling Batch Engine", type="primary", use_container_width=True):
            anthropic_keys = get_api_key_pool("ANTHROPIC_API_KEY")
            gemini_keys = get_api_key_pool("GEMINI_API_KEY")

            if not anthropic_keys and not gemini_keys:
                st.error("No `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` detected. Please set at least one in `.env` or sidebar.")
                return

            total_goal = int(r_max_leads)
            batch_size = int(r_batch_size)

            overall_progress = st.progress(0.0)
            status_ticker = st.empty()
            metrics_container = st.container()

            total_enriched = 0
            auto_approved_count = 0
            dm_found = 0
            email_found = 0
            phone_found = 0
            batch_number = 0
            processed_ids = []
            approved_leads_list = []

            from concurrent.futures import ThreadPoolExecutor
            import time

            overall_start_time = time.time()

            while total_enriched < total_goal:
                current_limit = min(batch_size, total_goal - total_enriched)
                batch_leads = db.get_unenriched_leads(
                    county=e_county,
                    industry=e_industry,
                    subset_type="Only Un-enriched Leads",
                    limit=current_limit,
                    exclude_ids=processed_ids,
                )

                if not batch_leads:
                    status_ticker.success("🎉 All available leads matching criteria have been enriched!")
                    break

                batch_number += 1
                status_ticker.markdown(f"🔄 **Rolling Batch #{batch_number}:** Sourcing contact ground-truth & running {parallel_workers} parallel AI workers for {len(batch_leads)} lead(s)...")

                def fetch_site(l):
                    w = l.get("website")
                    has_website = bool(w and isinstance(w, str) and w.strip() and w.strip().lower() not in ("nan", "none", "null"))

                    # Phase 1: Discover missing website via Firecrawl search
                    if not has_website:
                        discovered = firecrawl_client.find_business_website(
                            l.get("business_name", ""), l.get("county", "")
                        )
                        if discovered:
                            l["website"] = discovered  # write back so LLM gets context
                            w = discovered
                            has_website = True

                    # Phase 2: Scrape the website (Firecrawl → BeautifulSoup fallback)
                    if has_website:
                        text = firecrawl_client.fetch_with_firecrawl(
                            w, fallback_scraper=web_scraper.fetch_footer_contact_details
                        )
                        return (l["id"], text or None)
                    return (l["id"], None)

                scraped_map = {}
                with ThreadPoolExecutor(max_workers=10) as executor:
                    for lid, text in executor.map(fetch_site, batch_leads):
                        if text:
                            scraped_map[lid] = text

                def custom_scraper(url):
                    # Return pre-scraped text keyed by lead id via closure lookup
                    return web_scraper.fetch_footer_contact_details(url)

                batch_results = llm_client.enrich_leads_batch_concurrent(
                    batch_leads,
                    anthropic_keys=anthropic_keys,
                    gemini_keys=gemini_keys,
                    max_workers=int(parallel_workers),
                    fetch_footer_contact_details=custom_scraper,
                )

                to_save = []
                failed_ids = []
                for lead_rec, enriched_data, err in batch_results:
                    processed_ids.append(lead_rec["id"])
                    if enriched_data:
                        to_save.append((lead_rec["id"], enriched_data))
                        total_enriched += 1
                        if enriched_data.get("decision_maker_name"):
                            dm_found += 1
                        if enriched_data.get("contact_email"):
                            email_found += 1
                        if enriched_data.get("contact_phone"):
                            phone_found += 1
                    else:
                        failed_ids.append(lead_rec["id"])

                if to_save:
                    db.save_enriched_leads_batch(to_save)
                    for lid, _ in to_save:
                        saved_lead = db.get_lead_by_id(lid)
                        if saved_lead and saved_lead.get("audit_status") == "Verified (Active Queue)":
                            auto_approved_count += 1
                            approved_leads_list.append(saved_lead)

                if failed_ids:
                    db.mark_leads_attempted_batch(failed_ids)

                overall_progress.progress(min(1.0, total_enriched / total_goal))

                if total_enriched < total_goal:
                    time.sleep(1.0)

            total_elapsed = max(time.time() - overall_start_time, 0.1)
            overall_speed = round((total_enriched / total_elapsed) * 60)

            status_ticker.success(f"🏁 Rolling Enrichment Finished! Processed {total_enriched} lead(s) across {batch_number} batch(es) in {round(total_elapsed, 1)}s (⚡ {overall_speed} leads/min).")

            m1, m2, m3, m4, m5 = metrics_container.columns(5)
            with m1:
                st.metric("Total Enriched", total_enriched)
            with m2:
                st.metric("Auto-Approved (Score≥85)", auto_approved_count)
            with m3:
                st.metric("Decision Makers", f"{dm_found} ({round(dm_found/max(1, total_enriched)*100)}%)")
            with m4:
                st.metric("Emails Sourced", f"{email_found} ({round(email_found/max(1, total_enriched)*100)}%)")
            with m5:
                st.metric("Phones Sourced", f"{phone_found} ({round(phone_found/max(1, total_enriched)*100)}%)")

            if approved_leads_list:
                import pandas as pd
                app_df = pd.DataFrame(approved_leads_list)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "⬇️ Export Auto-Approved Leads (CSV)",
                    data=app_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"peninsula_auto_approved_{timestamp}.csv",
                    mime="text/csv",
                    key="dl_auto_approved_rolling",
                )


def render_pending_approval_vault(provider_label: str):
    st.header("📥 Pending Approval Vault (Pre-Approval Lead Queue)")
    st.caption(
        "Search, inspect, manually edit, or AI-enrich raw imported company listings before approving them into the active sales pipeline."
    )

    scol1, scol2, scol3, scol4 = st.columns([2, 1, 1, 1])
    with scol1:
        p_search = st.text_input("🔍 Search Pending Leads (Name, Eircode, CRO #)", key="p_search")
    with scol2:
        p_county = st.selectbox("Filter County", ["All"] + ROI_COUNTIES, key="p_county")
    with scol3:
        p_industry = st.selectbox("Filter Industry", ["All"] + TARGET_INDUSTRIES, key="p_industry")
    with scol4:
        p_page_size = st.selectbox("Per page", [25, 50, 100], index=0, key="p_page_size")

    if "p_page_num" not in st.session_state:
        st.session_state.p_page_num = 0

    page_num = st.session_state.p_page_num
    offset = page_num * p_page_size

    pending_leads, total_pending = db.search_pending_leads(
        search_term=p_search,
        county=p_county,
        industry=p_industry,
        limit=p_page_size,
        offset=offset,
    )

    total_pages = max(1, (total_pending + p_page_size - 1) // p_page_size)

    # Page Navigation Bar
    pcol1, pcol2, pcol3, pcol4 = st.columns([2, 2, 1, 1])
    with pcol1:
        st.markdown(f"**Found {total_pending:,} Lead(s) Pending Approval**")
    with pcol2:
        st.caption(f"Page {page_num + 1} of {total_pages:,} (Records {offset + 1} – {min(offset + p_page_size, total_pending)})")
    with pcol3:
        if st.button("⬅️ Previous", disabled=(page_num == 0), key="p_prev_btn"):
            st.session_state.p_page_num = max(0, page_num - 1)
            st.rerun()
    with pcol4:
        if st.button("Next ➡️", disabled=(page_num >= total_pages - 1), key="p_next_btn"):
            st.session_state.p_page_num = min(total_pages - 1, page_num + 1)
            st.rerun()

    if not pending_leads:
        st.info("No pending leads match your search criteria.")
        return

    st.divider()

    # Individual Pending Lead Editing Cards
    for lead in pending_leads:
        lid = lead["id"]
        biz_name = lead.get("business_name") or "Unnamed Business"
        county_str = lead.get("county") or "—"
        ind_str = lead.get("industry") or "Unclassified"
        dm_str = lead.get("decision_maker_name") or "Not Identified"

        card_title = f"🏢 {biz_name} ({county_str} | {ind_str}) — DM: {dm_str}"
        with st.expander(card_title, expanded=False):
            st.markdown("#### ✏️ Edit Lead Details & Approve")

            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                e_name = st.text_input("Business Name", value=lead.get("business_name") or "", key=f"pe_name_{lid}")
                county_idx = ROI_COUNTIES.index(lead["county"]) if lead.get("county") in ROI_COUNTIES else 0
                e_county = st.selectbox("County", ROI_COUNTIES, index=county_idx, key=f"pe_county_{lid}")
                ind_list = TARGET_INDUSTRIES + ["Unclassified"]
                ind_idx = ind_list.index(lead["industry"]) if lead.get("industry") in ind_list else len(ind_list) - 1
                e_industry = st.selectbox("Industry", ind_list, index=ind_idx, key=f"pe_ind_{lid}")

            with ec2:
                e_website = st.text_input("Website URL", value=lead.get("website") or "", key=f"pe_web_{lid}")
                e_phone = st.text_input("Phone Number", value=lead.get("contact_phone") or "", key=f"pe_phone_{lid}")
                e_email = st.text_input("Contact Email", value=lead.get("contact_email") or "", key=f"pe_email_{lid}")

            with ec3:
                e_dm_name = st.text_input("Decision Maker Name", value=lead.get("decision_maker_name") or "", key=f"pe_dm_{lid}")
                e_dm_title = st.text_input("Decision Maker Title", value=lead.get("decision_maker_title") or "", key=f"pe_title_{lid}")
                e_hook = st.text_area("Peninsula Pitch Hook", value=lead.get("peninsula_pitch_hook") or "", key=f"pe_hook_{lid}", height=68)

            act_col1, act_col2, act_col3, act_col4 = st.columns(4)
            with act_col1:
                if st.button("💾 Save Manual Edits", key=f"save_edit_{lid}"):
                    db.update_lead(lid, {
                        "business_name": e_name,
                        "county": e_county,
                        "industry": e_industry,
                        "website": e_website,
                        "contact_phone": e_phone,
                        "contact_email": e_email,
                        "decision_maker_name": e_dm_name,
                        "decision_maker_title": e_dm_title,
                        "peninsula_pitch_hook": e_hook,
                    })
                    st.success(f"Saved changes for '{e_name}'!")
                    st.rerun()

            with act_col2:
                if st.button("🚀 Enrich via AI & Approve", type="primary", key=f"ai_enrich_{lid}"):
                    provider_type = llm_client.PROVIDERS.get(provider_label, "anthropic")
                    key_name = "ANTHROPIC_API_KEY" if provider_type == "anthropic" else "GEMINI_API_KEY"
                    api_key = get_api_key(key_name)
                    if not api_key:
                        st.error(f"No {key_name} key found.")
                    else:
                        with st.spinner(f"Enriching {biz_name}..."):
                            enriched, err = llm_client.enrich_single_lead(
                                provider_type,
                                api_key,
                                lead,
                                fetch_footer_contact_details=web_scraper.fetch_footer_contact_details,
                            )
                            if enriched:
                                db.save_enriched_lead(lid, enriched)
                                st.success(f"Approved and enriched '{biz_name}'!")
                                st.rerun()
                            else:
                                st.error(f"Enrichment error: {err}")

            with act_col3:
                if st.button("✅ Mark Approved Manually", key=f"man_approve_{lid}"):
                    db.approve_lead(lid)
                    st.success(f"Approved '{biz_name}' and moved to Active Sales Pipeline!")
                    st.rerun()

            with act_col4:
                if st.button("❌ Reject / Archive", key=f"disqualify_{lid}"):
                    db.update_lead(lid, {"audit_status": "Disqualified Archive", "status": "Rejected"})
                    st.success(f"Archived '{biz_name}'.")
                    st.rerun()


tab_pending, tab_enrichment, tab_analytics, tab_latest, tab_database = st.tabs(
    ["📥 Pending Approval Vault", "🔍 AI Lead Enrichment Hub", "📊 Approved Sales Pipeline & Analytics", "📋 Latest Run", "🗂️ Lead Database"]
)

with tab_pending:
    render_pending_approval_vault(provider_label)

with tab_enrichment:
    render_enrichment_hub(provider_label)

with tab_analytics:
    render_analytics_and_pipeline()



with tab_latest:
    leads = st.session_state.last_run_leads
    meta = st.session_state.last_run_meta

    if not leads:
        st.info("Configure the pipeline in the sidebar and click **Run Pipeline** to source leads.")
    else:
        st.caption(
            f"Last run: {meta.get('timestamp', '')} — {meta.get('industry')} in "
            f"{meta.get('county')} ({meta.get('mode')}, {meta.get('provider', 'Claude Sonnet 5')}) · "
            f"{meta.get('inserted', 0)} new / {meta.get('skipped', 0)} duplicate"
        )

        table_df = pd.DataFrame(leads)
        display_cols = [
            c
            for c in [
                "business_name",
                "county",
                "industry",
                "overall_score",
                "decision_maker_name",
                "decision_maker_title",
                "audit_status",
                "outreach_type",
            ]
            if c in table_df.columns
        ]
        st.dataframe(
            table_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "overall_score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%d"
                ),
                "business_name": "Business Name",
                "decision_maker_name": "Decision Maker",
                "decision_maker_title": "Title",
                "audit_status": "Status",
            },
        )

        st.subheader("Lead Detail Cards")
        for i, lead in enumerate(leads):
            score = lead.get("overall_score")
            title = (
                f"{score_indicator(score)} {lead.get('business_name') or 'Unknown Business'} "
                f"— {lead.get('decision_maker_name') or 'Decision maker not identified'} "
                f"({score if score is not None else '—'}) · {lead.get('audit_status') or 'Unclassified'}"
            )
            with st.expander(title):
                render_lead_card(lead, key_prefix=f"latest_{lead.get('lead_id') or i}")

        if st.session_state.last_run_raw_text:
            with st.expander("View full raw model output (JSON + markdown table)"):
                st.code(st.session_state.last_run_raw_text, language=None)

with tab_database:
    st.subheader("Saved Leads")

    # Row 1: dropdown filters
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
    with fcol1:
        f_county = st.selectbox("Filter: County", ["All"] + ROI_COUNTIES, key="filter_county")
    with fcol2:
        f_industry = st.selectbox("Filter: Industry", ["All"] + TARGET_INDUSTRIES, key="filter_industry")
    with fcol3:
        f_audit_status = st.selectbox("Filter: Audit Status", ["All"] + AUDIT_STATUS_OPTIONS, key="filter_audit_status")
    with fcol4:
        f_status = st.selectbox("Filter: CRM Status", ["All"] + STATUS_OPTIONS, key="filter_status")
    with fcol5:
        f_min_score = st.number_input("Filter: Min Score", min_value=0, max_value=100, value=0, step=5, key="filter_score")

    # Row 2: name search
    scol1, scol2 = st.columns([4, 1])
    with scol1:
        f_search_name = st.text_input(
            "🔍 Search by Business Name",
            placeholder="Type a company name to search...",
            key="filter_name_search",
        )
    with scol2:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        if st.button("✕ Clear", key="clear_name_search", use_container_width=True):
            st.session_state["filter_name_search"] = ""
            st.rerun()

    df = db.get_all_leads(
        county=f_county,
        industry=f_industry,
        status=f_status,
        audit_status=f_audit_status,
        min_score=f_min_score if f_min_score > 0 else None,
        search_name=f_search_name if f_search_name and f_search_name.strip() else None,
    )
    match_label = f"**{len(df)}** lead(s) match current filters"
    if f_search_name and f_search_name.strip():
        match_label += f' — searching for **"{f_search_name.strip()}"**'
    st.caption(match_label)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"peninsula_leads_export_{timestamp}.csv",
        mime="text/csv",
    )

    if df.empty:
        st.info("No leads saved yet — run the pipeline to populate the database.")
    else:
        editor_cols = [
            "id",
            "business_name",
            "county",
            "industry",
            "overall_score",
            "decision_maker_name",
            "audit_status",
            "status",
            "created_at",
        ]
        editor_df = df[editor_cols].copy()

        EDITABLE_COLUMNS = ["business_name", "decision_maker_name", "audit_status", "status"]

        edited_df = st.data_editor(
            editor_df,
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "business_name": st.column_config.TextColumn("Business Name"),
                "county": st.column_config.TextColumn("County", disabled=True),
                "industry": st.column_config.TextColumn("Industry", disabled=True),
                "overall_score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%d"
                ),
                "decision_maker_name": st.column_config.TextColumn("Decision Maker"),
                "audit_status": st.column_config.TextColumn("Audit Status"),
                "status": st.column_config.SelectboxColumn("CRM Status", options=STATUS_OPTIONS),
                "created_at": st.column_config.TextColumn("Added", disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="db_editor",
        )

        if st.button("💾 Save changes"):
            changed = 0
            original_rows = editor_df.set_index("id")[EDITABLE_COLUMNS].to_dict("index")
            for _, row in edited_df.iterrows():
                row_id = int(row["id"])
                original = original_rows.get(row_id, {})
                diff = {
                    col: row[col]
                    for col in EDITABLE_COLUMNS
                    if original.get(col) != row[col]
                }
                if diff:
                    db.update_lead(row_id, diff)
                    changed += 1
            st.success(f"Updated {changed} lead(s).")
            st.rerun()

        st.subheader("Lead Detail Cards")
        for _, row in df.iterrows():
            lead = row.to_dict()
            score = lead.get("overall_score")
            title = (
                f"{score_indicator(score)} {lead.get('business_name') or 'Unknown Business'} "
                f"— {lead.get('decision_maker_name') or 'Decision maker not identified'} "
                f"({int(score) if score is not None else '—'}) · {lead.get('audit_status') or 'Unclassified'} · {lead.get('status')}"
            )
            with st.expander(title):
                render_lead_card(lead, key_prefix=f"db_{lead['id']}")
