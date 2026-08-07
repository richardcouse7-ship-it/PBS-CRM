"""
SQLite persistence layer for the Peninsula Ireland B2B Lead Sourcing CRM.

Schema mirrors the flattened shape of the Master System Prompt's lead
object: lead_id / business_name / decision_maker{} / contact_details{} /
audit_results{} / peninsula_pitch_hook. Nested objects are flattened into
columns at insert time (see app.py's flatten_lead()); the full original
JSON is preserved verbatim in raw_json.

All functions open a short-lived connection per call (safe for Streamlit's
rerun model, which may execute on different threads).
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

DB_PATH = "leads.db"

LEAD_COLUMNS = [
    "lead_id",
    "business_name",
    "county",
    "industry",
    "decision_maker_name",
    "decision_maker_title",
    "decision_maker_linkedin",
    "contact_email",
    "website",
    "contact_phone",
    "eircode",
    "overall_score",
    "audit_status",
    "directory_source",
    "outreach_type",
    "disqualification_reason",
    "peninsula_pitch_hook",
    "source_mode",
    "output_mode",
    "status",
    "scrape_verified",
    "cro_verified",
    "cro_number",
]


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the leads table if it does not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id TEXT,
                business_name TEXT NOT NULL,
                county TEXT,
                industry TEXT,
                decision_maker_name TEXT,
                decision_maker_title TEXT,
                decision_maker_linkedin TEXT,
                contact_email TEXT,
                website TEXT,
                contact_phone TEXT,
                eircode TEXT,
                overall_score INTEGER,
                audit_status TEXT,
                directory_source TEXT,
                outreach_type TEXT,
                disqualification_reason TEXT,
                peninsula_pitch_hook TEXT,
                source_mode TEXT,
                output_mode TEXT,
                status TEXT DEFAULT 'New',
                scrape_verified INTEGER DEFAULT 0,
                cro_verified INTEGER DEFAULT 0,
                cro_number TEXT,
                raw_json TEXT,
                dedupe_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(dedupe_key)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_county ON leads(county)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_industry ON leads(industry)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_audit_status ON leads(audit_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_created ON leads(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_score ON leads(overall_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_pending ON leads(audit_status, scrape_verified)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_biz_name ON leads(business_name)")


def insert_lead(lead: dict) -> int | None:
    """
    Insert a single flattened lead. Returns the new row id, or None if the
    lead was a duplicate (same business_name + county, case-insensitive)
    and was skipped.
    """
    now = datetime.utcnow().isoformat()

    values = {col: lead.get(col) for col in LEAD_COLUMNS}
    values["status"] = lead.get("status", "New")
    values["raw_json"] = json.dumps(lead.get("raw_json", lead), default=str)
    values["created_at"] = now
    values["updated_at"] = now
    # Dedupe on a normalized key rather than raw columns: SQLite's UNIQUE
    # constraint treats NULL as distinct from NULL, so relying on nullable
    # columns (e.g. contact_phone) directly would let every lead missing
    # that field bypass dedupe.
    business_name = (lead.get("business_name") or "").strip().lower()
    county = (lead.get("county") or "").strip().lower()
    values["dedupe_key"] = f"{business_name}|{county}"

    columns = list(values.keys())
    placeholders = ", ".join(f":{c}" for c in columns)
    column_list = ", ".join(columns)

    with get_connection() as conn:
        cur = conn.execute(
            f"INSERT OR IGNORE INTO leads ({column_list}) VALUES ({placeholders})",
            values,
        )
        return cur.lastrowid if cur.rowcount > 0 else None


def insert_leads(leads: list[dict]) -> tuple[int, int]:
    """Insert many leads. Returns (inserted_count, skipped_duplicate_count)."""
    inserted, skipped = 0, 0
    for lead in leads:
        result = insert_lead(lead)
        if result:
            inserted += 1
        else:
            skipped += 1
    return inserted, skipped


def update_lead_status(lead_id: int, status: str):
    with get_connection() as conn:
        conn.execute(
            "UPDATE leads SET status = ?, updated_at = ? WHERE id = ?",
            (status, datetime.utcnow().isoformat(), lead_id),
        )


def update_lead(lead_id: int, fields: dict):
    """Update arbitrary columns on a lead."""
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = :{k}" for k in fields)
    fields["id"] = lead_id
    with get_connection() as conn:
        conn.execute(f"UPDATE leads SET {set_clause} WHERE id = :id", fields)


def get_all_leads(
    county: str | None = None,
    industry: str | None = None,
    status: str | None = None,
    audit_status: str | None = None,
    min_score: int | None = None,
    search_name: str | None = None,
    limit: int | None = 500,
    offset: int = 0,
) -> pd.DataFrame:
    """Query leads with optional filters, newest first."""
    query = "SELECT * FROM leads WHERE 1=1"
    params: list = []

    if county and county != "All":
        query += " AND county = ?"
        params.append(county)
    if industry and industry != "All":
        query += " AND industry = ?"
        params.append(industry)
    if status and status != "All":
        query += " AND status = ?"
        params.append(status)
    if audit_status and audit_status != "All":
        query += " AND audit_status = ?"
        params.append(audit_status)
    if min_score is not None:
        query += " AND (overall_score IS NULL OR overall_score >= ?)"
        params.append(min_score)
    if search_name and search_name.strip():
        query += " AND LOWER(business_name) LIKE ?"
        params.append(f"%{search_name.strip().lower()}%")

    query += " ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def get_lead_by_id(lead_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
        return dict(row) if row else None


def get_lead_count() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM leads").fetchone()
        return row["c"]


def bulk_update_status(lead_ids: list[int], status: str) -> int:
    """Update status for multiple leads by ID list. Returns count updated."""
    if not lead_ids:
        return 0
    now = datetime.utcnow().isoformat()
    placeholders = ", ".join("?" for _ in lead_ids)
    params = [status, now] + list(lead_ids)
    with get_connection() as conn:
        cur = conn.execute(
            f"UPDATE leads SET status = ?, updated_at = ? WHERE id IN ({placeholders})",
            params,
        )
        return cur.rowcount


def bulk_delete_leads(lead_ids: list[int]) -> int:
    """Delete multiple leads by ID list. Returns count deleted."""
    if not lead_ids:
        return 0
    placeholders = ", ".join("?" for _ in lead_ids)
    with get_connection() as conn:
        cur = conn.execute(
            f"DELETE FROM leads WHERE id IN ({placeholders})",
            list(lead_ids),
        )
        return cur.rowcount


def get_pipeline_metrics() -> dict:
    """Return dictionary of aggregated metrics for dashboard visualizations using fast SQL aggregations."""
    with get_connection() as conn:
        total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        if total_leads == 0:
            return {
                "total_leads": 0,
                "active_pipeline": 0,
                "won_leads": 0,
                "conversion_rate": 0.0,
                "avg_score": 0.0,
                "status_counts": {},
                "audit_status_counts": {},
                "industry_counts": {},
                "county_counts": {},
                "score_distribution": {"High (90-100)": 0, "Medium (70-89)": 0, "Low (<70)": 0},
            }

        # Fast SQL Grouping
        status_rows = conn.execute("SELECT COALESCE(status, 'New'), COUNT(*) FROM leads GROUP BY status").fetchall()
        status_counts = {r[0]: r[1] for r in status_rows}

        audit_rows = conn.execute("SELECT COALESCE(audit_status, 'Unknown'), COUNT(*) FROM leads GROUP BY audit_status").fetchall()
        audit_status_counts = {r[0]: r[1] for r in audit_rows}

        ind_rows = conn.execute("SELECT COALESCE(industry, 'Unclassified'), COUNT(*) FROM leads GROUP BY industry").fetchall()
        industry_counts = {r[0]: r[1] for r in ind_rows}

        c_rows = conn.execute("SELECT COALESCE(county, 'Unknown'), COUNT(*) FROM leads GROUP BY county").fetchall()
        county_counts = {r[0]: r[1] for r in c_rows}

        avg_score_row = conn.execute("SELECT AVG(overall_score) FROM leads WHERE overall_score IS NOT NULL").fetchone()
        avg_score = round(avg_score_row[0], 1) if avg_score_row and avg_score_row[0] is not None else 0.0

        high_s = conn.execute("SELECT COUNT(*) FROM leads WHERE overall_score >= 90").fetchone()[0]
        med_s = conn.execute("SELECT COUNT(*) FROM leads WHERE overall_score >= 70 AND overall_score < 90").fetchone()[0]
        low_s = conn.execute("SELECT COUNT(*) FROM leads WHERE overall_score < 70 OR overall_score IS NULL").fetchone()[0]

    active_pipeline = status_counts.get("New", 0) + status_counts.get("Contacted", 0) + status_counts.get("Qualified", 0)
    won_leads = status_counts.get("Won", 0)
    conversion_rate = round((won_leads / total_leads * 100), 1) if total_leads > 0 else 0.0

    return {
        "total_leads": total_leads,
        "active_pipeline": active_pipeline,
        "won_leads": won_leads,
        "conversion_rate": conversion_rate,
        "avg_score": avg_score,
        "status_counts": status_counts,
        "audit_status_counts": audit_status_counts,
        "industry_counts": industry_counts,
        "county_counts": county_counts,
        "score_distribution": {
            "High (90-100)": int(high_s),
            "Medium (70-89)": int(med_s),
            "Low (<70)": int(low_s),
        },
    }


def get_unenriched_leads(
    county: str | None = None,
    industry: str | None = None,
    subset_type: str = "Only Un-enriched Leads",
    limit: int = 20,
    exclude_ids: list[int] | None = None,
) -> list[dict]:
    """
    Fetch leads requiring enrichment based on filter criteria.
    subset_type options: 'Only Un-enriched Leads', 'All Leads', 'Low Score Leads (<70)'
    """
    query = "SELECT * FROM leads WHERE (audit_status != 'Disqualified Archive' AND (status IS NULL OR status != 'Rejected'))"
    params: list = []

    if county and county != "All":
        query += " AND county = ?"
        params.append(county)
    if industry and industry != "All":
        query += " AND industry = ?"
        params.append(industry)

    if subset_type == "Only Un-enriched Leads":
        query += " AND (scrape_verified IS NULL OR scrape_verified = 0)"
    elif subset_type == "Low Score Leads (<70)":
        query += " AND (overall_score IS NULL OR overall_score < 70)"

    if exclude_ids:
        placeholders = ",".join("?" for _ in exclude_ids)
        query += f" AND id NOT IN ({placeholders})"
        params.extend(exclude_ids)

    query += " ORDER BY id ASC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df.to_dict("records")


def mark_leads_attempted_batch(lead_ids: list[int]) -> int:
    """Mark failed/attempted lead IDs as scrape_verified = 1 and Needs Attention so they don't block pipeline."""
    if not lead_ids:
        return 0
    sql = "UPDATE leads SET scrape_verified = 1, audit_status = COALESCE(audit_status, 'Needs Attention') WHERE id = ?"
    params = [(lid,) for lid in lead_ids]
    with get_connection() as conn:
        conn.executemany(sql, params)
    return len(lead_ids)


def save_enriched_lead(lead_id: int, enriched_fields: dict) -> None:
    """Update database record for lead_id with newly enriched values and mark as verified."""
    if not enriched_fields:
        return

    update_dict = {}
    allowed_keys = [
        "business_name",
        "county",
        "decision_maker_name",
        "decision_maker_title",
        "decision_maker_linkedin",
        "contact_email",
        "contact_phone",
        "website",
        "peninsula_pitch_hook",
        "overall_score",
        "audit_status",
        "industry",
        "scrape_verified",
        "source_mode",
    ]

    for key in allowed_keys:
        if key in enriched_fields and enriched_fields[key] is not None:
            val = enriched_fields[key]
            if key == "overall_score":
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = None
            update_dict[key] = val

    score = update_dict.get("overall_score")
    has_dm = bool(update_dict.get("decision_maker_name"))
    has_contact = bool(update_dict.get("contact_email") or update_dict.get("contact_phone"))

    if (score is not None and score >= 85) or (has_dm and has_contact):
        update_dict["audit_status"] = "Verified (Active Queue)"
        update_dict["scrape_verified"] = 1
    else:
        update_dict["audit_status"] = "Needs Attention"
        update_dict["scrape_verified"] = 0

    update_lead(lead_id, update_dict)


def save_enriched_leads_batch(enriched_records: list[tuple[int, dict]]) -> int:
    """
    High-throughput bulk transaction writer for enriched leads.
    `enriched_records` is a list of (lead_id, enriched_fields) tuples.
    Executes all updates in a single SQLite transaction (<10ms for 100 rows).
    Returns count of updated records.
    """
    if not enriched_records:
        return 0

    allowed_keys = {
        "business_name", "county", "industry", "website", "contact_phone",
        "contact_email", "decision_maker_name", "decision_maker_title",
        "peninsula_pitch_hook", "overall_score", "scrape_verified", "audit_status", "eircode"
    }

    params_list = []
    for lead_id, enriched_fields in enriched_records:
        update_dict = {}
        for key in allowed_keys:
            if key in enriched_fields and enriched_fields[key] is not None:
                val = enriched_fields[key]
                if key == "overall_score":
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        val = None
                update_dict[key] = val

        score = update_dict.get("overall_score")
        has_dm = bool(update_dict.get("decision_maker_name"))
        has_contact = bool(update_dict.get("contact_email") or update_dict.get("contact_phone"))

        if (score is not None and score >= 85) or (has_dm and has_contact):
            audit_status = "Verified (Active Queue)"
            scrape_verified = 1
        else:
            audit_status = "Needs Attention"
            scrape_verified = 0

        params_list.append((
            update_dict.get("business_name"),
            update_dict.get("county"),
            update_dict.get("industry"),
            update_dict.get("website"),
            update_dict.get("contact_phone"),
            update_dict.get("contact_email"),
            update_dict.get("decision_maker_name"),
            update_dict.get("decision_maker_title"),
            update_dict.get("peninsula_pitch_hook"),
            score,
            scrape_verified,
            audit_status,
            update_dict.get("eircode"),
            lead_id,
        ))

    sql = """
        UPDATE leads
        SET 
            business_name = COALESCE(?, business_name),
            county = COALESCE(?, county),
            industry = COALESCE(?, industry),
            website = COALESCE(?, website),
            contact_phone = COALESCE(?, contact_phone),
            contact_email = COALESCE(?, contact_email),
            decision_maker_name = COALESCE(?, decision_maker_name),
            decision_maker_title = COALESCE(?, decision_maker_title),
            peninsula_pitch_hook = COALESCE(?, peninsula_pitch_hook),
            overall_score = COALESCE(?, overall_score),
            scrape_verified = ?,
            audit_status = ?,
            eircode = COALESCE(?, eircode)
        WHERE id = ?
    """

    with get_connection() as conn:
        conn.executemany(sql, params_list)
    return len(params_list)


def approve_lead(lead_id: int) -> None:
    """Mark a lead as manually approved and move to Verified Active Queue."""
    lead = get_lead_by_id(lead_id)
    current_score = lead.get("overall_score") if lead else None
    update_lead(lead_id, {
        "audit_status": "Verified (Active Queue)",
        "scrape_verified": 1,
        "overall_score": 85 if current_score is None else current_score
    })


def search_pending_leads(
    search_term: str | None = None,
    county: str | None = None,
    industry: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Fetch leads in Pending Approval state with search and filter capabilities."""
    where_clause = "WHERE (audit_status = 'Pending Approval' OR (scrape_verified = 0 AND (decision_maker_name IS NULL OR decision_maker_name = '')))"
    params: list = []

    if search_term and search_term.strip():
        where_clause += " AND (business_name LIKE ? OR eircode LIKE ? OR cro_number LIKE ?)"
        st = f"%{search_term.strip()}%"
        params.extend([st, st, st])

    if county and county != "All":
        where_clause += " AND county = ?"
        params.append(county)

    if industry and industry != "All":
        where_clause += " AND industry = ?"
        params.append(industry)

    count_query = f"SELECT COUNT(*) FROM leads {where_clause}"
    query = f"SELECT * FROM leads {where_clause} ORDER BY id ASC LIMIT ? OFFSET ?"

    with get_connection() as conn:
        total_count = conn.execute(count_query, params).fetchone()[0]
        df = pd.read_sql_query(query, conn, params=params + [limit, offset])

    return df.to_dict("records"), total_count



