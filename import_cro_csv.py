"""
Ultra-fast vectorized script to import active ('Normal') company records from the
CRO register CSV dump into Peninsula CRM (leads.db).
"""

import os
import sqlite3
import time
from datetime import datetime
import pandas as pd

CSV_PATH = r"C:\Users\richa\Downloads\3fef41bc-b8f4-4b10-8434-ce51c29b1bba.csv"
DB_PATH = os.path.join(os.path.dirname(__file__), "leads.db")

ROI_COUNTIES = [
    "Carlow", "Cavan", "Clare", "Cork", "Donegal", "Dublin", "Galway",
    "Kerry", "Kildare", "Kilkenny", "Laois", "Leitrim", "Limerick",
    "Longford", "Louth", "Mayo", "Meath", "Monaghan", "Offaly",
    "Roscommon", "Sligo", "Tipperary", "Waterford", "Westmeath",
    "Wexford", "Wicklow",
]

COUNTY_TUPLES = [(c, c.upper()) for c in ROI_COUNTIES]


def init_db(conn):
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


def run_import():
    print(f"Opening CSV file: {CSV_PATH}", flush=True)
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}", flush=True)
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    start_time = time.time()
    total_processed = 0
    total_active = 0
    total_inserted = 0

    chunk_size = 100000

    use_cols = [
        "company_num",
        "company_name",
        "company_status",
        "company_address_1",
        "company_address_2",
        "company_address_3",
        "company_address_4",
        "eircode",
    ]

    print("Beginning high-speed vectorized import...", flush=True)

    sql = """
    INSERT OR IGNORE INTO leads (
        lead_id, business_name, county, industry, eircode, overall_score,
        audit_status, directory_source, outreach_type, source_mode, status,
        cro_verified, cro_number, dedupe_key, created_at, updated_at
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    )
    """

    now = datetime.now().isoformat()

    for chunk_idx, chunk in enumerate(pd.read_csv(CSV_PATH, usecols=use_cols, chunksize=chunk_size, low_memory=False)):
        total_processed += len(chunk)

        # Filter for Normal (Active) status
        active_chunk = chunk[chunk["company_status"].astype(str).str.strip() == "Normal"]
        if active_chunk.empty:
            continue

        active_count = len(active_chunk)
        total_active += active_count

        names = active_chunk["company_name"].fillna("").astype(str).str.strip().values
        nums = active_chunk["company_num"].fillna("").astype(str).str.strip().values
        eircodes = active_chunk["eircode"].fillna("").astype(str).str.strip().values

        a1 = active_chunk["company_address_1"].fillna("").astype(str).values
        a2 = active_chunk["company_address_2"].fillna("").astype(str).values
        a3 = active_chunk["company_address_3"].fillna("").astype(str).values
        a4 = active_chunk["company_address_4"].fillna("").astype(str).values

        rows_to_insert = []
        for name, num, eircode, addr1, addr2, addr3, addr4 in zip(names, nums, eircodes, a1, a2, a3, a4):
            if not name:
                continue

            full_addr = f" {addr1} {addr2} {addr3} {addr4} ".upper()
            county = "Unknown"
            for c_name, c_upper in COUNTY_TUPLES:
                if c_upper in full_addr:
                    county = c_name
                    break

            lead_id = f"CRO-{num}" if num else None
            eircode_val = eircode if eircode else None
            dedupe_key = f"{name.lower()}|{county.lower()}"

            rows_to_insert.append((
                lead_id,
                name,
                county,
                "Unclassified",
                eircode_val,
                75,
                "Verified (Active Queue)",
                "CRO Register CSV",
                "Direct B2B",
                "CRO_IMPORT",
                "New",
                1,
                num,
                dedupe_key,
                now,
                now,
            ))

        with conn:
            cur = conn.executemany(sql, rows_to_insert)
            inserted = cur.rowcount
            total_inserted += inserted

        print(
            f"Chunk {chunk_idx + 1}: Scanned {len(chunk):,} | "
            f"Active: {active_count:,} | Inserted new: {inserted:,}",
            flush=True,
        )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60, flush=True)
    print("IMPORT COMPLETE!", flush=True)
    print(f"Total CSV rows scanned: {total_processed:,}", flush=True)
    print(f"Total Active ('Normal') companies found: {total_active:,}", flush=True)
    print(f"Total New Leads Inserted: {total_inserted:,}", flush=True)
    print(f"Duplicates Skipped: {total_active - total_inserted:,}", flush=True)
    print(f"Time Taken: {elapsed:.2f} seconds", flush=True)
    print("=" * 60, flush=True)

    conn.close()


if __name__ == "__main__":
    run_import()
