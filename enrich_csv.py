"""
Standalone CLI enrichment runner: takes a leads CSV export (same schema as
leads.db) and runs it through the same Smart Hybrid enrichment pipeline used
by app.py's Enrichment Hub (firecrawl_client website discovery/scrape ->
llm_client Anthropic/Gemini enrichment), then writes results back into
leads.db by `id` and emits an enriched CSV copy.

Usage:
    python enrich_csv.py "C:\\path\\to\\export.csv" [--limit N] [--workers N] [--db-only | --csv-only]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

import database as db
import firecrawl_client  # noqa: F401  (imported for parity with app.py; used indirectly by llm_client)
import llm_client
import web_scraper

load_dotenv(override=True)


def get_key_pool(env_var_name: str) -> list[str]:
    keys = []
    main = os.getenv(env_var_name)
    if main and main.strip():
        for k in main.split(","):
            if k.strip():
                keys.append(k.strip())
    for i in range(2, 10):
        alt = os.getenv(f"{env_var_name}_{i}")
        if alt and alt.strip() and alt.strip() not in keys:
            keys.append(alt.strip())
    return keys


def main():
    parser = argparse.ArgumentParser(description="Enrich a Peninsula CRM leads CSV via Firecrawl + LLM.")
    parser.add_argument("csv_path", help="Path to the leads export CSV")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent enrichment threads")
    parser.add_argument("--db-only", action="store_true", help="Only write results to leads.db, skip CSV output")
    parser.add_argument("--csv-only", action="store_true", help="Only write enriched CSV, skip leads.db writeback")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV not found at {args.csv_path}")
        sys.exit(1)

    anthropic_keys = get_key_pool("ANTHROPIC_API_KEY")
    gemini_keys = get_key_pool("GEMINI_API_KEY")
    if not anthropic_keys and not gemini_keys:
        print("Error: No ANTHROPIC_API_KEY or GEMINI_API_KEY found in environment/.env")
        sys.exit(1)
    if not firecrawl_client._get_firecrawl_key():
        print("Warning: No FIRECRAWL_API_KEY found — website discovery/scrape will fall back to BeautifulSoup only.")

    df = pd.read_csv(args.csv_path)
    if args.limit:
        df = df.head(args.limit)

    leads = df.to_dict("records")
    for lead in leads:
        lead["id"] = int(lead["id"]) if pd.notna(lead.get("id")) else None

    print(f"Loaded {len(leads)} lead(s) from {args.csv_path}")
    print(f"Running Smart Hybrid enrichment with {args.workers} workers "
          f"(Anthropic keys: {len(anthropic_keys)}, Gemini keys: {len(gemini_keys)})...")

    def custom_scraper(url):
        return web_scraper.fetch_footer_contact_details(url)

    start = time.time()
    llm_client.reset_usage_totals()
    results = llm_client.enrich_leads_batch_concurrent(
        leads,
        anthropic_keys=anthropic_keys,
        gemini_keys=gemini_keys,
        max_workers=args.workers,
        fetch_footer_contact_details=custom_scraper,
    )
    elapsed = time.time() - start

    enriched_rows = []
    to_save = []
    ok_count = 0
    err_count = 0

    for lead_rec, enriched_data, err in results:
        row = dict(lead_rec)
        if enriched_data:
            ok_count += 1
            for k, v in enriched_data.items():
                row[k] = v
            if lead_rec.get("id") is not None:
                to_save.append((lead_rec["id"], enriched_data))
        else:
            err_count += 1
            row["enrichment_error"] = err
        enriched_rows.append(row)

    print(f"Done in {elapsed:.1f}s — {ok_count} enriched, {err_count} failed.")

    usage = llm_client.get_usage_totals()
    if usage["input_tokens"] or usage["output_tokens"]:
        print(
            f"Anthropic usage: {usage['input_tokens']:,} input / {usage['output_tokens']:,} output tokens "
            f"— est. ${usage['estimated_cost_usd']:.4f}"
        )

    if not args.csv_only and to_save:
        updated = db.save_enriched_leads_batch(to_save)
        print(f"Wrote {updated} enriched record(s) back into leads.db")

    if not args.db_only:
        out_df = pd.DataFrame(enriched_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(args.csv_path))[0]
        out_path = os.path.join(os.path.dirname(args.csv_path), f"{base}_enriched_{timestamp}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Wrote enriched CSV to {out_path}")

    if err_count:
        print("\nFailures:")
        for lead_rec, enriched_data, err in results:
            if not enriched_data:
                print(f"  - {lead_rec.get('business_name')}: {err}")


if __name__ == "__main__":
    main()
