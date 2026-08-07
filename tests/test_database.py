import database as db


def test_insert_lead_dedupes_case_insensitively(temp_db):
    first_id = db.insert_lead({"business_name": "Acme Ltd", "county": "Cork"})
    assert first_id is not None

    dup_id = db.insert_lead({"business_name": "acme ltd", "county": "cork"})
    assert dup_id is None
    assert db.get_lead_count() == 1


def test_insert_leads_reports_inserted_and_skipped_counts(temp_db):
    leads = [
        {"business_name": "A", "county": "Cork"},
        {"business_name": "A", "county": "Cork"},  # duplicate of the row above
        {"business_name": "B", "county": "Dublin"},
    ]
    inserted, skipped = db.insert_leads(leads)
    assert inserted == 2
    assert skipped == 1


def test_update_lead_status_persists(temp_db):
    lid = db.insert_lead({"business_name": "A", "county": "Cork"})
    db.update_lead_status(lid, "Contacted")
    assert db.get_lead_by_id(lid)["status"] == "Contacted"


def test_update_lead_updates_arbitrary_columns(temp_db):
    lid = db.insert_lead({"business_name": "A", "county": "Cork"})
    db.update_lead(lid, {"business_name": "A Renamed", "audit_status": "Needs Attention"})
    lead = db.get_lead_by_id(lid)
    assert lead["business_name"] == "A Renamed"
    assert lead["audit_status"] == "Needs Attention"


def test_save_enriched_leads_batch_verifies_complete_leads(temp_db):
    lid = db.insert_lead({"business_name": "A", "county": "Cork"})
    db.save_enriched_leads_batch([(lid, {
        "decision_maker_name": "Jane",
        "contact_email": "jane@a.ie",
        "contact_phone": "091 123456",
        "overall_score": 90,
    })])
    lead = db.get_lead_by_id(lid)
    assert lead["audit_status"] == "Verified (Active Queue)"
    assert lead["scrape_verified"] == 1


def test_save_enriched_leads_batch_flags_incomplete_leads_as_needs_attention(temp_db):
    lid = db.insert_lead({"business_name": "A", "county": "Cork"})
    db.save_enriched_leads_batch([(lid, {"overall_score": 40})])
    lead = db.get_lead_by_id(lid)
    assert lead["audit_status"] == "Needs Attention"
    assert lead["scrape_verified"] == 0


def test_get_all_leads_filters_by_county(temp_db):
    db.insert_lead({"business_name": "A", "county": "Cork"})
    db.insert_lead({"business_name": "B", "county": "Dublin"})

    df = db.get_all_leads(county="Cork")
    assert len(df) == 1
    assert df.iloc[0]["business_name"] == "A"


def test_get_all_leads_search_name_is_case_insensitive_substring(temp_db):
    db.insert_lead({"business_name": "Acme Accountants", "county": "Cork"})
    db.insert_lead({"business_name": "Other Ltd", "county": "Cork"})

    df = db.get_all_leads(search_name="accountants")
    assert len(df) == 1
    assert df.iloc[0]["business_name"] == "Acme Accountants"


def test_bulk_update_status(temp_db):
    id1 = db.insert_lead({"business_name": "A", "county": "Cork"})
    id2 = db.insert_lead({"business_name": "B", "county": "Dublin"})

    updated = db.bulk_update_status([id1, id2], "Qualified")
    assert updated == 2
    assert db.get_lead_by_id(id1)["status"] == "Qualified"
    assert db.get_lead_by_id(id2)["status"] == "Qualified"


def test_bulk_delete_leads(temp_db):
    id1 = db.insert_lead({"business_name": "A", "county": "Cork"})
    id2 = db.insert_lead({"business_name": "B", "county": "Dublin"})

    deleted = db.bulk_delete_leads([id1, id2])
    assert deleted == 2
    assert db.get_lead_count() == 0


def test_approve_lead_defaults_score_when_missing(temp_db):
    lid = db.insert_lead({"business_name": "A", "county": "Cork"})
    db.approve_lead(lid)
    lead = db.get_lead_by_id(lid)
    assert lead["audit_status"] == "Verified (Active Queue)"
    assert lead["overall_score"] == 85


def test_approve_lead_preserves_existing_score(temp_db):
    lid = db.insert_lead({"business_name": "A", "county": "Cork", "overall_score": 72})
    db.approve_lead(lid)
    assert db.get_lead_by_id(lid)["overall_score"] == 72


def test_get_unenriched_leads_only_unverified_by_default(temp_db):
    verified_id = db.insert_lead({"business_name": "A", "county": "Cork", "audit_status": "Verified (Active Queue)"})
    db.update_lead(verified_id, {"scrape_verified": 1})
    db.insert_lead({"business_name": "B", "county": "Dublin", "audit_status": "Pending Approval"})

    unenriched = db.get_unenriched_leads(subset_type="Only Un-enriched Leads", limit=10)
    names = [lead["business_name"] for lead in unenriched]
    assert names == ["B"]
