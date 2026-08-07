import pytest

import database as db


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Point database.py at a throwaway SQLite file for the duration of one test."""
    db_path = tmp_path / "test_leads.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    db.init_db()
    return db_path
