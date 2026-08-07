from unittest.mock import MagicMock, patch

import httpx

import registry_client as rc


def _fake_httpx_get(json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def test_check_cro_status_returns_none_without_credentials():
    assert rc.check_cro_status("Acme Ltd", "Cork", None, None) is None
    assert rc.check_cro_status(None, "Cork", "a@b.com", "key") is None
    assert rc.check_cro_status("Acme Ltd", "Cork", "a@b.com", None) is None


def test_check_cro_status_matches_active_company():
    data = [{"company_name": "Acme Limited", "company_num": "123456", "company_status": "Normal"}]
    with patch("registry_client.httpx.get", return_value=_fake_httpx_get(data)):
        result = rc.check_cro_status("Acme Limited", "Cork", "a@b.com", "key")
    assert result is not None
    assert result["active"] is True
    assert result["cro_number"] == "123456"


def test_check_cro_status_detects_dissolved_company_as_inactive():
    data = [{"company_name": "Acme Limited", "company_num": "123456", "company_status": "Dissolved"}]
    with patch("registry_client.httpx.get", return_value=_fake_httpx_get(data)):
        result = rc.check_cro_status("Acme Limited", "Cork", "a@b.com", "key")
    assert result is not None
    assert result["active"] is False


def test_check_cro_status_returns_none_when_no_confident_name_match():
    data = [{"company_name": "Totally Different Business", "company_num": "999", "company_status": "Normal"}]
    with patch("registry_client.httpx.get", return_value=_fake_httpx_get(data)):
        result = rc.check_cro_status("Acme Limited", "Cork", "a@b.com", "key")
    assert result is None


def test_check_cro_status_returns_none_when_status_vocabulary_unrecognized():
    data = [{"company_name": "Acme Limited", "company_num": "123456", "company_status": "Some Unknown Value"}]
    with patch("registry_client.httpx.get", return_value=_fake_httpx_get(data)):
        result = rc.check_cro_status("Acme Limited", "Cork", "a@b.com", "key")
    assert result is None


def test_check_cro_status_returns_none_on_network_error():
    with patch("registry_client.httpx.get", side_effect=httpx.ConnectError("connection failed")):
        assert rc.check_cro_status("Acme Limited", "Cork", "a@b.com", "key") is None


def test_check_cro_status_returns_none_on_empty_results():
    with patch("registry_client.httpx.get", return_value=_fake_httpx_get([])):
        assert rc.check_cro_status("Acme Limited", "Cork", "a@b.com", "key") is None


def test_normalize_strips_punctuation_and_lowercases():
    assert rc._normalize("Acme & Sons, Ltd.") == "acme  sons ltd"
