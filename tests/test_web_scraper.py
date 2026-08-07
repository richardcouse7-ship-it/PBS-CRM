from unittest.mock import MagicMock, patch

import httpx

import web_scraper


def _fake_response(html):
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_footer_contact_details_extracts_email_and_eircode():
    html = """
    <html><body>
    <footer>
    <p>Contact us: info@acme.ie</p>
    <p>Registered office Eircode: H91 XXXX</p>
    </footer>
    </body></html>
    """
    with patch("web_scraper.httpx.get", return_value=_fake_response(html)):
        result = web_scraper.fetch_footer_contact_details("https://acme.ie")
    assert result is not None
    assert "info@acme.ie" in result
    assert "H91" in result


def test_fetch_footer_contact_details_rejects_invalid_urls_without_network_call():
    for bad in (None, "", "nan", "none", "null"):
        assert web_scraper.fetch_footer_contact_details(bad) is None


def test_fetch_footer_contact_details_returns_none_on_http_error():
    with patch("web_scraper.httpx.get", side_effect=httpx.ConnectError("connection failed")):
        assert web_scraper.fetch_footer_contact_details("https://doesnotexist.ie") is None


def test_fetch_footer_contact_details_returns_none_when_no_contact_signal_found():
    html = "<html><body><p>Just some generic marketing copy with nothing useful.</p></body></html>"
    with patch("web_scraper.httpx.get", return_value=_fake_response(html)):
        assert web_scraper.fetch_footer_contact_details("https://acme.ie") is None


def test_fetch_footer_contact_details_excludes_asset_false_positive_emails():
    html = '<html><body><footer><img src="logo@2x.png"><p>info@acme.ie</p></footer></body></html>'
    with patch("web_scraper.httpx.get", return_value=_fake_response(html)):
        result = web_scraper.fetch_footer_contact_details("https://acme.ie")
    assert result is not None
    assert "logo@2x.png" not in result
    assert "info@acme.ie" in result
