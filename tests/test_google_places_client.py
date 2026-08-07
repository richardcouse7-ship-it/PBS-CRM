from unittest.mock import MagicMock, patch

import google_places_client as gpc


def test_search_google_places_returns_empty_without_api_key():
    assert gpc.search_google_places("accountants", "") == []
    assert gpc.search_google_places("accountants", None) == []


def test_search_google_places_parses_business_and_address_fields():
    payload = {
        "places": [
            {
                "displayName": {"text": "Acme Accountants"},
                "formattedAddress": "1 Main St, Cork",
                "nationalPhoneNumber": "021 1234567",
                "websiteUri": "https://acme.ie",
                "id": "place123",
                "rating": 4.5,
                "userRatingCount": 12,
                "addressComponents": [
                    {"types": ["postal_code"], "longText": "T12 ABCD"},
                    {"types": ["administrative_area_level_1"], "longText": "County Cork"},
                ],
            }
        ]
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload

    with patch("google_places_client.httpx.post", return_value=resp):
        results = gpc.search_google_places("accountants", "fake-key", county="Cork")

    assert len(results) == 1
    result = results[0]
    assert result["business_name"] == "Acme Accountants"
    assert result["eircode"] == "T12 ABCD"
    assert result["county"] == "Cork"
    assert result["website"] == "https://acme.ie"
    assert result["google_place_id"] == "place123"


def test_search_google_places_defaults_county_when_no_postal_component():
    payload = {"places": [{"displayName": {"text": "Acme"}, "addressComponents": []}]}
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload

    with patch("google_places_client.httpx.post", return_value=resp):
        results = gpc.search_google_places("accountants", "fake-key", county="ALL")

    assert results[0]["county"] == "Dublin"  # documented fallback default


def test_search_google_places_returns_empty_list_on_request_error():
    with patch("google_places_client.httpx.post", side_effect=Exception("network boom")):
        assert gpc.search_google_places("accountants", "fake-key") == []
