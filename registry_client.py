"""
Irish company register (CRO) verification for the Peninsula Ireland B2B
Lead Sourcing CRM.

Confirms a business's active status against the CRO Open Services API
(services.cro.ie) so the pipeline can exclude dissolved/struck-off/
liquidated entities from the audited lead set. No Streamlit dependency;
functions return results or None, never raise.

VERIFICATION NOTE: services.cro.ie's own API docs page (cws/help) returns
403 to automated fetches, so the request shape below (base URL, endpoint
path, Basic-auth-with-email+key scheme) is reconstructed from independent
third-party references, not CRO's own documentation, and the exact response
JSON field names for company status are NOT confirmed. Rather than assume
one exact key name that might be wrong, _find_status_in_result() scans a
result's string values for known CRO status vocabulary ("Normal",
"Dissolved", "Liquidation", "Receiver", "Struck Off", ...). Once a real
CRO_API_EMAIL / CRO_API_KEY is available, run a live lookup and confirm the
actual response shape — adjust the key lists below if needed.
"""

import re
from difflib import SequenceMatcher

import httpx

CRO_BASE_URL = "https://services.cro.ie/cws/"
REQUEST_TIMEOUT = 8.0

# Standard CRO status vocabulary (Irish company register terminology).
INACTIVE_KEYWORDS = ("dissolved", "struck off", "liquidat", "receiv", "wound up", "revoked")
ACTIVE_KEYWORDS = ("normal",)

# Below this fuzzy name-match ratio, don't trust the result belongs to the
# business we're checking — an inconclusive lookup is not evidence of
# anything and should never disqualify a lead.
NAME_MATCH_THRESHOLD = 0.72

NAME_FIELD_CANDIDATES = ("company_name", "companyName", "name")
NUMBER_FIELD_CANDIDATES = ("company_num", "companyNumber", "cro_number", "number")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _first_string_field(result: dict, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _find_status_in_result(result: dict) -> str | None:
    """Best-effort scan of a result object's string values for known CRO
    status vocabulary, since the exact field name is unconfirmed."""
    for value in result.values():
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        if any(kw in lowered for kw in INACTIVE_KEYWORDS) or any(
            kw in lowered for kw in ACTIVE_KEYWORDS
        ):
            return value
    return None


def check_cro_status(
    business_name: str | None,
    county: str | None,
    email: str | None,
    api_key: str | None,
    timeout: float = REQUEST_TIMEOUT,
) -> dict | None:
    """
    Look up a business by name in the CRO register.

    Returns {"cro_number": str | None, "status": str, "active": bool,
    "matched_name": str, "match_confidence": float} for a confident single
    best match, or None if no confident match was found, credentials are
    missing, or the lookup failed for any reason. Never raises — a network
    error, malformed response, or no results all resolve to None so one bad
    lookup can't break a batch run or wrongly disqualify a lead.

    `county` is accepted for future address cross-checking but not currently
    used to filter results (the CRO search response's address field name is
    unconfirmed — see module docstring).
    """
    if not business_name or not email or not api_key:
        return None

    try:
        response = httpx.get(
            CRO_BASE_URL + "companies",
            params={"company_name": business_name, "searchType": 3, "format": "json"},
            auth=(email, api_key),
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        results = data.get("results") or data.get("companies") or []
    else:
        results = []

    if not isinstance(results, list) or not results:
        return None

    target = _normalize(business_name)
    best_match, best_score = None, 0.0
    for result in results:
        if not isinstance(result, dict):
            continue
        candidate_name = _first_string_field(result, NAME_FIELD_CANDIDATES)
        if not candidate_name:
            continue
        score = SequenceMatcher(None, target, _normalize(candidate_name)).ratio()
        if score > best_score:
            best_match, best_score = result, score

    if best_match is None or best_score < NAME_MATCH_THRESHOLD:
        return None

    status_text = _find_status_in_result(best_match)
    if status_text is None:
        return None

    status_lower = status_text.lower()
    is_active = any(kw in status_lower for kw in ACTIVE_KEYWORDS) and not any(
        kw in status_lower for kw in INACTIVE_KEYWORDS
    )

    return {
        "cro_number": _first_string_field(best_match, NUMBER_FIELD_CANDIDATES),
        "status": status_text,
        "active": is_active,
        "matched_name": _first_string_field(best_match, NAME_FIELD_CANDIDATES),
        "match_confidence": round(best_score, 2),
    }
