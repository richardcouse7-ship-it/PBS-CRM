"""
Google Places API (New) client for lead generation and enrichment in Peninsula CRM.
Uses https://places.googleapis.com/v1/places:searchText
"""

import httpx

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


def search_google_places(
    query: str,
    api_key: str,
    county: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Search real Irish business listings using Places API (New) Text Search.
    Returns list of standardized lead dictionaries.
    """
    if not api_key or not api_key.strip():
        return []

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key.strip(),
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.nationalPhoneNumber,places.websiteUri,places.rating,places.userRatingCount,places.googleMapsUri,places.addressComponents",
    }

    location_str = f"County {county}, Ireland" if county and county not in ("All", "ALL") else "Ireland"
    text_query = f"{query} in {location_str}"

    payload = {
        "textQuery": text_query,
        "languageCode": "en",
        "regionCode": "IE",
        "pageSize": min(limit, 20),
    }

    try:
        resp = httpx.post(PLACES_SEARCH_URL, json=payload, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        places = data.get("places", [])

        results = []
        for p in places:
            display_name = p.get("displayName", {}).get("text") or "Unknown Business"
            address = p.get("formattedAddress") or ""
            phone = p.get("nationalPhoneNumber") or ""
            website = p.get("websiteUri") or ""
            place_id = p.get("id") or ""

            eircode = ""
            county_found = county if county and county not in ("All", "ALL") else "Dublin"
            for comp in p.get("addressComponents", []):
                types = comp.get("types", [])
                if "postal_code" in types:
                    eircode = comp.get("longText") or ""
                if "administrative_area_level_1" in types or "administrative_area_level_2" in types:
                    c_name = comp.get("longText") or ""
                    if "County" in c_name:
                        county_found = c_name.replace("County", "").strip()

            results.append({
                "business_name": display_name,
                "address": address,
                "county": county_found,
                "eircode": eircode,
                "contact_phone": phone,
                "website": website,
                "google_place_id": place_id,
                "google_maps_url": p.get("googleMapsUri"),
                "rating": p.get("rating"),
                "reviews_count": p.get("userRatingCount"),
            })
        return results
    except Exception as e:
        print(f"Google Places API Search Error: {e}")
        return []


def enrich_lead_via_google_places(
    business_name: str,
    county: str | None,
    api_key: str,
) -> dict | None:
    """
    Look up a single business on Google Places API (New) to enrich phone, website, eircode, and address.
    """
    if not business_name or not api_key:
        return None
    results = search_google_places(business_name, api_key, county=county, limit=1)
    return results[0] if results else None
