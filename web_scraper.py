"""
Website scraping helper for the Peninsula Ireland B2B Lead Sourcing CRM.

Fetches a lead's homepage and extracts footer/contact-area text (phone
numbers, Eircodes, address-like lines) so Phase 2 re-verification can be
grounded in real page content instead of search-snippet summaries alone.
"""

import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; PeninsulaLeadCRM/1.0; +https://peninsula.ie)"
MAX_RESPONSE_BYTES = 300_000
REQUEST_TIMEOUT = 8.0

PHONE_PATTERN = re.compile(r"(?:\+353\s?|0)(?:\d[\s-]?){8,10}\d")
EIRCODE_PATTERN = re.compile(r"\b[A-Za-z]\d{2}\s?[A-Za-z0-9]{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def fetch_footer_contact_details(url: str | None, timeout: float = REQUEST_TIMEOUT) -> str | None:
    """
    Fetch a business's homepage and extract contact details (email addresses,
    phone numbers, Eircodes, decision maker/address lines) from its <footer>
    and page contact areas.

    Never raises: missing/invalid URL, network errors return None gracefully.
    """
    if not url or not isinstance(url, str) or not url.strip() or url.strip().lower() in ("nan", "none", "null"):
        return None
    url = url.strip()

    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        return None
    if not parsed.scheme:
        parsed = urlparse(f"https://{url}")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None

    try:
        response = httpx.get(
            parsed.geturl(),
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    html = response.text[:MAX_RESPONSE_BYTES]

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return None

    # Extract all emails on the page (from text and mailto links)
    emails = list(dict.fromkeys(EMAIL_PATTERN.findall(html)))
    # Exclude common static asset false positives (like .png, .jpg, .svg)
    valid_emails = [
        e for e in emails 
        if not any(e.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".wixpress.com"])
    ]

    footer = (
        soup.find("footer")
        or soup.find(class_=re.compile("footer", re.IGNORECASE))
        or soup.find(id=re.compile("footer", re.IGNORECASE))
    )

    if footer is not None:
        region_text = footer.get_text(separator="\n", strip=True)
    else:
        body = soup.find("body") or soup
        full_text = body.get_text(separator="\n", strip=True)
        region_text = "\n".join(full_text.splitlines()[-50:])

    lines = [ln.strip() for ln in (region_text or "").splitlines() if ln.strip()]
    phones = list(dict.fromkeys(PHONE_PATTERN.findall(html)))
    eircodes = list(dict.fromkeys(EIRCODE_PATTERN.findall(html)))

    candidate_lines = [
        ln
        for ln in lines
        if len(ln) <= 200
        and (PHONE_PATTERN.search(ln) or EIRCODE_PATTERN.search(ln) or EMAIL_PATTERN.search(ln) or "director" in ln.lower() or "owner" in ln.lower() or "partner" in ln.lower() or "county" in ln.lower())
    ]

    if not phones and not eircodes and not valid_emails and not candidate_lines:
        return None

    parts = []
    if valid_emails:
        parts.append("Direct Emails found: " + ", ".join(valid_emails[:5]))
    if phones:
        parts.append("Phone numbers found: " + ", ".join(phones[:5]))
    if eircodes:
        parts.append("Eircodes found: " + ", ".join(eircodes[:3]))
    if candidate_lines:
        joined = " | ".join(dict.fromkeys(candidate_lines))[:1000]
        parts.append("Contact/address lines: " + joined)

    return "\n".join(parts)
