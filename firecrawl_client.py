from __future__ import annotations
import os, logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

def _get_firecrawl_key():
    key = os.environ.get('FIRECRAWL_API_KEY', '').strip()
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('FIRECRAWL_API_KEY='):
                    val = line.split('=', 1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith('fc-your'):
                        return val
    return None

def fetch_with_firecrawl(url, fallback_scraper=None):
    if not url or not isinstance(url, str):
        return ''
    url = url.strip()
    if url.lower() in ('nan', 'none', 'null', 'n/a', ''):
        return ''
    api_key = _get_firecrawl_key()
    if api_key:
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=api_key)
            result = app.scrape_url(url, formats=['markdown'], only_main_content=True, timeout=8000)
            markdown = (result or {}).get('markdown', '') or ''
            if markdown.strip():
                return markdown[:5000]
        except Exception as e:
            logger.debug('Firecrawl failed for %s: %s', url, e)
    if fallback_scraper:
        try:
            return fallback_scraper(url) or ''
        except Exception:
            pass
    return ''

def bulk_scrape_urls(urls, max_workers=10, fallback_scraper=None):
    if not urls:
        return {}
    def _one(url):
        return url, fetch_with_firecrawl(url, fallback_scraper=fallback_scraper)
    results = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as ex:
        for url, text in ex.map(_one, urls):
            results[url] = text
    return results

def search_firecrawl(query, limit=5):
    api_key = _get_firecrawl_key()
    if not api_key:
        return []
    try:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=api_key)
        result = app.search(query, limit=limit)
        data = result if isinstance(result, list) else result.get('data', [])
        return data or []
    except Exception as e:
        logger.debug('Firecrawl search failed: %s', e)
        return []

def find_business_website(business_name, county):
    query = f'"{business_name}" {county} Ireland official website'
    results = search_firecrawl(query, limit=3)
    for r in results:
        url = r.get('url', '')
        if url and 'linkedin' not in url and 'facebook' not in url:
            return url
    return None
