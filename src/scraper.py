"""scraper.py
Lightweight wrapper for scraping product pages. If a full scraper.py exists at repo root,
this module will attempt to import and expose its main functions; otherwise provides a small
requests + BeautifulSoup helper used by notebooks.
"""
try:
    from .. import scraper as full_scraper  # type: ignore
except Exception:
    full_scraper = None

if full_scraper:
    get_product_info = getattr(full_scraper, 'get_product_info', None)
else:
    # Minimal fallback
    import requests
    from bs4 import BeautifulSoup

    def get_product_info(url: str):
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'lxml')
        title = soup.title.string if soup.title else ''
        return {'url': url, 'title': title}


__all__ = ['get_product_info']
