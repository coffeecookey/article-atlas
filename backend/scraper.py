from trafilatura import fetch_url, extract
import time
import requests
from typing import Optional, List, Dict
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime
import trafilatura


# (import ArticleContent model and use that)

# Global State
_last_request_time = {}

# To enforce rate limiting based on the domain of website
def _enforce_rate_limit(url: str, delay: float = 1.0):
    # To find the network location from the url
    domain = urlparse(url).netloc
    
    # check if the domain was previously visited
    if domain in _last_request_time:
        elapsed = time.time() - _last_request_time[domain]
        if elapsed < delay:
            time.sleep(delay - elapsed)
    
    _last_request_time[domain] = time.time()


def _fetch_with_retry(url: str, max_retries: int = 3, timeout: int = 10):
    headers = {
        # user-agent name is added to identify my program since many websites require this
        'User-Agent': 'ArticleBot/1.0 (academic project)'
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text
        
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  
                time.sleep(wait_time)
                continue
            return None
    
    return None

def scrape_article(url: str) -> Dict:  # returns a dictionary instead of ArticleContent right now
    
    # Enforce rate limiting
    _enforce_rate_limit(url)
    html = _fetch_with_retry(url, max_retries=3, timeout=10)
    
    if html is None:
        raise RuntimeError(f"Failed to fetch content from {url} after 3 attempts")
    
    # Extract metadata using Trafilatura
    metadata = trafilatura.extract_metadata(html)
    
    # Extract main text
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=True,        
        favor_precision=False,  
        url=url
    )
    
    if text is None:
        raise RuntimeError(f"Failed to extract article text from {url}")
    
    # return ArticleContent(url=url, title=..., author=..., ...) once ArticleContent is added
    return {
        'url': url,
        'title': metadata.title if metadata else None,
        'author': metadata.author if metadata else None,
        'publish_date': metadata.date if metadata else None,
        'tags': metadata.tags if metadata and metadata.tags else [],
        'text': text,
        'scraped_at': datetime.now().isoformat()
    }

# To test
'''
if __name__ == "__main__":
    try:
        print("Starting to scrape article...")
        
        article = scrape_article(
            "https://timesofindia.indiatimes.com/india/ajit-pawar-death-news-ncp-dada-to-supporters-heir-to-pawar-politics-5-turning-points-of-workaholic-pawar-life/articleshow/127695282.cms"
        )
        
        print(f"\n scrape works phew")
        # CHANGED: Access dictionary keys instead of object attributes
        print(f"Title: {article['title']}")
        print(f"Author: {article['author']}")
        print(f"Published: {article['publish_date']}")
        print(f"Tags: {article['tags']}")
        print(f"Text length: {len(article['text']) if article['text'] else 0} characters")
        
        # Print first 500 characters of text
        if article['text']:
            print(f"\nFirst 500 characters of text:")
            print(f"{article['text'][:500]}...")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
'''