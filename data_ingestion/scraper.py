import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time

START_URL = "https://dubaigolf.com/"
MAX_PAGES = 400  # Limit for demonstration and time constraints
OUTPUT_FILE = "raw_data.jsonl"

def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    return bool(parsed.netloc) and parsed.netloc.endswith(base_domain)

def scraper():
    queue = deque([START_URL])
    visited = set([START_URL])
    base_domain = urlparse(START_URL).netloc
    
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    
    print(f"Starting BFS scrape of {START_URL}")
    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        while queue and count < MAX_PAGES:
            url = queue.popleft()
            print(f"Scraping ({count+1}/{MAX_PAGES}): {url}")
            try:
                # Add headers to simulate a real browser
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    html_content = response.text
                    
                    data = {
                        "url": url,
                        "html": html_content
                    }
                    f.write(json.dumps(data) + "\n")
                    count += 1
                    
                    soup = BeautifulSoup(html_content, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(url, href)
                        full_url = urlparse(full_url)._replace(fragment="").geturl()
                        
                        if is_valid_url(full_url, base_domain) and full_url not in visited:
                            visited.add(full_url)
                            queue.append(full_url)
                            
                time.sleep(1) # Polite delay
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
                
    print(f"Scraping completed. Raw data saved to {output_path}")

if __name__ == "__main__":
    scraper()
