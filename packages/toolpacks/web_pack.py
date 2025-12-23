"""
nAI Toolpacks - Web Pack
========================

Web scraping and URL content extraction.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urljoin
from datetime import datetime


@dataclass
class URLContent:
    """Extracted content from a URL."""
    url: str
    title: str
    text: str
    html: Optional[str] = None
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    @property
    def domain(self) -> str:
        """Get the domain from the URL."""
        parsed = urlparse(self.url)
        return parsed.netloc
    
    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.text.split())


class WebScraper:
    """
    Web scraper for extracting content from URLs.
    
    Features:
    - HTML to clean text conversion
    - Metadata extraction (title, description, author)
    - Link and image extraction
    - JavaScript rendering (optional, requires Playwright)
    """
    
    def __init__(
        self,
        user_agent: str = "nAI Web Scraper/0.1",
        timeout: int = 30,
        render_js: bool = False,
        extract_links: bool = True,
        extract_images: bool = False
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.render_js = render_js
        self.extract_links = extract_links
        self.extract_images = extract_images
    
    def _clean_html(self, html: str) -> str:
        """Convert HTML to clean text."""
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.IGNORECASE)
        html = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html, flags=re.IGNORECASE)
        html = re.sub(r'<noscript[^>]*>[\s\S]*?</noscript>', '', html, flags=re.IGNORECASE)
        
        # Remove HTML comments
        html = re.sub(r'<!--[\s\S]*?-->', '', html)
        
        # Replace common block elements with newlines
        html = re.sub(r'<(?:p|div|br|h[1-6]|li|tr|section|article)[^>]*>', '\n', html, flags=re.IGNORECASE)
        
        # Remove remaining HTML tags
        html = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode common HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        html = html.replace('&#39;', "'")
        html = html.replace('&apos;', "'")
        
        # Clean up whitespace
        html = re.sub(r' +', ' ', html)
        html = re.sub(r'\n\s*\n', '\n\n', html)
        html = re.sub(r'\n{3,}', '\n\n', html)
        
        return html.strip()
    
    def _extract_title(self, html: str) -> str:
        """Extract page title."""
        # Try <title> tag
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return self._clean_html(match.group(1)).strip()
        
        # Try <h1> tag
        match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return self._clean_html(match.group(1)).strip()
        
        return ""
    
    def _extract_meta(self, html: str) -> Dict[str, str]:
        """Extract metadata from meta tags."""
        meta = {}
        
        # Description
        match = re.search(
            r'<meta\s+[^>]*(?:name|property)=["\'](?:description|og:description)["\'][^>]*content=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        )
        if match:
            meta["description"] = match.group(1).strip()
        
        # Author
        match = re.search(
            r'<meta\s+[^>]*name=["\']author["\'][^>]*content=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        )
        if match:
            meta["author"] = match.group(1).strip()
        
        # Keywords
        match = re.search(
            r'<meta\s+[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        )
        if match:
            meta["keywords"] = match.group(1).strip()
        
        # OG Image
        match = re.search(
            r'<meta\s+[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        )
        if match:
            meta["og_image"] = match.group(1).strip()
        
        return meta
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from the page."""
        links = []
        pattern = r'<a\s+[^>]*href=["\']([^"\'#]+)["\']'
        
        for match in re.finditer(pattern, html, re.IGNORECASE):
            href = match.group(1).strip()
            if href and not href.startswith(('javascript:', 'mailto:', 'tel:')):
                # Make absolute URL
                absolute = urljoin(base_url, href)
                links.append(absolute)
        
        return list(set(links))  # Remove duplicates
    
    def _extract_images(self, html: str, base_url: str) -> List[str]:
        """Extract all image URLs from the page."""
        images = []
        pattern = r'<img\s+[^>]*src=["\']([^"\']+)["\']'
        
        for match in re.finditer(pattern, html, re.IGNORECASE):
            src = match.group(1).strip()
            if src and not src.startswith('data:'):
                absolute = urljoin(base_url, src)
                images.append(absolute)
        
        return list(set(images))
    
    async def scrape_async(self, url: str) -> URLContent:
        """Scrape a URL asynchronously."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": self.user_agent}
            
            async with session.get(url, headers=headers, timeout=self.timeout) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}: {response.reason}")
                
                html = await response.text()
                content_type = response.headers.get("content-type", "")
        
        return self._process_html(url, html, content_type)
    
    def scrape(self, url: str) -> URLContent:
        """Scrape a URL synchronously."""
        import requests
        
        headers = {"User-Agent": self.user_agent}
        response = requests.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        
        return self._process_html(url, response.text, response.headers.get("content-type", ""))
    
    def _process_html(self, url: str, html: str, content_type: str) -> URLContent:
        """Process HTML into URLContent."""
        title = self._extract_title(html)
        meta = self._extract_meta(html)
        text = self._clean_html(html)
        
        links = self._extract_links(html, url) if self.extract_links else []
        images = self._extract_images(html, url) if self.extract_images else []
        
        return URLContent(
            url=url,
            title=title,
            text=text,
            html=html,
            links=links,
            images=images,
            metadata={
                "content_type": content_type,
                **meta,
            },
        )
    
    def scrape_with_js(self, url: str) -> URLContent:
        """
        Scrape a URL with JavaScript rendering using Playwright.
        
        Requires: playwright (pip install playwright && playwright install)
        """
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=self.user_agent)
            page.goto(url, timeout=self.timeout * 1000)
            page.wait_for_load_state("networkidle")
            
            html = page.content()
            browser.close()
        
        return self._process_html(url, html, "text/html")


def scrape_url(
    url: str,
    render_js: bool = False,
    timeout: int = 30
) -> str:
    """
    Convenience function to extract text from a URL.
    
    Args:
        url: URL to scrape
        render_js: Use headless browser for JS rendering
        timeout: Request timeout in seconds
    
    Returns:
        Extracted text as string
    """
    scraper = WebScraper(timeout=timeout, render_js=render_js)
    
    if render_js:
        content = scraper.scrape_with_js(url)
    else:
        content = scraper.scrape(url)
    
    return f"# {content.title}\n\n{content.text}"

