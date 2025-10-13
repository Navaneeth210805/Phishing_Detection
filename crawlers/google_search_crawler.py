"""
Google Search Crawler
Searches for potential phishing sites using Google Custom Search API (MEDIUM Priority - Every 30 min)
"""
import asyncio
import json
from typing import List
from datetime import datetime
from base_crawler import BaseCrawler
import logging

logger = logging.getLogger(__name__)

class GoogleSearchCrawler(BaseCrawler):
    """Crawler using Google Custom Search API"""
    
    def __init__(self, api_key: str = None, search_engine_id: str = None):
        super().__init__("Google Search", 30, "MEDIUM")  # Every 30 minutes
        self.api_key = api_key or "YOUR_GOOGLE_API_KEY"  # Replace with actual API key
        self.search_engine_id = search_engine_id or "YOUR_SEARCH_ENGINE_ID"  # Replace with actual CSE ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Search queries for potential phishing sites
        self.search_queries = [
            "login secure verification site:*.com",
            "paypal secure login verify",
            "amazon account suspended",
            "microsoft account verification",
            "google account security alert",
            "apple id verification required",
            "bank account verify immediately",
            "urgent action required login",
            "suspended account reactivate",
            "confirm identity security"
        ]
    
    def should_crawl(self) -> bool:
        """Check if 30 minutes have passed since last crawl"""
        if not self.last_crawl_time:
            return True
        return (datetime.now().timestamp() - self.last_crawl_time) >= (self.frequency_minutes * 60)
    
    async def crawl(self) -> List[str]:
        """Search Google for suspicious websites"""
        all_urls = []
        
        if self.api_key == "YOUR_GOOGLE_API_KEY":
            logger.warning("Google API key not configured, using demo URLs")
            return self._get_demo_urls()
        
        for query in self.search_queries[:5]:  # Limit queries to avoid API quota
            urls = await self._search_google(query)
            all_urls.extend(urls)
            
            # Rate limiting for API
            await asyncio.sleep(2)
        
        # Remove duplicates and filter
        unique_urls = list(set(all_urls))
        filtered_urls = [url for url in unique_urls if self._is_suspicious_url(url)]
        
        return filtered_urls[:30]  # Limit results
    
    async def _search_google(self, query: str) -> List[str]:
        """Perform Google Custom Search"""
        urls = []
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': 10,  # Number of results
                'safe': 'off',
                'fields': 'items(link,title,snippet)'
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'items' in data:
                        for item in data['items']:
                            url = item.get('link', '')
                            title = item.get('title', '')
                            snippet = item.get('snippet', '')
                            
                            # Check if result looks suspicious
                            if self._is_search_result_suspicious(url, title, snippet):
                                urls.append(url)
                                logger.info(f"Suspicious URL found via Google: {url}")
                
                elif response.status == 403:
                    logger.error("Google API quota exceeded or access denied")
                else:
                    logger.error(f"Google API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error searching Google: {str(e)}")
        
        return urls
    
    def _is_search_result_suspicious(self, url: str, title: str, snippet: str) -> bool:
        """Determine if search result looks suspicious"""
        suspicious_indicators = [
            # URL indicators
            'bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly',
            
            # Title indicators
            'urgent', 'verify now', 'suspended', 'expires today',
            'immediate action', 'click here', 'limited time',
            
            # Snippet indicators
            'verify your account', 'suspended account', 'expires soon',
            'click to verify', 'urgent action required', 'security alert'
        ]
        
        text_to_check = f"{url.lower()} {title.lower()} {snippet.lower()}"
        
        return any(indicator in text_to_check for indicator in suspicious_indicators)
    
    def _is_suspicious_url(self, url: str) -> bool:
        """Additional URL filtering"""
        url_lower = url.lower()
        
        # Skip legitimate sites
        legitimate_domains = [
            'google.com', 'microsoft.com', 'apple.com', 'paypal.com',
            'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com',
            'github.com', 'stackoverflow.com', 'wikipedia.org'
        ]
        
        for domain in legitimate_domains:
            if domain in url_lower:
                return False
        
        # Look for suspicious patterns
        suspicious_patterns = [
            'secure-', 'login-', 'verify-', 'account-',
            'paypal-', 'amazon-', 'microsoft-', 'google-',
            '.tk', '.ml', '.ga', '.cf', '.pw'
        ]
        
        return any(pattern in url_lower for pattern in suspicious_patterns)
    
    def _get_demo_urls(self) -> List[str]:
        """Return demo URLs when API key is not configured"""
        return [
            "https://secure-paypal-verify.com",
            "https://amazon-security-alert.net", 
            "https://microsoft-account-verify.org",
            "https://google-security-check.info",
            "https://apple-id-verification.biz"
        ]
