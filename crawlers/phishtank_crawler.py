"""
PhishTank Crawler
Fetches latest phishing URLs from PhishTank API (MEDIUM Priority - Every 15 min)
"""
import asyncio
import json
from typing import List
from datetime import datetime, timedelta
from base_crawler import BaseCrawler
import logging

logger = logging.getLogger(__name__)

class PhishTankCrawler(BaseCrawler):
    """Crawler for PhishTank phishing database"""
    
    def __init__(self, api_key: str = None):
        super().__init__("PhishTank", 15, "MEDIUM")  # Every 15 minutes
        self.api_key = api_key  # Optional API key for higher rate limits
        self.base_url = "http://data.phishtank.com/data"
        
        # PhishTank endpoints
        self.endpoints = {
            'online': f"{self.base_url}/online-valid.json",
            'verified': f"{self.base_url}/verified_online.json",
        }
    
    def should_crawl(self) -> bool:
        """Check if 15 minutes have passed since last crawl"""
        if not self.last_crawl_time:
            return True
        return (datetime.now().timestamp() - self.last_crawl_time) >= (self.frequency_minutes * 60)
    
    async def crawl(self) -> List[str]:
        """Fetch latest phishing URLs from PhishTank"""
        all_urls = []
        
        # Fetch from different endpoints
        for endpoint_name, endpoint_url in self.endpoints.items():
            urls = await self._fetch_phishtank_data(endpoint_name, endpoint_url)
            all_urls.extend(urls)
        
        # Remove duplicates and filter recent ones
        unique_urls = list(set(all_urls))
        recent_urls = self._filter_recent_urls(unique_urls)
        
        return recent_urls[:50]  # Limit to 50 most recent URLs
    
    async def _fetch_phishtank_data(self, endpoint_name: str, endpoint_url: str) -> List[str]:
        """Fetch data from a PhishTank endpoint"""
        urls = []
        
        try:
            params = {}
            if self.api_key:
                params['app_key'] = self.api_key
            
            # Add format parameter
            params['format'] = 'json'
            
            async with self.session.get(endpoint_url, params=params) as response:
                if response.status == 200:
                    # PhishTank returns JSONP, need to extract JSON
                    text = await response.text()
                    
                    # Handle different response formats
                    if text.startswith('var phishTankData = '):
                        # JSONP format
                        json_start = text.find('[')
                        json_end = text.rfind(']') + 1
                        if json_start != -1 and json_end != -1:
                            json_text = text[json_start:json_end]
                    else:
                        # Direct JSON format
                        json_text = text
                    
                    try:
                        data = json.loads(json_text)
                        
                        if isinstance(data, list):
                            for entry in data:
                                if isinstance(entry, dict):
                                    url = entry.get('url', '')
                                    verified = entry.get('verified', 'no')
                                    online = entry.get('online', 'no')
                                    
                                    # Only include verified and online phishing URLs
                                    if verified == 'yes' and online == 'yes' and url:
                                        urls.append(url)
                                        logger.debug(f"PhishTank URL found: {url}")
                        
                        logger.info(f"Fetched {len(urls)} URLs from PhishTank {endpoint_name}")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse PhishTank JSON from {endpoint_name}: {e}")
                        
                elif response.status == 509:
                    logger.warning("PhishTank rate limit exceeded")
                else:
                    logger.error(f"PhishTank API error for {endpoint_name}: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching PhishTank data from {endpoint_name}: {str(e)}")
        
        return urls
    
    def _filter_recent_urls(self, urls: List[str]) -> List[str]:
        """Filter URLs to get most recent/relevant ones"""
        # Since PhishTank doesn't provide timestamps in free API,
        # we'll use other heuristics to prioritize URLs
        
        # Prioritize URLs with suspicious patterns
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for url in urls:
            url_lower = url.lower()
            
            # High priority: Major brand impersonation
            if any(brand in url_lower for brand in [
                'paypal', 'amazon', 'microsoft', 'google', 'apple',
                'facebook', 'instagram', 'twitter', 'linkedin'
            ]):
                high_priority.append(url)
            
            # Medium priority: Financial/security terms
            elif any(term in url_lower for term in [
                'bank', 'login', 'secure', 'verify', 'account',
                'payment', 'billing', 'credit'
            ]):
                medium_priority.append(url)
            
            # Low priority: Everything else
            else:
                low_priority.append(url)
        
        # Return prioritized list
        return high_priority + medium_priority + low_priority
    
    async def get_phishtank_stats(self) -> dict:
        """Get PhishTank statistics (optional utility method)"""
        try:
            async with self.session.get(f"{self.base_url}/stats.json") as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error fetching PhishTank stats: {e}")
        
        return {}
