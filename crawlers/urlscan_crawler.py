"""
URLScan.io Crawler
Fetches URLs from URLScan.io public scans (LOW Priority - Every 20 min)
"""
import asyncio
import json
from typing import List, Dict
from datetime import datetime, timedelta
from base_crawler import BaseCrawler
import logging

logger = logging.getLogger(__name__)

class URLScanCrawler(BaseCrawler):
    """Crawler for URLScan.io submissions"""
    
    def __init__(self, api_key: str = None):
        super().__init__("URLScan.io", 20, "LOW")  # Every 20 minutes
        self.api_key = api_key  # Optional API key for private scans
        self.base_url = "https://urlscan.io/api/v1"
        
        # Search parameters for suspicious content
        self.search_terms = [
            'phishing', 'suspicious', 'malicious', 'fraud',
            'paypal', 'amazon', 'microsoft', 'google', 'apple',
            'login', 'verify', 'secure', 'account'
        ]
    
    def should_crawl(self) -> bool:
        """Check if 20 minutes have passed since last crawl"""
        if not self.last_crawl_time:
            return True
        return (datetime.now().timestamp() - self.last_crawl_time) >= (self.frequency_minutes * 60)
    
    async def crawl(self) -> List[str]:
        """Fetch URLs from URLScan.io"""
        all_urls = []
        
        # Search for recent suspicious scans
        recent_scans = await self._search_recent_scans()
        
        # Extract URLs from scan results
        for scan in recent_scans:
            url = self._extract_url_from_scan(scan)
            if url and self._is_url_suspicious(url, scan):
                all_urls.append(url)
        
        # Remove duplicates
        unique_urls = list(set(all_urls))
        
        return unique_urls[:40]  # Limit results
    
    async def _search_recent_scans(self) -> List[Dict]:
        """Search for recent scans on URLScan.io"""
        all_scans = []
        
        try:
            # Search for scans from last 24 hours
            since_date = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d')
            
            # Build search query
            search_query = f"date:>{since_date}"
            
            params = {
                'q': search_query,
                'size': 100,  # Maximum results per request
                'sort': '_score',  # Sort by relevance
            }
            
            headers = {}
            if self.api_key:
                headers['API-Key'] = self.api_key
            
            async with self.session.get(
                f"{self.base_url}/search/", 
                params=params,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if 'results' in data:
                        scans = data['results']
                        
                        # Filter for suspicious scans
                        for scan in scans:
                            if self._is_scan_suspicious(scan):
                                all_scans.append(scan)
                        
                        logger.info(f"Found {len(all_scans)} suspicious scans from URLScan.io")
                
                elif response.status == 429:
                    logger.warning("URLScan.io rate limit exceeded")
                elif response.status == 401:
                    logger.warning("URLScan.io API authentication failed")
                else:
                    logger.error(f"URLScan.io API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error searching URLScan.io: {str(e)}")
        
        return all_scans
    
    def _is_scan_suspicious(self, scan: Dict) -> bool:
        """Determine if a scan looks suspicious"""
        try:
            # Check scan metadata
            page_info = scan.get('page', {})
            task_info = scan.get('task', {})
            stats = scan.get('stats', {})
            
            url = page_info.get('url', '').lower()
            title = page_info.get('title', '').lower()
            domain = page_info.get('domain', '').lower()
            
            # Check for suspicious indicators
            suspicious_indicators = [
                # URL patterns
                'login', 'verify', 'secure', 'account', 'suspend',
                'paypal', 'amazon', 'microsoft', 'google', 'apple',
                'facebook', 'twitter', 'instagram', 'linkedin',
                
                # Title patterns  
                'verification', 'suspended', 'expires', 'urgent',
                'security alert', 'action required', 'confirm'
            ]
            
            text_to_check = f"{url} {title} {domain}"
            
            # Check if any suspicious indicators are present
            if any(indicator in text_to_check for indicator in suspicious_indicators):
                return True
            
            # Check for malicious tags or verdicts
            verdicts = scan.get('verdicts', {})
            if verdicts:
                overall = verdicts.get('overall', {})
                if overall.get('malicious', False):
                    return True
            
            # Check for suspicious domain patterns
            if self._is_domain_suspicious(domain):
                return True
            
        except Exception as e:
            logger.debug(f"Error checking scan suspiciousness: {e}")
        
        return False
    
    def _is_domain_suspicious(self, domain: str) -> bool:
        """Check if domain has suspicious patterns"""
        if not domain:
            return False
        
        domain = domain.lower()
        
        # Check for suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.pw', '.cc']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            return True
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'secure-', 'login-', 'verify-', 'account-',
            'paypal-', 'amazon-', 'microsoft-', 'google-',
            'facebook-', 'twitter-', 'instagram-'
        ]
        
        return any(pattern in domain for pattern in suspicious_patterns)
    
    def _extract_url_from_scan(self, scan: Dict) -> str:
        """Extract URL from scan data"""
        try:
            page_info = scan.get('page', {})
            return page_info.get('url', '')
        except:
            return ''
    
    def _is_url_suspicious(self, url: str, scan: Dict) -> bool:
        """Additional URL validation"""
        if not url:
            return False
        
        url_lower = url.lower()
        
        # Skip known legitimate sites
        legitimate_domains = [
            'google.com', 'microsoft.com', 'apple.com', 'paypal.com',
            'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com',
            'github.com', 'stackoverflow.com', 'wikipedia.org',
            'youtube.com', 'reddit.com'
        ]
        
        for domain in legitimate_domains:
            if domain in url_lower:
                return False
        
        return True
    
    async def submit_url_for_scan(self, url: str) -> Dict:
        """Submit a URL for scanning (requires API key)"""
        if not self.api_key:
            logger.warning("API key required for URL submission")
            return {}
        
        try:
            headers = {
                'API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'url': url,
                'visibility': 'public'
            }
            
            async with self.session.post(
                f"{self.base_url}/scan/",
                headers=headers,
                json=data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully submitted URL for scan: {url}")
                    return result
                else:
                    logger.error(f"Failed to submit URL: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error submitting URL for scan: {e}")
        
        return {}
