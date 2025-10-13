"""
Certificate Transparency Logs Crawler
Monitors CT logs for newly registered domains (HIGH Priority - Every 5-10 min)
"""
import asyncio
import json
import re
from typing import List
from datetime import datetime, timedelta
from base_crawler import BaseCrawler
import logging

logger = logging.getLogger(__name__)

class CTLogsCrawler(BaseCrawler):
    """Crawler for Certificate Transparency logs"""
    
    def __init__(self):
        super().__init__("CT Logs", 7, "HIGH")  # 7 minutes (5-10 min range)
        self.ct_logs_endpoints = [
            "https://crt.sh/json",
            "https://certstream.calidog.io/",
        ]
        self.suspicious_keywords = [
            'login', 'secure', 'bank', 'paypal', 'amazon', 'microsoft',
            'google', 'apple', 'facebook', 'twitter', 'instagram',
            'verification', 'confirm', 'update', 'suspend', 'urgent'
        ]
    
    def should_crawl(self) -> bool:
        """Check if 7 minutes have passed since last crawl"""
        if not self.last_crawl_time:
            return True
        return (datetime.now().timestamp() - self.last_crawl_time) >= (self.frequency_minutes * 60)
    
    async def crawl(self) -> List[str]:
        """Crawl CT logs for suspicious domains"""
        all_urls = []
        
        # Get recent certificates from crt.sh
        urls_from_crt = await self._crawl_crt_sh()
        all_urls.extend(urls_from_crt)
        
        # Remove duplicates and validate URLs
        unique_urls = list(set(all_urls))
        valid_urls = [url for url in unique_urls if self._is_valid_url(url)]
        
        return valid_urls[:100]  # Limit to 100 URLs per crawl
    
    async def _crawl_crt_sh(self) -> List[str]:
        """Crawl crt.sh for recent certificates"""
        urls = []
        try:
            # Search for certificates from last hour
            yesterday = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d')
            
            for keyword in self.suspicious_keywords[:5]:  # Limit keywords to avoid rate limiting
                params = {
                    'q': f'%.{keyword}.%',
                    'output': 'json',
                    'exclude': 'expired'
                }
                
                async with self.session.get('https://crt.sh/', params=params) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if isinstance(data, list):
                                for cert in data[:20]:  # Limit results
                                    if 'name_value' in cert:
                                        domain = cert['name_value'].strip()
                                        if '\n' in domain:
                                            domain = domain.split('\n')[0]
                                        if domain.startswith('*.'):
                                            domain = domain[2:]
                                        
                                        # Convert domain to URL
                                        url = f"https://{domain}"
                                        if self._is_suspicious_domain(domain):
                                            urls.append(url)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON from crt.sh for keyword: {keyword}")
                
                # Rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error crawling crt.sh: {str(e)}")
        
        return urls
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check if domain looks suspicious"""
        domain = domain.lower()
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'[0-9]{4,}',  # Long numbers
            r'[a-z]+-[a-z]+-[a-z]+',  # Multiple hyphens
            r'(secure|login|verify|update|confirm).*\.(com|net|org)',
            r'(paypal|amazon|microsoft|google|apple).*[0-9]',
            r'\b(bit\.ly|tinyurl|goo\.gl|t\.co)\b',  # URL shorteners
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain):
                return True
        
        # Check for typosquatting indicators
        popular_brands = ['paypal', 'amazon', 'microsoft', 'google', 'apple', 'facebook']
        for brand in popular_brands:
            if brand in domain and domain != f"{brand}.com":
                return True
        
        return False
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url)) and len(url) < 200
