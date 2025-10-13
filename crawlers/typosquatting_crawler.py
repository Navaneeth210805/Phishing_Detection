"""
Typosquatting Crawler
Generates and checks potential typosquatting domains (HIGH Priority - Every 1 hour)
"""
import asyncio
import itertools
from typing import List
from datetime import datetime
from base_crawler import BaseCrawler
import logging

logger = logging.getLogger(__name__)

class TyposquattingCrawler(BaseCrawler):
    """Crawler for typosquatting domains"""
    
    def __init__(self):
        super().__init__("Typosquatting", 60, "HIGH")  # Every 1 hour
        self.target_brands = [
            'paypal.com', 'amazon.com', 'microsoft.com', 'google.com',
            'apple.com', 'facebook.com', 'twitter.com', 'instagram.com',
            'linkedin.com', 'netflix.com', 'ebay.com', 'walmart.com',
            'chase.com', 'bankofamerica.com', 'wells-fargo.com'
        ]
        self.tlds = [
            '.com', '.net', '.org', '.info', '.biz', '.co', '.io',
            '.cc', '.tk', '.ml', '.ga', '.cf', '.pw'
        ]
    
    def should_crawl(self) -> bool:
        """Check if 1 hour has passed since last crawl"""
        if not self.last_crawl_time:
            return True
        return (datetime.now().timestamp() - self.last_crawl_time) >= (self.frequency_minutes * 60)
    
    async def crawl(self) -> List[str]:
        """Generate typosquatting variants and check if they exist"""
        all_urls = []
        
        for brand in self.target_brands[:5]:  # Limit to avoid overwhelming
            domain_base = brand.split('.')[0]
            variants = self._generate_typosquatting_variants(domain_base)
            
            # Check if domains are active
            active_urls = await self._check_domain_variants(variants)
            all_urls.extend(active_urls)
        
        return all_urls[:50]  # Limit results
    
    def _generate_typosquatting_variants(self, domain: str) -> List[str]:
        """Generate typosquatting variants for a domain"""
        variants = set()
        
        # Character substitution (common typos)
        char_substitutions = {
            'a': ['@', '4'], 'e': ['3'], 'i': ['1', '!'], 'o': ['0'],
            's': ['$', '5'], 'g': ['9'], 'l': ['1'], 't': ['7']
        }
        
        # Generate substitution variants
        for i, char in enumerate(domain):
            if char.lower() in char_substitutions:
                for sub_char in char_substitutions[char.lower()]:
                    variant = domain[:i] + sub_char + domain[i+1:]
                    variants.add(variant)
        
        # Character omission
        for i in range(len(domain)):
            if i > 0:  # Don't remove first character
                variant = domain[:i] + domain[i+1:]
                if len(variant) >= 3:  # Minimum length
                    variants.add(variant)
        
        # Character insertion (common adjacent keys)
        adjacent_keys = {
            'a': ['s', 'q', 'w'], 'b': ['v', 'g', 'h', 'n'], 'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'], 'e': ['w', 'r', 't', 'd', 's'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'], 'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'], 'i': ['u', 'o', 'p', 'k', 'j'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'], 'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p'], 'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'p', 'l', 'k'], 'p': ['o', 'l'], 'q': ['w', 'a'],
            'r': ['e', 't', 'f', 'd'], 's': ['a', 'w', 'e', 'd', 'x', 'z'],
            't': ['r', 'y', 'g', 'f'], 'u': ['y', 'i', 'j', 'h'], 'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'e', 's', 'a'], 'x': ['z', 's', 'd', 'c'], 'y': ['t', 'u', 'h', 'g'],
            'z': ['x', 's']
        }
        
        for i, char in enumerate(domain):
            if char.lower() in adjacent_keys:
                for adj_char in adjacent_keys[char.lower()][:2]:  # Limit to 2 adjacent chars
                    # Insert before
                    variant = domain[:i] + adj_char + domain[i:]
                    variants.add(variant)
                    # Replace
                    variant = domain[:i] + adj_char + domain[i+1:]
                    variants.add(variant)
        
        # Add common prefixes/suffixes
        prefixes = ['secure', 'login', 'verify', 'my', 'www', 'new', 'mobile']
        suffixes = ['secure', 'login', 'verify', 'online', 'site', 'web', 'app']
        
        for prefix in prefixes[:3]:  # Limit
            variants.add(f"{prefix}{domain}")
            variants.add(f"{prefix}-{domain}")
        
        for suffix in suffixes[:3]:  # Limit
            variants.add(f"{domain}{suffix}")
            variants.add(f"{domain}-{suffix}")
        
        # Convert to full URLs with different TLDs
        full_urls = []
        for variant in list(variants)[:30]:  # Limit variants
            for tld in self.tlds[:5]:  # Limit TLDs
                full_urls.append(f"https://{variant}{tld}")
        
        return full_urls
    
    async def _check_domain_variants(self, variants: List[str]) -> List[str]:
        """Check which domain variants are active"""
        active_domains = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(10)
        
        async def check_single_domain(url: str):
            async with semaphore:
                try:
                    async with self.session.head(url, timeout=5) as response:
                        if response.status < 400:
                            active_domains.append(url)
                            logger.info(f"Active typosquatting domain found: {url}")
                except Exception:
                    # Domain not active or unreachable
                    pass
                
                # Rate limiting
                await asyncio.sleep(0.5)
        
        # Check all variants concurrently
        tasks = [check_single_domain(url) for url in variants]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return active_domains
