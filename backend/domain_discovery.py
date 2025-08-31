#!/usr/bin/env python3
"""
Domain Discovery and Monitoring System
======================================

This module handles the discovery of potential phishing domains targeting CSEs.
It searches for domains similar to whitelisted domains using various techniques:
1. Certificate Transparency logs
2. DNS monitoring
3. Domain registration feeds
4. Subdomain enumeration
5. Typosquatting generation and checking
"""

import asyncio
import aiohttp
import dns.resolver
import dns.exception
import json
import time
import re
import requests
from typing import Dict, List, Set, Optional, Tuple, AsyncGenerator
from datetime import datetime, timedelta
import threading
import queue
import ssl
import socket
from urllib.parse import urlparse
import subprocess
import os
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from cse_manager import CSEManager
from phishing_feature_extractor import PhishingFeatureExtractor


class DomainDiscoveryEngine:
    """Engine for discovering potential phishing domains targeting CSEs."""
    
    def __init__(self, cse_manager: CSEManager):
        """Initialize the domain discovery engine."""
        self.cse_manager = cse_manager
        self.feature_extractor = PhishingFeatureExtractor()
        self.discovered_domains = set()
        self.monitoring_active = False
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Certificate Transparency log servers
        self.ct_logs = [
            "https://crt.sh",
            "https://ct.googleapis.com/logs/argon2024/",
            "https://ct.googleapis.com/logs/xenon2024/"
        ]
        
        # DNS servers to use
        self.dns_servers = ['8.8.8.8', '1.1.1.1', '9.9.9.9']
        
    def generate_typosquatting_variants(self, domain: str) -> Set[str]:
        """
        Generate potential typosquatting variants of a domain.
        
        Common techniques:
        - Character omission
        - Character insertion  
        - Character substitution
        - Character swapping
        - Homograph attacks
        - Subdomain hijacking
        """
        variants = set()
        
        # Parse domain
        parts = domain.split('.')
        if len(parts) < 2:
            return variants
        
        base_domain = parts[0]
        tld = '.'.join(parts[1:])
        
        # 1. Character omission
        for i in range(len(base_domain)):
            variant = base_domain[:i] + base_domain[i+1:]
            if len(variant) > 2:  # Avoid too short domains
                variants.add(f"{variant}.{tld}")
        
        # 2. Character insertion
        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
        for i in range(len(base_domain) + 1):
            for char in alphabet:
                variant = base_domain[:i] + char + base_domain[i:]
                variants.add(f"{variant}.{tld}")
        
        # 3. Character substitution
        substitutions = {
            'a': ['@', '4'], 'e': ['3'], 'i': ['1', 'l'], 'o': ['0'], 's': ['5', '$'],
            'l': ['1', 'i'], 'g': ['9'], 't': ['7'], 'b': ['6'], 'z': ['2']
        }
        
        for i, char in enumerate(base_domain):
            if char in substitutions:
                for sub_char in substitutions[char]:
                    variant = base_domain[:i] + sub_char + base_domain[i+1:]
                    variants.add(f"{variant}.{tld}")
        
        # 4. Character swapping (adjacent characters)
        for i in range(len(base_domain) - 1):
            variant = (base_domain[:i] + 
                      base_domain[i+1] + 
                      base_domain[i] + 
                      base_domain[i+2:])
            variants.add(f"{variant}.{tld}")
        
        # 5. Common TLD variations
        common_tlds = ['com', 'org', 'net', 'info', 'biz', 'co.in', 'in', 'co.uk']
        for alt_tld in common_tlds:
            if alt_tld != tld:
                variants.add(f"{base_domain}.{alt_tld}")
        
        # 6. Subdomain variations
        subdomains = ['www', 'secure', 'login', 'mail', 'webmail', 'ftp', 'admin', 'portal']
        for subdomain in subdomains:
            variants.add(f"{subdomain}.{domain}")
            variants.add(f"{subdomain}-{base_domain}.{tld}")
        
        # 7. Hyphen insertion
        for i in range(1, len(base_domain)):
            variant = base_domain[:i] + '-' + base_domain[i:]
            variants.add(f"{variant}.{tld}")
        
        # 8. Double character
        for i in range(len(base_domain)):
            variant = base_domain[:i] + base_domain[i] + base_domain[i:]
            variants.add(f"{variant}.{tld}")
        
        # Remove original domain and limit results
        variants.discard(domain)
        return variants
    
    async def check_domain_existence(self, domain: str) -> Dict[str, any]:
        """
        Check if a domain exists and gather information about it.
        
        Returns dictionary with domain information or None if domain doesn't exist.
        """
        domain_info = {
            'domain': domain,
            'exists': False,
            'has_dns_record': False,
            'has_mx_record': False,
            'has_ssl': False,
            'status_code': None,
            'ip_addresses': [],
            'creation_date': None,
            'registrar': None,
            'nameservers': [],
            'discovery_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check DNS resolution
            resolver = dns.resolver.Resolver()
            resolver.nameservers = self.dns_servers
            resolver.timeout = 5
            resolver.lifetime = 10
            
            try:
                # Try A record
                answers = resolver.resolve(domain, 'A')
                domain_info['has_dns_record'] = True
                domain_info['ip_addresses'] = [str(rdata) for rdata in answers]
                domain_info['exists'] = True
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                pass
            
            try:
                # Try MX record
                answers = resolver.resolve(domain, 'MX')
                domain_info['has_mx_record'] = True
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                pass
            
            # If domain has DNS records, check HTTP/HTTPS
            if domain_info['has_dns_record']:
                await self._check_http_response(domain, domain_info)
                
                # Check SSL certificate
                domain_info['has_ssl'] = self._check_ssl_certificate(domain)
                
                # Get WHOIS information (if available)
                whois_info = self._get_whois_info(domain)
                if whois_info:
                    domain_info.update(whois_info)
        
        except Exception as e:
            self.logger.error(f"Error checking domain {domain}: {e}")
        
        return domain_info if domain_info['exists'] else None
    
    async def _check_http_response(self, domain: str, domain_info: Dict):
        """Check HTTP response for domain."""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for protocol in ['https', 'http']:
                    try:
                        url = f"{protocol}://{domain}"
                        async with session.get(url, allow_redirects=False) as response:
                            domain_info['status_code'] = response.status
                            domain_info['responds_to_http'] = True
                            domain_info['protocol'] = protocol
                            break
                    except:
                        continue
        except Exception as e:
            self.logger.debug(f"HTTP check failed for {domain}: {e}")
    
    def _check_ssl_certificate(self, domain: str) -> bool:
        """Check if domain has valid SSL certificate."""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    return True
        except:
            return False
    
    def _get_whois_info(self, domain: str) -> Optional[Dict]:
        """Get WHOIS information for domain."""
        try:
            import whois
            w = whois.whois(domain)
            if w:
                return {
                    'creation_date': str(w.creation_date) if w.creation_date else None,
                    'registrar': str(w.registrar) if w.registrar else None,
                    'nameservers': w.name_servers if w.name_servers else []
                }
        except Exception as e:
            self.logger.debug(f"WHOIS lookup failed for {domain}: {e}")
        return None
    
    async def search_certificate_transparency(self, target_domain: str) -> List[str]:
        """
        Search Certificate Transparency logs for domains similar to target.
        """
        discovered = []
        
        try:
            # Search crt.sh
            url = f"https://crt.sh/?q=%25{target_domain}%25&output=json"
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for cert in data:
                            if 'name_value' in cert:
                                domains = cert['name_value'].split('\n')
                                for domain in domains:
                                    domain = domain.strip().lower()
                                    if domain and domain != target_domain:
                                        # Check if it's similar to target
                                        target_cse, score, _ = self.cse_manager.identify_target_cse(domain)
                                        if target_cse and score > 0.3:
                                            discovered.append(domain)
            
        except Exception as e:
            self.logger.error(f"Certificate Transparency search failed: {e}")
        
        return list(set(discovered))  # Remove duplicates
    
    async def discover_domains_for_cse(self, cse_name: str, 
                                     max_variants: int = 1000) -> List[Dict]:
        """
        Discover potential phishing domains targeting a specific CSE.
        
        Returns list of domain information dictionaries.
        """
        self.logger.info(f"🔍 Starting domain discovery for CSE: {cse_name}")
        
        cse_info = self.cse_manager.get_cse_by_name(cse_name)
        if not cse_info:
            self.logger.error(f"CSE not found: {cse_name}")
            return []
        
        discovered_domains = []
        
        # Get whitelisted domains for this CSE
        whitelisted_domains = cse_info['whitelisted_domains']
        
        for target_domain in whitelisted_domains:
            self.logger.info(f"  🎯 Targeting: {target_domain}")
            
            # 1. Generate typosquatting variants
            variants = self.generate_typosquatting_variants(target_domain)
            self.logger.info(f"    Generated {len(variants)} variants")
            
            # Limit variants to avoid overwhelming
            if len(variants) > max_variants:
                variants = list(variants)[:max_variants]
            
            # 2. Check Certificate Transparency logs
            ct_domains = await self.search_certificate_transparency(target_domain)
            variants.update(ct_domains)
            self.logger.info(f"    Found {len(ct_domains)} domains in CT logs")
            
            # 3. Check domain existence in batches
            batch_size = 50
            semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
            
            async def check_domain_with_semaphore(domain):
                async with semaphore:
                    return await self.check_domain_existence(domain)
            
            for i in range(0, len(variants), batch_size):
                batch = list(variants)[i:i+batch_size]
                self.logger.info(f"    Checking batch {i//batch_size + 1}/{(len(variants)-1)//batch_size + 1}")
                
                tasks = [check_domain_with_semaphore(domain) for domain in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result:
                        # Analyze with CSE manager
                        target_cse, similarity_score, reasoning = \
                            self.cse_manager.identify_target_cse(result['domain'])
                        
                        if target_cse == cse_name and similarity_score > 0.3:
                            result['target_cse'] = target_cse
                            result['similarity_score'] = similarity_score
                            result['risk_classification'] = \
                                self.cse_manager.classify_domain_risk(result['domain'], similarity_score)
                            result['reasoning'] = reasoning
                            discovered_domains.append(result)
                
                # Add delay between batches to be respectful
                await asyncio.sleep(1)
        
        self.logger.info(f"✅ Discovery complete for {cse_name}: {len(discovered_domains)} domains found")
        return discovered_domains
    
    async def discover_all_cse_domains(self, max_variants_per_domain: int = 500) -> Dict[str, List[Dict]]:
        """
        Discover potential phishing domains for all CSEs.
        
        Returns dictionary mapping CSE names to their discovered domains.
        """
        self.logger.info("🚀 Starting comprehensive domain discovery for all CSEs")
        
        all_cses = self.cse_manager.get_all_cses()
        results = {}
        
        for cse_name in all_cses.keys():
            try:
                discovered = await self.discover_domains_for_cse(cse_name, max_variants_per_domain)
                results[cse_name] = discovered
                
                # Save intermediate results
                self._save_discovery_results(cse_name, discovered)
                
            except Exception as e:
                self.logger.error(f"Error discovering domains for {cse_name}: {e}")
                results[cse_name] = []
        
        # Save comprehensive results
        self._save_comprehensive_results(results)
        
        return results
    
    def _save_discovery_results(self, cse_name: str, domains: List[Dict]):
        """Save discovery results for a specific CSE."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"discovery_results_{cse_name.replace(' ', '_')}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'cse_name': cse_name,
                    'discovery_timestamp': datetime.now().isoformat(),
                    'total_domains_found': len(domains),
                    'domains': domains
                }, f, indent=2, default=str)
            
            self.logger.info(f"💾 Saved results for {cse_name} to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving results for {cse_name}: {e}")
    
    def _save_comprehensive_results(self, all_results: Dict[str, List[Dict]]):
        """Save comprehensive discovery results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_discovery_results_{timestamp}.json"
        
        # Calculate statistics
        total_domains = sum(len(domains) for domains in all_results.values())
        phishing_domains = sum(
            len([d for d in domains if d.get('risk_classification') == 'PHISHING'])
            for domains in all_results.values()
        )
        suspected_domains = sum(
            len([d for d in domains if d.get('risk_classification') == 'SUSPECTED'])
            for domains in all_results.values()
        )
        
        summary = {
            'discovery_timestamp': datetime.now().isoformat(),
            'total_cses_processed': len(all_results),
            'total_domains_discovered': total_domains,
            'phishing_domains': phishing_domains,
            'suspected_domains': suspected_domains,
            'results_by_cse': all_results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"💾 Saved comprehensive results to {filename}")
            self.logger.info(f"📊 Summary: {total_domains} domains found, "
                           f"{phishing_domains} phishing, {suspected_domains} suspected")
        except Exception as e:
            self.logger.error(f"Error saving comprehensive results: {e}")
    
    def start_continuous_monitoring(self, check_interval_hours: int = 24):
        """
        Start continuous monitoring for new phishing domains.
        
        This runs in a separate thread and performs periodic discovery.
        """
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self.logger.info("🔄 Starting scheduled domain discovery")
                    
                    # Run discovery
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(self.discover_all_cse_domains())
                    loop.close()
                    
                    # Wait for next check
                    time.sleep(check_interval_hours * 3600)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        self.logger.info(f"🚀 Started continuous monitoring (interval: {check_interval_hours}h)")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        self.logger.info("🛑 Stopped continuous monitoring")


async def main():
    """Demo function to test domain discovery."""
    print("🔍 Domain Discovery Engine Demo")
    print("===============================")
    
    # Initialize components
    cse_manager = CSEManager()
    discovery_engine = DomainDiscoveryEngine(cse_manager)
    
    # Test with a single CSE
    cse_name = "State Bank of India (SBI)"
    print(f"\n🎯 Testing discovery for: {cse_name}")
    
    discovered = await discovery_engine.discover_domains_for_cse(cse_name, max_variants=100)
    
    print(f"\n📊 Discovery Results:")
    print(f"  Total domains found: {len(discovered)}")
    
    if discovered:
        phishing = [d for d in discovered if d.get('risk_classification') == 'PHISHING']
        suspected = [d for d in discovered if d.get('risk_classification') == 'SUSPECTED']
        
        print(f"  Phishing domains: {len(phishing)}")
        print(f"  Suspected domains: {len(suspected)}")
        
        print(f"\n🚨 Top findings:")
        for domain_info in discovered[:5]:
            print(f"    • {domain_info['domain']} "
                  f"({domain_info['risk_classification']}, "
                  f"similarity: {domain_info['similarity_score']:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
