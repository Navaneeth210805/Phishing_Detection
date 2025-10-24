#!/usr/bin/env python3
"""
Phishing Detection Feature Extractor
=====================================

This script extracts various features from URLs/domains that can be used for phishing detection.
Features include URL-based, domain-based, and content-based characteristics.

Usage:
    python phishing_feature_extractor.py --url <url_to_analyze>
    python phishing_feature_extractor.py --file <file_with_urls>
    python phishing_feature_extractor.py --batch  # Process training dataset
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import tldextract
import whois
import re
import ssl
import socket
from urllib.parse import urlparse, parse_qs, urljoin
import datetime
import time
import os
import argparse
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class PhishingFeatureExtractor:
    """Extract features from URLs/domains for phishing detection."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.timeout = 10
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        
    def extract_url_features(self, url: str) -> Dict[str, Any]:
        """Extract URL-based features.""" 
        features = {}   
        original_url = url  # Store original URL
        
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path
            query = parsed.query
            
            # Basic URL metrics
            features['url_length'] = len(url)
            features['domain_length'] = len(domain)
            features['path_length'] = len(path)
            features['query_length'] = len(query)
            
            # URL structure features
            features['has_ip_address'] = self._has_ip_address(domain)
            features['has_suspicious_tld'] = self._has_suspicious_tld(domain)
            features['subdomain_count'] = len(domain.split('.')) - 2 if domain.count('.') >= 2 else 0
            features['dash_count'] = domain.count('-')
            features['dot_count'] = domain.count('.')
            features['slash_count'] = url.count('/')
            features['at_symbol'] = 1 if '@' in url else 0
            features['double_slash_redirecting'] = 1 if '//' in path else 0
            
            # Suspicious patterns
            features['has_suspicious_keywords'] = self._has_suspicious_keywords(url)
            features['has_url_shortening'] = self._is_shortened_url(domain)
            features['has_suspicious_port'] = self._has_suspicious_port(parsed.port)
            
            # Domain features using tldextract
            extracted = tldextract.extract(url)
            features['domain_parts'] = {
                'subdomain': extracted.subdomain,
                'domain': extracted.domain, 
                'suffix': extracted.suffix,
                'fqdn': extracted.fqdn
            }
            
            # Character analysis
            features['digit_ratio'] = self._calculate_digit_ratio(domain)
            features['special_char_ratio'] = self._calculate_special_char_ratio(domain)
            features['entropy'] = self._calculate_entropy(domain)
            
            # HTTPS usage - improved detection logic
            if original_url.startswith('https://'):
                features['uses_https'] = 1
                print(f"DEBUG: Domain {domain} uses HTTPS (from URL)")
            elif original_url.startswith('http://'):
                # Explicitly HTTP, check if HTTPS is also available
                https_support = self._test_https_support(domain)
                features['uses_https'] = https_support
                print(f"DEBUG: Domain {domain} HTTP URL, HTTPS support: {https_support}")
            else:
                # For bare domains, test if they support HTTPS
                https_support = self._test_https_support(domain)
                features['uses_https'] = https_support
                print(f"DEBUG: Domain {domain} bare domain, HTTPS support: {https_support}")
            
        except Exception as e:
            print(f"Error extracting URL features: {e}")
            # Set default values for failed extraction
            features.update(self._get_default_url_features())
            
        return features
    
    def _test_https_support(self, domain: str) -> int:
        """Test if a domain supports HTTPS with proper SSL verification."""
        try:
            import ssl
            import socket
            import requests
            
            # Method 1: Try HTTPS request
            try:
                response = requests.get(f'https://{domain}', timeout=10, verify=True)
                if response.status_code < 400:
                    print(f"DEBUG: {domain} - HTTPS works via HTTP request")
                    return 1
            except requests.exceptions.SSLError:
                print(f"DEBUG: {domain} - SSL Error in HTTPS request")
                return 0
            except requests.exceptions.ConnectionError:
                print(f"DEBUG: {domain} - Connection error for HTTPS")
                return 0
            except Exception as e:
                print(f"DEBUG: {domain} - HTTPS request failed: {e}")
                # Continue to socket test
            
            # Method 2: Socket connection test with SSL
            try:
                context = ssl.create_default_context()
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
                
                with socket.create_connection((domain, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                        if cert:
                            print(f"DEBUG: {domain} - Valid SSL certificate found")
                            return 1
                        else:
                            print(f"DEBUG: {domain} - No valid SSL certificate")
                            return 0
            except ssl.SSLError as e:
                print(f"DEBUG: {domain} - SSL error: {e}")
                return 0
            except socket.gaierror as e:
                print(f"DEBUG: {domain} - DNS resolution failed: {e}")
                return 0
            except Exception as e:
                print(f"DEBUG: {domain} - Socket connection failed: {e}")
                return 0
                
        except Exception as e:
            print(f"DEBUG: {domain} - HTTPS test completely failed: {e}")
            return 0
    
    def extract_domain_features(self, domain: str) -> Dict[str, Any]:
        """Extract domain-based features using WHOIS and DNS."""
        features = {}
        
        try:
            # Remove protocol if present
            if '://' in domain:
                domain = urlparse(domain).netloc
            # WHOIS features
            whois_info = self._get_whois_info(domain)
            features.update(whois_info)
            
            # DNS features
            dns_info = self._get_dns_info(domain)
            features.update(dns_info)
            
            # SSL certificate features
            ssl_info = self._get_ssl_info(domain)
            features.update(ssl_info)
            
        except Exception as e:
            print(f"Error extracting domain features for {domain}: {e}")
            features.update(self._get_default_domain_features())
            
        return features
    
    def extract_content_features(self, url: str) -> Dict[str, Any]:
        """Extract content-based features from webpage."""
        features = {}
        
        try:
            # Try HTTPS first, then HTTP
            original_url = url
            if not url.startswith(('http://', 'https://')):
                # Try HTTPS first for better security detection
                url = 'https://' + url
                
            headers = {'User-Agent': self.user_agent}
            
            # First try with the URL as provided or with HTTPS
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout, verify=False)
            except:
                # If HTTPS fails and we added it, try HTTP
                if url.startswith('https://') and not original_url.startswith(('http://', 'https://')):
                    url = 'http://' + original_url
                    response = requests.get(url, headers=headers, timeout=self.timeout, verify=False)
                else:
                    raise
            
            # Record the actual protocol used
            features['actual_protocol'] = 'https' if url.startswith('https://') else 'http'
            response = requests.get(url, headers=headers, timeout=self.timeout, verify=False)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Basic content metrics
                features['status_code'] = response.status_code
                features['content_length'] = len(response.content)
                features['has_title'] = 1 if soup.title and soup.title.string else 0
                features['title_length'] = len(soup.title.string) if soup.title and soup.title.string else 0
                
                # Form analysis
                forms = soup.find_all('form')
                features['form_count'] = len(forms)
                features['has_password_field'] = self._has_password_field(soup)
                features['has_hidden_fields'] = self._has_hidden_fields(soup)
                
                # Link analysis
                links = soup.find_all('a', href=True)
                features['total_links'] = len(links)
                features['external_links'] = self._count_external_links(links, url)
                
                # Image analysis
                images = soup.find_all('img')
                features['image_count'] = len(images)
                features['external_images'] = self._count_external_images(images, url)
                
                # JavaScript analysis
                scripts = soup.find_all('script')
                features['script_count'] = len(scripts)
                features['has_external_scripts'] = self._has_external_scripts(scripts, url)
                
                # Suspicious content patterns
                features['has_suspicious_content'] = self._has_suspicious_content(soup)
                features['has_copyright'] = 1 if 'copyright' in response.text.lower() else 0
            
                # Redirect analysis
                features['redirect_count'] = len(response.history)
                
                # Sitemap analysis
                sitemap_features = self._extract_sitemap_features(url)
                features.update(sitemap_features)
                
                
            else:
                features = self._get_default_content_features()
                features['status_code'] = response.status_code
                
        except Exception as e:
            print(f"Error extracting content features for {url}: {e}")
            features = self._get_default_content_features()

        print("testing bla bla bla", features)
        return features
    
    def extract_all_features(self, url: str) -> Dict[str, Any]:
        """Extract all features for a given URL."""
        print(f"Extracting features for: {url}")
        
        all_features = {'url': url}
        
        # Extract URL features
        url_features = self.extract_url_features(url)
        all_features.update({f'url_{k}': v for k, v in url_features.items() if k != 'domain_parts'})
        
        # Add domain parts separately
        if 'domain_parts' in url_features:
            all_features.update(url_features['domain_parts'])
        
        # Extract domain features
        domain_features = self.extract_domain_features(url)
        all_features.update({f'domain_{k}': v for k, v in domain_features.items()})
        
        # Extract content features
        content_features = self.extract_content_features(url)
        all_features.update({f'content_{k}': v for k, v in content_features.items()})
        
        return all_features
    
    # Helper methods
    def _has_ip_address(self, domain: str) -> int:
        """Check if domain uses IP address instead of domain name."""
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        return 1 if ip_pattern.match(domain) else 0
    
    def _has_suspicious_tld(self, domain: str) -> int:
        """Check for suspicious top-level domains."""
        suspicious_tlds = ['.tk', '.cf', '.ga', '.ml', '.click', '.download', '.zip']
        return 1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0
    
    def _has_suspicious_keywords(self, url: str) -> int:
        """Check for suspicious keywords in URL."""
        suspicious_keywords = [
            'secure', 'account', 'update', 'login', 'verify', 'bank', 'paypal', 
            'amazon', 'apple', 'microsoft', 'google', 'confirm', 'suspended'
        ]
        url_lower = url.lower()
        return 1 if any(keyword in url_lower for keyword in suspicious_keywords) else 0
    
    def _is_shortened_url(self, domain: str) -> int:
        """Check if URL uses a URL shortening service."""
        shortening_services = [
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'short.link',
            'ow.ly', 'buff.ly', 'adf.ly', 'tiny.cc'
        ]
        return 1 if any(service in domain for service in shortening_services) else 0
    
    def _has_suspicious_port(self, port: Optional[int]) -> int:
        """Check for suspicious ports."""
        if port is None:
            return 0
        suspicious_ports = [8080, 8081, 8888, 3000, 3001, 4000, 5000]
        return 1 if port in suspicious_ports else 0
    
    def _calculate_digit_ratio(self, text: str) -> float:
        """Calculate ratio of digits in text."""
        if not text:
            return 0.0
        digits = sum(1 for char in text if char.isdigit())
        return digits / len(text)
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters in text."""
        if not text:
            return 0.0
        special_chars = sum(1 for char in text if not char.isalnum() and char != '.')
        return special_chars / len(text)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_len = len(text)
        for count in freq.values():
            p = count / text_len
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _get_whois_info(self, domain: str) -> Dict[str, Any]:
        """Get WHOIS information for domain."""
        features = {}
        try:
            w = whois.whois(domain)
            
            print(w)
            
            # Domain age
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            if creation_date:
                age_days = (datetime.datetime.now() - creation_date).days
                features['domain_age_days'] = age_days
                features['is_new_domain'] = 1 if age_days < 365 else 0
            else:
                features['domain_age_days'] = -1
                features['is_new_domain'] = 1
            
            # Expiration date
            expiration_date = w.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            
            if expiration_date:
                days_to_expire = (expiration_date - datetime.datetime.now()).days
                features['days_to_expire'] = days_to_expire
                features['expires_soon'] = 1 if days_to_expire < 90 else 0
            else:
                features['days_to_expire'] = -1
                features['expires_soon'] = 1
            
            # Registration info
            features['has_registrar'] = 1 if w.registrar else 0
            features['has_registrant'] = 1 if w.registrant else 0
            
        except Exception:
            features.update(self._get_default_whois_features())
            
        return features
    
    def _get_dns_info(self, domain: str) -> Dict[str, Any]:
        """Get DNS information for domain."""
        features = {}
        try:
            # A record count
            a_records = socket.getaddrinfo(domain, None)
            print(a_records)
            features['a_record_count'] = len(set(result[4][0] for result in a_records))
            
            # Check if domain resolves
            features['dns_resolves'] = 1
            
        except Exception:
            features['a_record_count'] = 0
            features['dns_resolves'] = 0
            
        return features
    
    def _get_ssl_info(self, domain: str) -> Dict[str, Any]:
        """Get SSL certificate information."""
        features = {}
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    
                    # Certificate validity
                    not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_to_ssl_expire = (not_after - datetime.datetime.now()).days
                    
                    features['has_ssl'] = 1
                    features['ssl_days_to_expire'] = days_to_ssl_expire
                    features['ssl_expires_soon'] = 1 if days_to_ssl_expire < 30 else 0
                    
                    # Certificate issuer
                    issuer = dict(x[0] for x in cert['issuer'])
                    features['ssl_issuer'] = issuer.get('organizationName', 'Unknown')
                    
        except Exception:
            features['has_ssl'] = 0
            features['ssl_days_to_expire'] = -1
            features['ssl_expires_soon'] = 1
            features['ssl_issuer'] = 'None'
            
        return features
    
    def _has_password_field(self, soup: BeautifulSoup) -> int:
        """Check if page has password input fields."""
        password_fields = soup.find_all('input', {'type': 'password'})
        return 1 if password_fields else 0
    
    def _has_hidden_fields(self, soup: BeautifulSoup) -> int:
        """Check if page has hidden input fields."""
        hidden_fields = soup.find_all('input', {'type': 'hidden'})
        return 1 if hidden_fields else 0
    
    def _count_external_links(self, links: List, base_url: str) -> int:
        """Count external links on the page."""
        base_domain = urlparse(base_url).netloc
        external_count = 0
        
        for link in links:
            href = link.get('href', '')
            if href.startswith('http') and base_domain not in href:
                external_count += 1
                
        return external_count
    
    def _count_external_images(self, images: List, base_url: str) -> int:
        """Count external images on the page."""
        base_domain = urlparse(base_url).netloc
        external_count = 0
        
        for img in images:
            src = img.get('src', '')
            if src.startswith('http') and base_domain not in src:
                external_count += 1
                
        return external_count
    
    def _has_external_scripts(self, scripts: List, base_url: str) -> int:
        """Check if page loads external scripts."""
        base_domain = urlparse(base_url).netloc
        
        for script in scripts:
            src = script.get('src', '')
            if src.startswith('http') and base_domain not in src:
                return 1
                
        return 0
    
    def _has_suspicious_content(self, soup: BeautifulSoup) -> int:
        """Check for suspicious content patterns."""
        text = soup.get_text().lower()
        suspicious_phrases = [
            'verify your account', 'suspended account', 'click here immediately',
            'limited time offer', 'act now', 'urgent action required',
            'confirm your identity', 'update payment information'
        ]
        
        return 1 if any(phrase in text for phrase in suspicious_phrases) else 0
    
    def _get_sitemap_urls(self, base_url: str) -> List[str]:
        """
        Try to find sitemap URLs from robots.txt and default locations.
        """
        sitemap_urls = []
        
        # Try robots.txt first
        try:
            robots_url = urljoin(base_url, "/robots.txt")
            headers = {'User-Agent': self.user_agent}
            r = requests.get(robots_url, headers=headers, timeout=5)
            
            if r.status_code == 200:
                matches = re.findall(r"Sitemap:\s*(\S+)", r.text, re.IGNORECASE)
                sitemap_urls.extend(matches)
        except Exception as e:
            print(f"Error fetching robots.txt from {base_url}: {e}")
        
        # If no sitemap found in robots.txt, try default locations
        if not sitemap_urls:
            default_sitemaps = [
                urljoin(base_url, "/sitemap.xml"),
                urljoin(base_url, "/sitemap_index.xml"),
                urljoin(base_url, "/sitemaps.xml")
            ]
            sitemap_urls.extend(default_sitemaps)
        
        return sitemap_urls
    
    def _extract_sitemap_features(self, url: str) -> Dict[str, Any]:
        """
        Extract features from sitemap analysis.
        """
        features = {
            'has_sitemap': 0,
            'sitemap_url_count': 0,
            'sitemap_structure_complexity': 0,
            'has_sitemap_index': 0,
            'sitemap_last_modified': 0
        }
        
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            sitemap_urls = self._get_sitemap_urls(url)
            
            total_urls = 0
            has_any_sitemap = False
            has_index = False
            
            for sitemap_url in sitemap_urls:
                try:
                    headers = {'User-Agent': self.user_agent}
                    r = requests.get(sitemap_url, headers=headers, timeout=5)
                    
                    if r.status_code == 200:
                        has_any_sitemap = True
                        
                        # Try to parse as XML
                        try:
                            root = ET.fromstring(r.content)
                            
                            # Check if it's a sitemap index
                            if 'sitemapindex' in root.tag.lower():
                                has_index = True
                                # Count sitemap references
                                namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                                sitemaps = root.findall(".//ns:sitemap", namespace)
                                total_urls += len(sitemaps)
                            else:
                                # Regular sitemap - count URLs
                                namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                                urls_in_sitemap = root.findall(".//ns:url", namespace)
                                total_urls += len(urls_in_sitemap)
                                
                                # Check for lastmod to see if it's maintained
                                lastmod_entries = root.findall(".//ns:lastmod", namespace)
                                if lastmod_entries:
                                    features['sitemap_last_modified'] = 1
                                    
                        except ET.ParseError:
                            # Not valid XML, might still be accessible
                            has_any_sitemap = True
                            
                except Exception as e:
                    continue  # Try next sitemap URL
            
            features['has_sitemap'] = 1 if has_any_sitemap else 0
            features['sitemap_url_count'] = total_urls
            features['has_sitemap_index'] = 1 if has_index else 0
            
            # Calculate complexity score based on URL count
            if total_urls > 1000:
                features['sitemap_structure_complexity'] = 3  # High
            elif total_urls > 100:
                features['sitemap_structure_complexity'] = 2  # Medium
            elif total_urls > 0:
                features['sitemap_structure_complexity'] = 1  # Low
            else:
                features['sitemap_structure_complexity'] = 0  # None
                
        except Exception as e:
            print(f"Error extracting sitemap features for {url}: {e}")
            # Keep default values
            
        return features
    
    # Default feature dictionaries for error cases
    def _get_default_url_features(self) -> Dict[str, Any]:
        """Get default URL features when extraction fails."""
        return {
            'url_length': 0, 'domain_length': 0, 'path_length': 0, 'query_length': 0,
            'has_ip_address': 0, 'has_suspicious_tld': 0, 'subdomain_count': 0,
            'dash_count': 0, 'dot_count': 0, 'slash_count': 0, 'at_symbol': 0,
            'double_slash_redirecting': 0, 'has_suspicious_keywords': 0,
            'has_url_shortening': 0, 'has_suspicious_port': 0, 'digit_ratio': 0.0,
            'special_char_ratio': 0.0, 'entropy': 0.0, 'uses_https': 0
        }
    
    def _get_default_domain_features(self) -> Dict[str, Any]:
        """Get default domain features when extraction fails."""
        return self._get_default_whois_features()
    
    def _get_default_whois_features(self) -> Dict[str, Any]:
        """Get default WHOIS features when extraction fails."""
        return {
            'domain_age_days': -1, 'is_new_domain': 1, 'days_to_expire': -1,
            'expires_soon': 1, 'has_registrar': 0, 'has_registrant': 0,
            'a_record_count': 0, 'dns_resolves': 0, 'has_ssl': 0,
            'ssl_days_to_expire': -1, 'ssl_expires_soon': 1, 'ssl_issuer': 'None'
        }
    
    def _get_default_content_features(self) -> Dict[str, Any]:
        """Get default content features when extraction fails."""
        return {
            'status_code': -1, 'content_length': 0, 'has_title': 0, 'title_length': 0,
            'form_count': 0, 'has_password_field': 0, 'has_hidden_fields': 0,
            'total_links': 0, 'external_links': 0, 'image_count': 0, 'external_images': 0,
            'script_count': 0, 'has_external_scripts': 0, 'has_suspicious_content': 0,
            'has_copyright': 0, 'redirect_count': 0,
            'has_sitemap': 0, 'sitemap_url_count': 0, 'sitemap_structure_complexity': 0,
            'has_sitemap_index': 0, 'sitemap_last_modified': 0
        }


def process_training_dataset(output_file: str = 'phishing_features_training.csv'):
    """Process the training dataset and extract features."""
    training_file = "/home/vk/phishing/phishing/PS02_Training_set/PS02_Training_set/PS02_Training_set.xlsx"
    
    if not os.path.exists(training_file):
        print(f"Training file not found: {training_file}")
        return
    
    print("Loading training dataset...")
    df = pd.read_excel(training_file)
    
    extractor = PhishingFeatureExtractor()
    features_list = []
    
    print(f"Processing {len(df)} URLs...")
    
    for idx, row in df.iterrows():
        url = row['Identified Phishing/Suspected Domain Name']
        label = row['Phishing/Suspected Domains (i.e. Class Label)']
        cse_name = row['Critical Sector Entity Name']
        cse_domain = row['Corresponding CSE Domain Name']
        
        print(f"Processing {idx+1}/{len(df)}: {url}")
        
        try:
            # Extract features
            features = extractor.extract_all_features(url)
            
            # Add metadata
            features['label'] = label
            features['cse_name'] = cse_name
            features['cse_domain'] = cse_domain
            features['original_url'] = url
            
            features_list.append(features)
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue
        
        # Add delay to avoid overwhelming servers
        time.sleep(1)
    
    # Create DataFrame and save
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to: {output_file}")
    print(f"Extracted features for {len(features_df)} URLs")
    
    return features_df


def process_single_url(url: str):
    """Process a single URL and return features."""
    extractor = PhishingFeatureExtractor()
    features = extractor.extract_all_features(url)
    
    # Convert to DataFrame for better display
    features_df = pd.DataFrame([features])
    return features_df


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Phishing Detection Feature Extractor')
    parser.add_argument('--url', type=str, help='Single URL to analyze')
    parser.add_argument('--file', type=str, help='File containing URLs to analyze')
    parser.add_argument('--batch', action='store_true', help='Process training dataset')
    parser.add_argument('--output', type=str, default='phishing_features.csv', 
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.batch:
        print("Processing training dataset...")
        features_df = process_training_dataset(args.output)
        
    elif args.url:
        print(f"Processing single URL: {args.url}")
        features_df = process_single_url(args.url)
        print("\nExtracted Features:")
        print(features_df.to_string())
        
    elif args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
            
        print(f"Processing URLs from file: {args.file}")
        extractor = PhishingFeatureExtractor()
        features_list = []
        
        with open(args.file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        for i, url in enumerate(urls, 1):
            print(f"Processing {i}/{len(urls)}: {url}")
            try:
                features = extractor.extract_all_features(url)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {url}: {e}")
            time.sleep(1)
        
        features_df = pd.DataFrame(features_list)
        features_df.to_csv(args.output, index=False)
        print(f"Features saved to: {args.output}")
        
    else:
        print("Please specify --url, --file, or --batch option")
        parser.print_help()


if __name__ == "__main__":
    main()
