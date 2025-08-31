#!/usr/bin/env python3
"""
Critical Sector Entities (CSE) Manager
=====================================

This module manages the CSE whitelist and provides functionality to map
phishing domains to their target CSEs. It's designed to be modular and
expandable to handle any number of CSEs.
"""

import json
import pandas as pd
import re
from typing import Dict, List, Set, Optional, Tuple
from difflib import SequenceMatcher
import tldextract
from fuzzywuzzy import fuzz, process
import os


class CSEManager:
    """Manages Critical Sector Entities and their whitelisted domains."""
    
    def __init__(self, cse_data_file: str = "cse_whitelist.json"):
        """Initialize CSE Manager with data file."""
        self.cse_data_file = cse_data_file
        self.cse_data = {}
        self.domain_to_cse = {}
        self.load_cse_data()
    
    def load_cse_data(self):
        """Load CSE data from file or create default data."""
        if os.path.exists(self.cse_data_file):
            try:
                with open(self.cse_data_file, 'r') as f:
                    self.cse_data = json.load(f)
                self._build_domain_mapping()
                print(f"✅ Loaded CSE data for {len(self.cse_data)} entities")
            except Exception as e:
                print(f"❌ Error loading CSE data: {e}")
                self._create_default_data()
        else:
            self._create_default_data()
    
    def _create_default_data(self):
        """Create default CSE data based on the provided information."""
        default_cses = {
            "State Bank of India (SBI)": {
                "sector": "BFSI",
                "whitelisted_domains": [
                    "onlinesbi.sbi",
                    "sbi.co.in",
                    "sbicard.com",
                    "yonobusiness.sbi",
                    "sbiepay.sbi",
                    "sbilife.co.in"
                ],
                "keywords": ["sbi", "state bank", "onlinesbi", "yono", "sbicard"],
                "description": "State Bank of India - Largest public sector bank"
            },
            "ICICI Bank": {
                "sector": "BFSI",
                "whitelisted_domains": [
                    "icicibank.com",
                    "icicidirect.com",
                    "icici.com",
                    "icicilombard.com",
                    "iciciprulife.com"
                ],
                "keywords": ["icici", "icicidirect", "icicilombard", "icicipru"],
                "description": "ICICI Bank - Private sector bank"
            },
            "HDFC Bank": {
                "sector": "BFSI",
                "whitelisted_domains": [
                    "hdfcbank.com",
                    "hdfc.com",
                    "hdfcergo.com",
                    "hdfclife.com"
                ],
                "keywords": ["hdfc", "hdfcbank", "hdfcergo", "hdfclife"],
                "description": "HDFC Bank - Private sector bank"
            },
            "Punjab National Bank (PNB)": {
                "sector": "BFSI",
                "whitelisted_domains": [
                    "pnbindia.in",
                    "netpnb.com"
                ],
                "keywords": ["pnb", "punjab national", "pnbindia"],
                "description": "Punjab National Bank - Public sector bank"
            },
            "Bank of Baroda (BoB)": {
                "sector": "BFSI",
                "whitelisted_domains": [
                    "bankofbaroda.in",
                    "bobibanking.com"
                ],
                "keywords": ["baroda", "bob", "bankofbaroda", "bobibanking"],
                "description": "Bank of Baroda - Public sector bank"
            },
            "National Informatics Centre (NIC)": {
                "sector": "Government",
                "whitelisted_domains": [
                    "nic.gov.in",
                    "email.gov.in",
                    "kavach.mail.gov.in",
                    "accounts.mgovcloud.in"
                ],
                "keywords": ["nic", "gov.in", "mgovcloud", "kavach"],
                "description": "National Informatics Centre - Government IT services"
            },
            "Registrar General and Census Commissioner of India (RGCCI)": {
                "sector": "Government",
                "whitelisted_domains": [
                    "dc.crsorgi.gov.in"
                ],
                "keywords": ["census", "crsorgi", "registrar"],
                "description": "Census and registration services"
            },
            "Indian Railway Catering and Tourism Corporation (IRCTC)": {
                "sector": "Transport",
                "whitelisted_domains": [
                    "irctc.co.in",
                    "irctc.com"
                ],
                "keywords": ["irctc", "railway", "train booking"],
                "description": "Indian Railway booking and catering"
            },
            "Airtel": {
                "sector": "Telecom",
                "whitelisted_domains": [
                    "airtel.in",
                    "airtel.com"
                ],
                "keywords": ["airtel", "bharti airtel"],
                "description": "Airtel - Telecommunications company"
            },
            "Indian Oil Corporation Limited (IOCL)": {
                "sector": "P&E",
                "whitelisted_domains": [
                    "iocl.com"
                ],
                "keywords": ["iocl", "indian oil", "petrol", "fuel"],
                "description": "Indian Oil Corporation - Oil and gas company"
            }
        }
        
        self.cse_data = default_cses
        self._build_domain_mapping()
        self.save_cse_data()
        print(f"✅ Created default CSE data for {len(self.cse_data)} entities")
    
    def _build_domain_mapping(self):
        """Build reverse mapping from domains to CSEs."""
        self.domain_to_cse = {}
        for cse_name, cse_info in self.cse_data.items():
            for domain in cse_info['whitelisted_domains']:
                self.domain_to_cse[domain.lower()] = cse_name
    
    def save_cse_data(self):
        """Save CSE data to file."""
        try:
            with open(self.cse_data_file, 'w') as f:
                json.dump(self.cse_data, f, indent=2)
            print(f"💾 CSE data saved to {self.cse_data_file}")
        except Exception as e:
            print(f"❌ Error saving CSE data: {e}")
    
    def add_cse(self, name: str, sector: str, domains: List[str], 
                keywords: List[str] = None, description: str = "") -> bool:
        """Add a new CSE to the system."""
        try:
            if keywords is None:
                keywords = [name.lower()]
            
            self.cse_data[name] = {
                "sector": sector,
                "whitelisted_domains": domains,
                "keywords": keywords,
                "description": description
            }
            
            # Update domain mapping
            for domain in domains:
                self.domain_to_cse[domain.lower()] = name
            
            self.save_cse_data()
            print(f"✅ Added CSE: {name}")
            return True
        except Exception as e:
            print(f"❌ Error adding CSE {name}: {e}")
            return False
    
    def remove_cse(self, name: str) -> bool:
        """Remove a CSE from the system."""
        try:
            if name not in self.cse_data:
                print(f"❌ CSE {name} not found")
                return False
            
            # Remove domain mappings
            cse_info = self.cse_data[name]
            for domain in cse_info['whitelisted_domains']:
                self.domain_to_cse.pop(domain.lower(), None)
            
            # Remove CSE data
            del self.cse_data[name]
            
            self.save_cse_data()
            print(f"✅ Removed CSE: {name}")
            return True
        except Exception as e:
            print(f"❌ Error removing CSE {name}: {e}")
            return False
    
    def calculate_similarity(self, domain: str, target_cse: str) -> Dict:
        """
        Calculate similarity between a domain and a target CSE.
        
        Args:
            domain: Domain to analyze
            target_cse: Target CSE name
            
        Returns:
            Dictionary with similarity score and details
        """
        if target_cse not in self.cse_data:
            return {
                'similarity_score': 0,
                'target_cse': target_cse,
                'matched_domain': None,
                'reasoning': f'Target CSE "{target_cse}" not found'
            }
        
        cse_info = self.cse_data[target_cse]
        domain_lower = domain.lower().strip()
        
        # Extract domain parts
        extracted = tldextract.extract(domain_lower)
        domain_base = extracted.domain
        
        max_score = 0
        best_match_domain = ""
        best_reasoning = ""
        
        # Check against whitelisted domains
        for white_domain in cse_info['whitelisted_domains']:
            white_extracted = tldextract.extract(white_domain)
            white_base = white_extracted.domain
            
            # Calculate similarity scores
            scores = []
            
            # 1. Fuzzy string matching
            fuzzy_score = fuzz.ratio(domain_base, white_base) / 100.0
            scores.append(('fuzzy', fuzzy_score))
            
            # 2. Sequence matcher
            seq_score = SequenceMatcher(None, domain_base, white_base).ratio()
            scores.append(('sequence', seq_score))
            
            # 3. Substring matching
            if white_base in domain_base or domain_base in white_base:
                substring_score = 0.8
            else:
                substring_score = 0
            scores.append(('substring', substring_score))
            
            # 4. Character substitution patterns
            substitution_score = self._check_character_substitutions(domain_base, white_base)
            scores.append(('substitution', substitution_score))
            
            # 5. Keyword matching
            keyword_score = self._check_keyword_matching(domain_lower, cse_info['keywords'])
            scores.append(('keyword', keyword_score))
            
            # Calculate weighted average
            weights = [0.25, 0.25, 0.2, 0.2, 0.1]
            total_score = sum(score * weight for (_, score), weight in zip(scores, weights))
            
            if total_score > max_score:
                max_score = total_score
                best_match_domain = white_domain
                best_reasoning = f"Best match with {white_domain} (fuzzy: {fuzzy_score:.2f}, seq: {seq_score:.2f})"
        
        return {
            'similarity_score': max_score,
            'target_cse': target_cse,
            'matched_domain': best_match_domain,
            'reasoning': best_reasoning,
            'risk_level': self.classify_domain_risk(domain, max_score)
        }

    def map_domain_to_cse(self, domain: str) -> Optional[Dict]:
        """
        Map a domain to the most likely target CSE.
        
        Args:
            domain: Domain to analyze
            
        Returns:
            Dictionary with mapping information or None
        """
        target_cse, score, reasoning = self.identify_target_cse(domain)
        
        if target_cse:
            return {
                'target_cse': target_cse,
                'similarity_score': score,
                'reasoning': reasoning,
                'risk_level': self.classify_domain_risk(domain, score),
                'cse_sector': self.cse_data[target_cse]['sector']
            }
        
        return None

    def get_all_cses(self) -> Dict:
        """Get all CSE data."""
        return self.cse_data
    
    def get_cse_by_name(self, name: str) -> Optional[Dict]:
        """Get CSE data by name."""
        return self.cse_data.get(name)
    
    def get_whitelisted_domains(self, cse_name: str = None) -> List[str]:
        """Get whitelisted domains for a specific CSE or all CSEs."""
        if cse_name:
            cse_info = self.cse_data.get(cse_name)
            return cse_info['whitelisted_domains'] if cse_info else []
        else:
            all_domains = []
            for cse_info in self.cse_data.values():
                all_domains.extend(cse_info['whitelisted_domains'])
            return all_domains
    
    def identify_target_cse(self, suspicious_domain: str) -> Tuple[Optional[str], float, str]:
        """
        Identify which CSE a suspicious domain might be targeting.
        
        Returns:
            (target_cse_name, similarity_score, reasoning)
        """
        suspicious_domain = suspicious_domain.lower().strip()
        
        # Extract domain parts
        extracted = tldextract.extract(suspicious_domain)
        suspicious_base = extracted.domain
        
        best_match = None
        best_score = 0
        best_reasoning = ""
        
        for cse_name, cse_info in self.cse_data.items():
            max_score_for_cse = 0
            best_match_domain = ""
            
            # Check against whitelisted domains
            for white_domain in cse_info['whitelisted_domains']:
                white_extracted = tldextract.extract(white_domain)
                white_base = white_extracted.domain
                
                # Calculate similarity scores
                scores = []
                
                # 1. Fuzzy string matching
                fuzzy_score = fuzz.ratio(suspicious_base, white_base) / 100.0
                scores.append(('fuzzy', fuzzy_score))
                
                # 2. Sequence matcher
                seq_score = SequenceMatcher(None, suspicious_base, white_base).ratio()
                scores.append(('sequence', seq_score))
                
                # 3. Substring matching
                if white_base in suspicious_base or suspicious_base in white_base:
                    substring_score = 0.8
                else:
                    substring_score = 0
                scores.append(('substring', substring_score))
                
                # 4. Character substitution patterns (common phishing techniques)
                substitution_score = self._check_character_substitutions(suspicious_base, white_base)
                scores.append(('substitution', substitution_score))
                
                # 5. Keyword matching
                keyword_score = self._check_keyword_matching(suspicious_domain, cse_info['keywords'])
                scores.append(('keyword', keyword_score))
                
                # Calculate weighted average
                weights = [0.25, 0.25, 0.2, 0.2, 0.1]
                total_score = sum(score * weight for (_, score), weight in zip(scores, weights))
                
                if total_score > max_score_for_cse:
                    max_score_for_cse = total_score
                    best_match_domain = white_domain
            
            if max_score_for_cse > best_score:
                best_score = max_score_for_cse
                best_match = cse_name
                best_reasoning = f"Matches {best_match_domain} with {best_score:.2f} similarity"
        
        # Only return matches above threshold
        if best_score > 0.3:  # Adjustable threshold
            return best_match, best_score, best_reasoning
        else:
            return None, 0, "No significant similarity found"
    
    def _check_character_substitutions(self, suspicious: str, legitimate: str) -> float:
        """Check for common character substitutions used in phishing."""
        # Common substitutions: 0->o, 1->l, 3->e, 5->s, etc.
        substitution_map = {
            '0': 'o', 'o': '0',
            '1': 'l', 'l': '1', 'i': '1',
            '3': 'e', 'e': '3',
            '5': 's', 's': '5',
            '6': 'g', 'g': '6',
            '8': 'b', 'b': '8',
            'rn': 'm', 'm': 'rn',
            'vv': 'w', 'w': 'vv'
        }
        
        # Create variations of the legitimate domain
        variations = [legitimate]
        for old_char, new_char in substitution_map.items():
            variations.extend([var.replace(old_char, new_char) for var in variations])
        
        # Check if suspicious domain matches any variation
        for variation in variations:
            if suspicious == variation:
                return 0.9
            elif SequenceMatcher(None, suspicious, variation).ratio() > 0.8:
                return 0.7
        
        return 0
    
    def _check_keyword_matching(self, suspicious_domain: str, keywords: List[str]) -> float:
        """Check if suspicious domain contains CSE keywords."""
        suspicious_lower = suspicious_domain.lower()
        
        for keyword in keywords:
            if keyword.lower() in suspicious_lower:
                return 0.8
        
        return 0
    
    def classify_domain_risk(self, domain: str, similarity_score: float) -> str:
        """
        Classify domain risk based on similarity score and other factors.
        
        Returns: 'PHISHING', 'SUSPECTED', 'LOW_RISK'
        """
        if similarity_score >= 0.7:
            return 'PHISHING'
        elif similarity_score >= 0.4:
            return 'SUSPECTED'
        else:
            return 'LOW_RISK'
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded CSE data."""
        total_domains = sum(len(cse['whitelisted_domains']) for cse in self.cse_data.values())
        sectors = set(cse['sector'] for cse in self.cse_data.values())
        
        return {
            'total_cses': len(self.cse_data),
            'total_whitelisted_domains': total_domains,
            'sectors': list(sectors),
            'cses_by_sector': {
                sector: [name for name, data in self.cse_data.items() if data['sector'] == sector]
                for sector in sectors
            }
        }


def main():
    """Demo function to test CSE Manager."""
    print("🏢 CSE Manager Demo")
    print("==================")
    
    cse_manager = CSEManager()
    
    # Show statistics
    stats = cse_manager.get_statistics()
    print(f"\n📊 CSE Statistics:")
    print(f"  • Total CSEs: {stats['total_cses']}")
    print(f"  • Total Whitelisted Domains: {stats['total_whitelisted_domains']}")
    print(f"  • Sectors: {', '.join(stats['sectors'])}")
    
    # Test domain matching
    test_domains = [
        "sb1.co.in",           # SBI lookalike
        "hdfcbank.co.in",      # HDFC lookalike
        "icicinank.com",       # ICICI typosquatting
        "a1rtel.com",          # Airtel lookalike
        "irctc.net",           # IRCTC domain variation
        "google.com"           # Unrelated domain
    ]
    
    print(f"\n🔍 Testing Domain Matching:")
    print("=" * 50)
    
    for domain in test_domains:
        target_cse, score, reasoning = cse_manager.identify_target_cse(domain)
        risk_level = cse_manager.classify_domain_risk(domain, score)
        
        print(f"\n🌐 Domain: {domain}")
        if target_cse:
            print(f"  🎯 Target CSE: {target_cse}")
            print(f"  📊 Similarity: {score:.3f}")
            print(f"  🚨 Risk Level: {risk_level}")
            print(f"  💭 Reasoning: {reasoning}")
        else:
            print(f"  ✅ No CSE target identified")


if __name__ == "__main__":
    main()
