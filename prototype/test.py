#!/usr/bin/env python3
"""
TEST.PY - Phishing Detection Using Rule-Based Approach
=======================================================

Based on email.md Annexure A features:
- URL-based features (length, special chars, suspicious patterns)
- Domain analysis (typosquatting, suspicious TLDs)
- CSE targeting detection
- Phishing indicators (login, secure, verify, etc.)

WORKFLOW:
1. Load 1M domains
2. Apply rule-based phishing detection (NOT ML model)
3. Map phishing to CSEs
4. Capture screenshots + WHOIS for CSE-targeted phishing
"""

import pandas as pd
import numpy as np
import warnings
import re
import tldextract
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
import Levenshtein

warnings.filterwarnings('ignore')

# Playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# WHOIS
try:
    import whois as python_whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

# PDF
try:
    from PIL import Image
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    import io
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class RuleBasedPhishingDetector:
    """
    Rule-based phishing detector using features from email.md Annexure A
    Much more effective than the ML model!
    """
    
    def __init__(self):
        # CSE keywords and official domains
        self.cse_data = {
            'State Bank of India (SBI)': {
                'keywords': ['sbi', 'statebank', 'yono', 'onlinesbi', 'sbicard', 'statebankof'],
                'official': ['sbi.co.in', 'onlinesbi.sbi', 'sbicard.com']
            },
            'ICICI Bank': {
                'keywords': ['icici', 'icicibank', 'icicidirect', 'icicipru'],
                'official': ['icicibank.com', 'icicidirect.com']
            },
            'HDFC Bank': {
                'keywords': ['hdfc', 'hdfcbank', 'hdfcergo'],
                'official': ['hdfcbank.com', 'hdfc.com']
            },
            'Punjab National Bank (PNB)': {
                'keywords': ['pnb', 'punjabnational', 'netpnb', 'pnbindia'],
                'official': ['pnbindia.in', 'netpnb.com']
            },
            'Bank of Baroda (BoB)': {
                'keywords': ['baroda', 'bankofbaroda', 'bobibanking'],
                'official': ['bankofbaroda.in', 'bobibanking.com']
            },
            'Airtel': {
                'keywords': ['airtel', 'airtelmoney', 'airtelbank', 'airtelpayments'],
                'official': ['airtel.in', 'airtel.com']
            },
            'National Informatics Centre (NIC)': {
                'keywords': ['nicgov', 'govnic', 'nicindia', 'kavachmail', 'emailgov'],
                'official': ['nic.gov.in', 'email.gov.in']
            },
            'Registrar General and Census Commissioner of India (RGCCI)': {
                'keywords': ['censusindia', 'crsorgi', 'rgcci'],
                'official': ['dc.crsorgi.gov.in']
            },
            'Indian Railway Catering and Tourism Corporation (IRCTC)': {
                'keywords': ['irctc', 'indianrailway'],
                'official': ['irctc.co.in']
            },
            'Indian Oil Corporation Limited (IOCL)': {
                'keywords': ['indianoil', 'iocl'],
                'official': ['iocl.com']
            }
        }
        
        # Suspicious TLDs (from email.md)
        self.suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click', 'link', 'loan']
        
        # Phishing keywords
        self.phishing_keywords = [
            'login', 'signin', 'secure', 'account', 'verify', 'update', 'confirm',
            'banking', 'netbanking', 'payment', 'wallet', 'pay', 'transaction'
        ]
    
    def detect_phishing(self, domain):
        """
        Detect if domain is phishing or suspected using rule-based approach
        Returns: (classification, confidence, cse_name, reasons)
        classification: 'Phishing', 'Suspected', or None
        """
        domain_lower = domain.lower()
        extracted = tldextract.extract(domain)
        domain_part = extracted.domain.lower()
        tld = extracted.suffix.lower()
        subdomain = extracted.subdomain.lower()
        
        reasons = []
        phishing_score = 0
        cse_name = None
        
        # Check if domain targets a CSE
        for cse, data in self.cse_data.items():
            # Check keywords
            for keyword in data['keywords']:
                if len(keyword) >= 4:  # Only check keywords 4+ chars
                    # Exact match or at boundaries
                    if domain_part == keyword:
                        cse_name = cse
                        phishing_score += 30
                        reasons.append(f"Exact match: {keyword}")
                    elif domain_part.startswith(keyword) or domain_part.endswith(keyword):
                        cse_name = cse
                        phishing_score += 25
                        reasons.append(f"Starts/ends with: {keyword}")
                    elif keyword in domain_part and len(domain_part) > len(keyword):
                        cse_name = cse
                        phishing_score += 20
                        reasons.append(f"Contains: {keyword}")
            
            # Check official domain similarity (typosquatting)
            for official in data['official']:
                official_domain = tldextract.extract(official).domain
                official_tld = tldextract.extract(official).suffix
                
                # Not the official domain itself
                if domain_part == official_domain and tld == official_tld:
                    return None, 0, None, []  # Official domain, not phishing
                
                # Typosquatting detection
                similarity = Levenshtein.ratio(domain_part, official_domain)
                if similarity >= 0.75:
                    cse_name = cse
                    phishing_score += int(similarity * 30)
                    reasons.append(f"Typosquatting: {similarity:.2f} similar to {official_domain}")
        
        # If no CSE detected, not phishing for our purposes
        if not cse_name:
            return None, 0, None, []
        
        # Now check phishing indicators (from email.md Annexure A)
        
        # 1. Suspicious TLD
        if tld in self.suspicious_tlds:
            phishing_score += 15
            reasons.append(f"Suspicious TLD: .{tld}")
        
        # 2. Wrong TLD for Indian CSE
        if tld not in ['in', 'co.in', 'com', 'org', 'net']:
            phishing_score += 10
            reasons.append(f"Non-standard TLD for Indian entity: .{tld}")
        
        # 3. Phishing keywords in domain/subdomain
        for keyword in self.phishing_keywords:
            if keyword in domain_lower:
                phishing_score += 5
                reasons.append(f"Phishing keyword: {keyword}")
        
        # 4. Excessive hyphens (from email.md)
        hyphen_count = domain_lower.count('-')
        if hyphen_count >= 2:
            phishing_score += hyphen_count * 3
            reasons.append(f"Excessive hyphens: {hyphen_count}")
        
        # 5. Digits in domain (suspicious for banks)
        digit_count = sum(c.isdigit() for c in domain_part)
        if digit_count >= 2:
            phishing_score += digit_count * 2
            reasons.append(f"Digits in domain: {digit_count}")
        
        # 6. Long domain (from email.md)
        if len(domain_lower) > 30:
            phishing_score += 5
            reasons.append(f"Long domain: {len(domain_lower)} chars")
        
        # 7. Subdomain present (common in phishing)
        if subdomain and subdomain not in ['www', 'web', 'mobile', 'app']:
            phishing_score += 10
            reasons.append(f"Suspicious subdomain: {subdomain}")
        
        # 8. Multiple dots (from email.md)
        dot_count = domain_lower.count('.')
        if dot_count >= 3:
            phishing_score += dot_count * 3
            reasons.append(f"Multiple dots: {dot_count}")
        
        # Determine classification based on score
        # Phishing: 40+ points (high confidence)
        # Suspected: 25-39 points (CSE-related but lower confidence)
        if phishing_score >= 40:
            classification = 'Phishing'
        elif phishing_score >= 25:
            classification = 'Suspected'
        else:
            classification = None
        
        confidence = min(phishing_score / 100.0, 0.99)
        
        return classification, confidence, cse_name, reasons


class PhishingTester:
    """Main tester with rule-based detection"""
    
    def __init__(self, application_id="AIGR-123456"):
        self.application_id = application_id
        self.detector = RuleBasedPhishingDetector()
        
        # Submission folders
        self.submission_folder = f"PS-02_{application_id}_Submission_v3"
        self.evidence_folder = f"{self.submission_folder}/PS-02_{application_id}_Evidences"
        
        # CSE folders
        self.cse_folders = {
            'ICICI Bank': f"{self.evidence_folder}/ICICI",
            'HDFC Bank': f"{self.evidence_folder}/HDFC",
            'State Bank of India (SBI)': f"{self.evidence_folder}/SBI",
            'Punjab National Bank (PNB)': f"{self.evidence_folder}/PNB",
            'Bank of Baroda (BoB)': f"{self.evidence_folder}/BOB",
            'Airtel': f"{self.evidence_folder}/AIRTEL",
            'National Informatics Centre (NIC)': f"{self.evidence_folder}/NIC",
            'Registrar General and Census Commissioner of India (RGCCI)': f"{self.evidence_folder}/RGCCI",
            'Indian Railway Catering and Tourism Corporation (IRCTC)': f"{self.evidence_folder}/IRCTC",
            'Indian Oil Corporation Limited (IOCL)': f"{self.evidence_folder}/IOCL"
        }
        
        for folder in [self.submission_folder, self.evidence_folder]:
            os.makedirs(folder, exist_ok=True)
        
        for cse_folder in self.cse_folders.values():
            os.makedirs(cse_folder, exist_ok=True)
        
        # Browser
        self.playwright = None
        self.browser = None
        
        # Stats
        self.total_domains = 0
        self.phishing_detected = 0
        self.screenshots_captured = 0
        self.whois_collected = 0
        
        print("Using Rule-Based Phishing Detection")
    
    def init_browser(self):
        """Initialize browser"""
        if not PLAYWRIGHT_AVAILABLE:
            return False
        
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            return True
        except Exception as e:
            print(f"   [WARN] Browser init failed: {e}")
            return False
    
    def capture_screenshot(self, domain, cse_name, serial_no):
        """Capture screenshot"""
        if not self.browser:
            return None
        
        try:
            cse_short = self._get_cse_short_name(cse_name)
            extracted = tldextract.extract(domain)
            subdomain = extracted.subdomain if extracted.subdomain else ''
            domain_name = extracted.domain
            tld = extracted.suffix
            
            # Build filename per email.md
            subdomain_parts = subdomain.split('.') if subdomain else []
            if len(subdomain_parts) > 2:
                subdomain_parts = subdomain_parts[-2:]
            subdomain_str = '.'.join(subdomain_parts) if subdomain_parts else domain_name
            
            filename = f"{cse_short}_{subdomain_str}.{domain_name}.{tld}_{serial_no}.pdf"
            filename = filename.replace('..', '.').replace(' ', '_')
            
            cse_folder = self.cse_folders.get(cse_name, self.evidence_folder)
            pdf_path = f"{cse_folder}/{filename}"
            
            # Try both protocols
            screenshot_bytes = None
            for protocol in ['https', 'http']:
                try:
                    url = f"{protocol}://{domain}"
                    page = self.browser.new_page(viewport={"width": 1280, "height": 800})
                    page.goto(url, wait_until="networkidle", timeout=20000)
                    screenshot_bytes = page.screenshot(full_page=True)
                    page.close()
                    break
                except:
                    try:
                        page.close()
                    except:
                        pass
                    continue
            
            if not screenshot_bytes or not PDF_AVAILABLE:
                return None
            
            # Convert to PDF
            img = Image.open(io.BytesIO(screenshot_bytes))
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            page_width, page_height = A4
            img_display_width = page_width - 40
            img_display_height = img_display_width * aspect
            
            c = canvas.Canvas(pdf_path, pagesize=A4)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(20, page_height - 30, f"Phishing Evidence: {domain}")
            c.setFont("Helvetica", 10)
            c.drawString(20, page_height - 50, f"CSE: {cse_name}")
            c.drawString(20, page_height - 65, f"Captured: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            y_position = page_height - 85 - img_display_height
            if y_position < 0:
                img_display_height = page_height - 105
                y_position = 20
            
            c.drawImage(ImageReader(img), 20, y_position, 
                      width=img_display_width, height=img_display_height,
                      preserveAspectRatio=True)
            c.save()
            
            self.screenshots_captured += 1
            return filename
            
        except Exception as e:
            return None
    
    def collect_whois(self, domain):
        """Collect WHOIS with error suppression"""
        whois_data = {}
        
        if not WHOIS_AVAILABLE:
            return whois_data
        
        try:
            result = [None]
            
            def whois_lookup():
                try:
                    # Suppress whois library logging
                    import logging
                    logging.getLogger('whois.whois').setLevel(logging.CRITICAL)
                    result[0] = python_whois.whois(domain)
                except:
                    pass
            
            thread = threading.Thread(target=whois_lookup)
            thread.daemon = True
            thread.start()
            thread.join(timeout=5)
            
            if result[0]:
                w = result[0]
                if hasattr(w, 'creation_date'):
                    whois_data['registration_date'] = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                if hasattr(w, 'registrar'):
                    whois_data['registrar'] = w.registrar[0] if isinstance(w.registrar, list) else w.registrar
                if hasattr(w, 'name'):
                    whois_data['registrant'] = w.name[0] if isinstance(w.name, list) else w.name
                if hasattr(w, 'country'):
                    whois_data['country'] = w.country[0] if isinstance(w.country, list) else w.country
                if hasattr(w, 'name_servers'):
                    whois_data['name_servers'] = ', '.join(w.name_servers) if isinstance(w.name_servers, list) else str(w.name_servers)
                
                self.whois_collected += 1
        except:
            pass
        
        return whois_data
    
    def test_on_shortlisting(self, shortlisting_dir="backend/dataset/PS-02_Shortlisting_set"):
        """Run rule-based phishing detection"""
        
        print("\n[1/5] Loading Shortlisting Dataset...")
        
        files = list(Path(shortlisting_dir).glob("*.xlsx"))
        if not files:
            files = list(Path(shortlisting_dir).glob("*.csv"))
        
        print(f"   Found {len(files)} file(s)")
        
        all_domains = []
        for file in files:
            print(f"   Loading: {file.name}...")
            if file.suffix == '.xlsx':
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
            
            domain_col = df.columns[0]
            domains = df[domain_col].tolist()
            all_domains.extend(domains)
            print(f"   [OK] Loaded: {file.name} ({len(domains):,} domains)")
        
        self.total_domains = len(all_domains)
        print(f"\n   Total domains: {self.total_domains:,}")
        
        print("\n[2/5] Running Rule-Based Phishing Detection...")
        print(f"   Using features from email.md Annexure A")
        
        phishing_domains = []
        start_time = time.time()
        
        for idx, domain in enumerate(all_domains):
            try:
                classification, confidence, cse_name, reasons = self.detector.detect_phishing(domain)
                
                # Include both Phishing and Suspected
                if classification and cse_name:
                    phishing_domains.append({
                        'domain': domain,
                        'classification': classification,
                        'confidence': confidence,
                        'cse_name': cse_name,
                        'reasons': ' | '.join(reasons)
                    })
                    self.phishing_detected += 1
            except:
                continue
            
            if (idx + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (self.total_domains - idx - 1) / rate if rate > 0 else 0
                
                print(f"\r   [{idx+1:,}/{self.total_domains:,}] "
                      f"Phishing: {self.phishing_detected:,} | "
                      f"Rate: {rate:.0f}/s | "
                      f"ETA: {remaining/60:.0f}m", end='', flush=True)
        
        print(f"\n\n   [OK] Phishing Detection Complete!")
        print(f"   Total detected: {self.phishing_detected:,} ({self.phishing_detected/self.total_domains*100:.2f}%)")
        
        # Count by classification
        phishing_count = sum(1 for item in phishing_domains if item['classification'] == 'Phishing')
        suspected_count = sum(1 for item in phishing_domains if item['classification'] == 'Suspected')
        print(f"   - Phishing: {phishing_count:,}")
        print(f"   - Suspected: {suspected_count:,}")
        
        # Save all detections
        phishing_df = pd.DataFrame(phishing_domains)
        phishing_df.to_csv(f"{self.submission_folder}/phishing_detected.csv", index=False)
        print(f"   [OK] Saved: {self.submission_folder}/phishing_detected.csv")
        
        # Show CSE distribution
        cse_counts = {}
        for item in phishing_domains:
            cse = item['cse_name']
            cse_counts[cse] = cse_counts.get(cse, 0) + 1
        
        print(f"\n   CSE Distribution:")
        for cse, count in sorted(cse_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {cse}: {count:,}")
        
        print("\n[3/5] Initializing Browser for Screenshots...")
        
        if self.init_browser():
            print(f"   [OK] Browser ready")
        else:
            print(f"   [WARN] Browser not available")
        
        print("\n[4/5] Capturing Screenshots + WHOIS (Phishing only)...")
        
        # Load existing submission to skip already processed domains
        existing_domains = set()
        existing_excel = f"{self.submission_folder}/PS-02_{self.application_id}_Submission_Set.xlsx"
        if os.path.exists(existing_excel):
            try:
                existing_df = pd.read_excel(existing_excel)
                existing_domains = set(existing_df['Identified_Domain'].tolist())
                print(f"   [INFO] Found {len(existing_domains)} already processed domains, will skip them")
            except:
                pass
        
        # Separate phishing and suspected domains
        phishing_only = [item for item in phishing_domains if item['classification'] == 'Phishing']
        suspected_only = [item for item in phishing_domains if item['classification'] == 'Suspected']
        
        print(f"   [INFO] Processing {len(phishing_only)} Phishing domains for screenshots/WHOIS")
        print(f"   [INFO] Including {len(suspected_only)} Suspected domains in submission (no screenshots)")
        
        results = []
        serial_no = 1
        start_time = time.time()
        skipped = 0
        
        # Process Phishing domains (with screenshots/WHOIS)
        for idx, item in enumerate(phishing_only):
            domain = item['domain']
            cse_name = item['cse_name']
            confidence = item['confidence']
            classification = item['classification']
            
            # Skip if already processed
            if domain in existing_domains:
                skipped += 1
                continue
            
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1 - skipped) / elapsed if elapsed > 0 else 0
                remaining = (len(phishing_only) - idx - 1) / rate if rate > 0 else 0
                
                print(f"\r   [{idx+1}/{len(phishing_only):,}] "
                      f"📸 {self.screenshots_captured} | "
                      f"📋 {self.whois_collected} | "
                      f"⏭ {skipped} | "
                      f"⏱ {remaining/60:.0f}m", end='', flush=True)
            
            # Collect WHOIS
            whois_data = self.collect_whois(domain)
            
            # Capture screenshot
            evidence_file = self.capture_screenshot(domain, cse_name, serial_no)
            
            # Add to results
            results.append({
                'Application_ID': self.application_id,
                'Source_of_detection': 'Rule-Based Detection (email.md Annexure A)',
                'Identified_Domain': domain,
                'Corresponding_CSE_Domain': self._get_official_domain(cse_name),
                'CSE_Name': cse_name,
                'Classification': classification,
                'Registration_Date': str(whois_data.get('registration_date', '')),
                'Registrar_Name': whois_data.get('registrar', ''),
                'Registrant_Name': whois_data.get('registrant', ''),
                'Registrant_Country': whois_data.get('country', ''),
                'Name_Servers': whois_data.get('name_servers', ''),
                'Hosting_IP': '',
                'Hosting_ISP': '',
                'Hosting_Country': '',
                'DNS_Records': '',
                'Evidence_File': evidence_file or '',
                'Date_of_Detection': datetime.now().strftime('%d-%m-%Y'),
                'Time_of_Detection': datetime.now().strftime('%H-%M-%S'),
                'Date_of_Post': '',
                'Remarks': item['reasons']
            })
            
            serial_no += 1
        
        # Add Suspected domains (without screenshots/WHOIS)
        print(f"\n   [INFO] Adding {len(suspected_only)} Suspected domains to submission...")
        for item in suspected_only:
            domain = item['domain']
            cse_name = item['cse_name']
            classification = item['classification']
            
            # Skip if already processed
            if domain in existing_domains:
                continue
            
            # Add to results (no screenshots/WHOIS)
            results.append({
                'Application_ID': self.application_id,
                'Source_of_detection': 'Rule-Based Detection (email.md Annexure A)',
                'Identified_Domain': domain,
                'Corresponding_CSE_Domain': self._get_official_domain(cse_name),
                'CSE_Name': cse_name,
                'Classification': classification,
                'Registration_Date': '',
                'Registrar_Name': '',
                'Registrant_Name': '',
                'Registrant_Country': '',
                'Name_Servers': '',
                'Hosting_IP': '',
                'Hosting_ISP': '',
                'Hosting_Country': '',
                'DNS_Records': '',
                'Evidence_File': '',
                'Date_of_Detection': datetime.now().strftime('%d-%m-%Y'),
                'Time_of_Detection': datetime.now().strftime('%H-%M-%S'),
                'Date_of_Post': '',
                'Remarks': item['reasons']
            })
        
        if skipped > 0:
            print(f"\n   [INFO] Skipped {skipped} already processed domains")
        
        print(f"\n\n[5/5] Generating Submission...")
        print(f"   [OK] Total domains: {self.total_domains:,}")
        print(f"   [OK] Phishing detected: {self.phishing_detected:,} ({self.phishing_detected/self.total_domains*100:.2f}%)")
        print(f"   [OK] Screenshots: {self.screenshots_captured:,}")
        print(f"   [OK] WHOIS: {self.whois_collected:,}")
        
        # Save submission
        self._save_submission(results)
        
        # Cleanup
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        
        print(f"\n" + "="*80)
        print("[OK] TESTING COMPLETE!")
        print("="*80)
        print(f"   Submission: {self.submission_folder}/")
        print("="*80)
    
    def _get_cse_short_name(self, cse_name):
        """Get short name"""
        mapping = {
            'ICICI Bank': 'ICICI',
            'HDFC Bank': 'HDFC',
            'State Bank of India (SBI)': 'SBI',
            'Punjab National Bank (PNB)': 'PNB',
            'Bank of Baroda (BoB)': 'BOB',
            'Airtel': 'AIRTEL',
            'National Informatics Centre (NIC)': 'NIC',
            'Registrar General and Census Commissioner of India (RGCCI)': 'RGCCI',
            'Indian Railway Catering and Tourism Corporation (IRCTC)': 'IRCTC',
            'Indian Oil Corporation Limited (IOCL)': 'IOCL'
        }
        return mapping.get(cse_name, 'UNKNOWN')
    
    def _get_official_domain(self, cse_name):
        """Get official domain"""
        official = {
            'State Bank of India (SBI)': 'sbi.co.in',
            'ICICI Bank': 'icicibank.com',
            'HDFC Bank': 'hdfcbank.com',
            'Punjab National Bank (PNB)': 'pnbindia.in',
            'Bank of Baroda (BoB)': 'bankofbaroda.in',
            'National Informatics Centre (NIC)': 'nic.gov.in',
            'Registrar General and Census Commissioner of India (RGCCI)': 'dc.crsorgi.gov.in',
            'Indian Railway Catering and Tourism Corporation (IRCTC)': 'irctc.co.in',
            'Airtel': 'airtel.in',
            'Indian Oil Corporation Limited (IOCL)': 'iocl.com'
        }
        return official.get(cse_name, '')
    
    def _save_submission(self, results):
        """Save submission"""
        output_excel = f"{self.submission_folder}/PS-02_{self.application_id}_Submission_Set.xlsx"
        
        df = pd.DataFrame(results)
        df.to_excel(output_excel, index=False, sheet_name='Phishing_Domains')
        
        csv_path = output_excel.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"   [OK] Excel: {output_excel}")
        print(f"   [OK] CSV: {csv_path}")


def main():
    """Main"""
    
    print("="*80)
    print("PHISHING DETECTION - RULE-BASED APPROACH")
    print("Using features from email.md Annexure A")
    print("="*80)
    
    tester = PhishingTester(application_id="AIGR-123456")
    tester.test_on_shortlisting()
    
    print("\n" + "="*80)
    print("[OK] COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
