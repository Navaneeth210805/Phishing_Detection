#!/usr/bin/env python3
"""
Comprehensive Phishing Detection System
=======================================

Main orchestrator that integrates all components:
- CSE Management (reuses cse_manager.py)
- Domain Discovery (reuses domain_discovery.py) 
- Feature Extraction (reuses phishing_feature_extractor.py)
- Model Training and Prediction (reuses train_model.py)
- Web Interface (enhances web_app.py)
- Automated Monitoring and Reporting

This system is designed to be modular and expandable for any number of CSEs.
"""

import asyncio
import json
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Import existing modules
from cse_manager import CSEManager
from domain_discovery import DomainDiscoveryEngine
from phishing_feature_extractor import PhishingFeatureExtractor
from train_model import PhishingModelTrainer

class PhishingDetectionSystem:
    """Main system orchestrator for phishing detection targeting CSEs."""
    
    def __init__(self, config_file: str = "system_config.json"):
        """Initialize the comprehensive phishing detection system."""
        self.config_file = config_file
        self.config = self._load_config()
        
        # Initialize components (reusing existing modules)
        self.cse_manager = CSEManager()
        self.discovery_engine = DomainDiscoveryEngine(self.cse_manager)
        self.feature_extractor = PhishingFeatureExtractor()
        self.model_trainer = PhishingModelTrainer()
        
        # System state
        self.monitoring_active = False
        self.detected_domains = []
        self.reports = []
        
        # Configure logging
        self._setup_logging()
        
        # Load or train model
        self.model_data = self._load_or_train_model()
        
    def _load_config(self) -> Dict:
        """Load system configuration."""
        default_config = {
            "monitoring": {
                "enabled": True,
                "interval_hours": 6,
                "max_domains_per_scan": 1000,
                "save_results": True
            },
            "detection": {
                "similarity_threshold": 0.7,
                "confidence_threshold": 0.5,
                "check_content": True,
                "check_ssl": True
            },
            "reporting": {
                "auto_generate": True,
                "report_interval_hours": 24,
                "include_screenshots": False,
                "export_formats": ["json", "csv"]
            },
            "data_sources": {
                "certificate_transparency": True,
                "dns_monitoring": True,
                "typosquatting": True,
                "subdomain_enumeration": True
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            except Exception as e:
                logging.warning(f"Error loading config, using defaults: {e}")
        
        # Save current config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phishing_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        model_file = 'phishing_detection_model.pkl'
        
        if os.path.exists(model_file):
            try:
                import joblib
                model_data = joblib.load(model_file)
                self.logger.info("✅ Loaded existing phishing detection model")
                return model_data
            except Exception as e:
                self.logger.warning(f"Error loading model: {e}")
        
        # Train new model if needed
        self.logger.info("🔄 No model found, will train when data is available")
        return None
    
    def add_cse(self, name: str, sector: str, whitelisted_domains: List[str], 
               keywords: List[str] = None, description: str = "") -> bool:
        """Add a new CSE to the system (expanding modularity)."""
        try:
            return self.cse_manager.add_cse(name, sector, whitelisted_domains, keywords, description)
        except Exception as e:
            self.logger.error(f"Error adding CSE {name}: {e}")
            return False
    
    def remove_cse(self, name: str) -> bool:
        """Remove a CSE from the system."""
        try:
            return self.cse_manager.remove_cse(name)
        except Exception as e:
            self.logger.error(f"Error removing CSE {name}: {e}")
            return False
    
    def get_cse_list(self) -> Dict:
        """Get list of all CSEs in the system."""
        return self.cse_manager.get_all_cses()
    
    async def discover_phishing_domains(self, target_cses: List[str] = None) -> List[Dict]:
        """Discover potential phishing domains for target CSEs."""
        self.logger.info("🔍 Starting domain discovery process...")
        
        if target_cses is None:
            target_cses = list(self.cse_manager.cse_data.keys())
        
        discovered_domains = []
        
        for cse_name in target_cses:
            self.logger.info(f"Scanning for domains targeting: {cse_name}")
            
            try:
                # Use existing discovery engine
                cse_domains = await self.discovery_engine.discover_domains_for_cse(cse_name)
                
                for domain_info in cse_domains:
                    # Enhance with CSE mapping
                    domain_info['target_cse'] = cse_name
                    domain_info['discovery_timestamp'] = datetime.now().isoformat()
                    discovered_domains.append(domain_info)
                    
            except Exception as e:
                self.logger.error(f"Error discovering domains for {cse_name}: {e}")
        
        self.detected_domains.extend(discovered_domains)
        return discovered_domains
    
    def classify_domain(self, domain: str, target_cse: str = None) -> Dict:
        """Classify a domain as Phishing, Suspected, or Legitimate."""
        try:
            # First check if domain is whitelisted
            whitelist_check = self._check_whitelist(domain, target_cse)
            if whitelist_check:
                return whitelist_check
            
            # Extract features using existing extractor
            features = self.feature_extractor.extract_all_features(domain)
            
            # Add CSE-specific features
            if target_cse:
                cse_similarity = self.cse_manager.calculate_similarity(domain, target_cse)
                features['cse_similarity_score'] = cse_similarity['similarity_score']
                features['target_cse'] = target_cse
            
            # Use model for prediction if available
            prediction_result = None
            if self.model_data:
                prediction_result = self._predict_with_model(features)
            
            # Rule-based classification as fallback
            classification = self._rule_based_classification(features, target_cse)
            
            result = {
                'domain': domain,
                'target_cse': target_cse,
                'classification': classification,
                'confidence': classification.get('confidence', 0.5),
                'features': features,
                'ml_prediction': prediction_result,
                'timestamp': datetime.now().isoformat(),
                'evidence': self._collect_evidence(domain, features, target_cse)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying domain {domain}: {e}")
            return {'domain': domain, 'error': str(e)}
    
    def _predict_with_model(self, features: Dict) -> Dict:
        """Use trained ML model for prediction."""
        try:
            # Convert features to DataFrame (reusing logic from web_app.py)
            features_df = pd.DataFrame([features])
            
            # Remove non-feature columns
            exclude_cols = ['url', 'original_url', 'label', 'cse_name', 'cse_domain', 
                           'subdomain', 'domain', 'suffix', 'fqdn', 'domain_ssl_issuer']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            X = features_df[feature_cols].copy()
            
            # Handle missing values
            X = X.fillna(-1)
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col] = X[col].fillna(-1)
                    except:
                        X[col] = 0
            
            # Make prediction
            model = self.model_data['model']
            scaler = self.model_data['scaler']
            label_encoder = self.model_data['label_encoder']
            model_name = self.model_data['model_name']
            
            if model_name in ['LogisticRegression', 'SVM', 'NaiveBayes']:
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0]
            else:
                prediction = model.predict(X)[0]
                probability = model.predict_proba(X)[0]
            
            prediction_label = label_encoder.inverse_transform([prediction])[0]
            
            return {
                'prediction': prediction_label,
                'confidence': max(probability),
                'probabilities': {
                    'phishing': probability[1] if len(probability) > 1 else probability[0],
                    'legitimate': probability[0] if len(probability) > 1 else 1 - probability[0]
                },
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return None
    
    def _check_whitelist(self, domain: str, target_cse: str = None) -> Dict:
        """Check if domain is whitelisted for any CSE."""
        # Clean domain
        if '://' in domain:
            domain = domain.split('://')[1]
        domain = domain.split('/')[0].lower()
        
        all_cses = self.cse_manager.get_all_cses()
        
        # Check specific CSE first if provided
        if target_cse and target_cse in all_cses:
            cse_data = all_cses[target_cse]
            for whitelisted in cse_data.get('whitelisted_domains', []):
                if domain == whitelisted.lower() or domain.endswith('.' + whitelisted.lower()):
                    return {
                        'classification': {
                            'classification': 'Legitimate',
                            'confidence': 1.0,
                            'risk_score': 0.0,
                            'reasoning': [f'Domain is whitelisted for {target_cse}']
                        },
                        'target_cse': target_cse,
                        'domain': domain,
                        'timestamp': datetime.now().isoformat(),
                        'evidence': {
                            'whitelisted': True,
                            'whitelisted_for': target_cse
                        }
                    }
        
        # Check all CSEs for whitelist
        for cse_name, cse_data in all_cses.items():
            for whitelisted in cse_data.get('whitelisted_domains', []):
                if domain == whitelisted.lower() or domain.endswith('.' + whitelisted.lower()):
                    return {
                        'classification': {
                            'classification': 'Legitimate',
                            'confidence': 1.0,
                            'risk_score': 0.0,
                            'reasoning': [f'Domain is whitelisted for {cse_name}']
                        },
                        'target_cse': cse_name,
                        'domain': domain,
                        'timestamp': datetime.now().isoformat(),
                        'evidence': {
                            'whitelisted': True,
                            'whitelisted_for': cse_name
                        }
                    }
        
        return None  # Not whitelisted
    
    def _rule_based_classification(self, features: Dict, target_cse: str) -> Dict:
        """Rule-based classification with improved reasoning."""
        score = 0
        reasons = []
        
        # Debug: Log some key features
        self.logger.info(f"Features for reasoning: HTTPS={features.get('url_uses_https')}, "
                        f"Domain age={features.get('domain_age_days')}, "
                        f"SSL={features.get('domain_has_ssl')}, "
                        f"New domain={features.get('domain_is_new_domain')}")
        
        # Check HTTPS usage (improved detection)
        https_status = features.get('url_uses_https', 0)
        ssl_cert = features.get('domain_has_ssl', 0)
        if not https_status and not ssl_cert:
            score += 0.2
            reasons.append("Does not use HTTPS")
        elif https_status:
            reasons.append("Uses HTTPS encryption")
        
        # Check domain age with better thresholds
        domain_age = features.get('domain_age_days', 365)
        is_new = features.get('domain_is_new_domain', 0)
        if domain_age < 30 or is_new:
            score += 0.3
            reasons.append("Very new domain")
        elif domain_age < 90:
            score += 0.15
            reasons.append("Recently registered domain")
        elif domain_age > 1000:
            reasons.append("Well-established domain")
        
        # Check suspicious indicators
        if features.get('url_has_suspicious_keywords', 0):
            score += 0.3
            reasons.append("Contains suspicious keywords")
        
        if features.get('url_has_ip_address', 0):
            score += 0.4
            reasons.append("Uses IP address instead of domain")
        
        # Check registrar and WHOIS privacy
        has_registrar = features.get('domain_has_registrar', 1)
        if not has_registrar:
            score += 0.2
            reasons.append("Missing registrar information")
        
        # Check SSL certificate status
        ssl_expires_soon = features.get('domain_ssl_expires_soon', 0)
        if ssl_expires_soon:
            score += 0.1
            reasons.append("SSL certificate expires soon")
        
        # Check content-based features
        status_code = features.get('content_status_code', 200)
        if status_code == 403 or status_code == 404:
            score += 0.2
            reasons.append(f"Suspicious HTTP status: {status_code}")
        
        suspicious_content = features.get('content_has_suspicious_content', 0)
        if suspicious_content:
            score += 0.25
            reasons.append("Contains suspicious content patterns")
        
        password_field = features.get('content_has_password_field', 0)
        hidden_fields = features.get('content_has_hidden_fields', 0)
        if password_field and hidden_fields:
            score += 0.2
            reasons.append("Has login form with hidden fields")
        
        # Check similarity to target CSE (if available)
        if features.get('cse_similarity_score', 0) > 0.7:
            score += 0.4
            reasons.append("High similarity to target CSE")
        
        # Check similarity to target CSE (if available)
        if features.get('cse_similarity_score', 0) > 0.7:
            score += 0.4
            reasons.append("High similarity to target CSE")
        
        # Improve classification thresholds for better sensitivity
        if score >= 0.8:
            classification = "Phishing"
        elif score >= 0.5:
            classification = "Suspected"
        elif score >= 0.3:
            classification = "Potentially Suspicious"
        else:
            classification = "Legitimate"
        
        # Calculate confidence based on number of indicators
        confidence = min(0.95, 0.5 + (len(reasons) * 0.1) + (score * 0.3))
        
        return {
            'classification': classification,
            'confidence': confidence,
            'risk_score': score,
            'reasoning': reasons
        }
    
    def _collect_evidence(self, domain: str, features: Dict, target_cse: str) -> Dict:
        """Collect evidence for the classification."""
        evidence = {
            'domain_metadata': {
                'domain': domain,
                'registration_date': features.get('domain_registration_date'),
                'expiry_date': features.get('domain_expiry_date'),
                'registrar': features.get('domain_registrar'),
                'name_servers': features.get('domain_name_servers', [])
            },
            'security_indicators': {
                'has_ssl': features.get('domain_has_ssl', False),
                'ssl_issuer': features.get('domain_ssl_issuer'),
                'uses_https': features.get('url_uses_https', False)
            },
            'similarity_analysis': {
                'target_cse': target_cse,
                'similarity_score': features.get('cse_similarity_score', 0)
            },
            'suspicious_features': []
        }
        
        # Add suspicious features
        if features.get('url_has_suspicious_keywords'):
            evidence['suspicious_features'].append('Contains suspicious keywords')
        if features.get('url_has_ip_address'):
            evidence['suspicious_features'].append('Uses IP address')
        if features.get('url_url_length', 0) > 100:
            evidence['suspicious_features'].append('Very long URL')
        
        return evidence
    
    def generate_report(self, domains: List[Dict], format: str = "json") -> str:
        """Generate a comprehensive report of detected domains."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_domains_analyzed': len(domains),
                'phishing_domains': len([d for d in domains if d.get('classification', {}).get('classification') == 'Phishing']),
                'suspected_domains': len([d for d in domains if d.get('classification', {}).get('classification') == 'Suspected']),
                'system_version': '1.0.0'
            },
            'cse_summary': {},
            'detected_domains': domains
        }
        
        # Generate CSE summary
        for cse_name in self.cse_manager.cse_data.keys():
            cse_domains = [d for d in domains if d.get('target_cse') == cse_name]
            report['cse_summary'][cse_name] = {
                'total_detected': len(cse_domains),
                'phishing': len([d for d in cse_domains if d.get('classification', {}).get('classification') == 'Phishing']),
                'suspected': len([d for d in cse_domains if d.get('classification', {}).get('classification') == 'Suspected']),
                'sector': self.cse_manager.cse_data[cse_name]['sector']
            }
        
        # Save report
        if format.lower() == 'json':
            filename = f"phishing_report_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == 'csv':
            filename = f"phishing_report_{timestamp}.csv"
            df = pd.DataFrame(domains)
            df.to_csv(filename, index=False)
        
        self.logger.info(f"✅ Report generated: {filename}")
        return filename
    
    async def start_monitoring(self):
        """Start automated monitoring of phishing domains."""
        self.monitoring_active = True
        self.logger.info("🚀 Starting automated phishing domain monitoring...")
        
        while self.monitoring_active:
            try:
                # Discover domains
                discovered = await self.discover_phishing_domains()
                
                # Classify discovered domains
                classified_domains = []
                for domain_info in discovered:
                    domain = domain_info.get('domain')
                    target_cse = domain_info.get('target_cse')
                    
                    if domain:
                        classification = self.classify_domain(domain, target_cse)
                        classified_domains.append({**domain_info, 'classification': classification})
                
                # Generate report if configured
                if self.config['reporting']['auto_generate'] and classified_domains:
                    for fmt in self.config['reporting']['export_formats']:
                        self.generate_report(classified_domains, fmt)
                
                # Wait for next scan
                await asyncio.sleep(self.config['monitoring']['interval_hours'] * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    def stop_monitoring(self):
        """Stop automated monitoring."""
        self.monitoring_active = False
        self.logger.info("🛑 Stopped automated monitoring")
    
    def get_system_status(self) -> Dict:
        """Get current system status and statistics."""
        return {
            'monitoring_active': self.monitoring_active,
            'total_cses': len(self.cse_manager.cse_data),
            'detected_domains': len(self.detected_domains),
            'model_loaded': self.model_data is not None,
            'last_scan': self.detected_domains[-1]['discovery_timestamp'] if self.detected_domains else None,
            'config': self.config
        }

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phishing Detection System for CSEs")
    parser.add_argument('--mode', choices=['monitor', 'scan', 'classify', 'report', 'status'], 
                       default='status', help='Operation mode')
    parser.add_argument('--domain', help='Domain to classify')
    parser.add_argument('--cse', help='Target CSE for classification')
    parser.add_argument('--add-cse', nargs='+', help='Add new CSE: name sector domain1 domain2 ...')
    
    args = parser.parse_args()
    
    # Initialize system
    system = PhishingDetectionSystem()
    
    if args.mode == 'monitor':
        print("🚀 Starting monitoring mode...")
        asyncio.run(system.start_monitoring())
    
    elif args.mode == 'scan':
        print("🔍 Scanning for phishing domains...")
        results = asyncio.run(system.discover_phishing_domains())
        print(f"Found {len(results)} potential domains")
        
        # Generate report
        if results:
            report_file = system.generate_report(results)
            print(f"📄 Report saved to: {report_file}")
    
    elif args.mode == 'classify' and args.domain:
        print(f"🔍 Classifying domain: {args.domain}")
        result = system.classify_domain(args.domain, args.cse)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.mode == 'status':
        print("📊 System Status:")
        status = system.get_system_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.add_cse and len(args.add_cse) >= 3:
        name, sector = args.add_cse[0], args.add_cse[1]
        domains = args.add_cse[2:]
        success = system.add_cse(name, sector, domains)
        print(f"✅ CSE added successfully: {name}" if success else f"❌ Failed to add CSE: {name}")
    
    else:
        print("ℹ️  Available commands:")
        print("  --mode monitor                    # Start automated monitoring")
        print("  --mode scan                       # One-time scan for all CSEs")
        print("  --mode classify --domain <url>    # Classify a specific domain")
        print("  --mode status                     # Show system status")
        print("  --add-cse <name> <sector> <domain1> <domain2> ...  # Add new CSE")
