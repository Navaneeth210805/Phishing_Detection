#!/usr/bin/env python3
"""
Updated Phishing Detection System with Binary Classification
===========================================================

This updated system implements:
1. Binary classification (Suspicious vs Phishing)
2. CSE whitelist-first approach for legitimate domains
3. Advanced ensemble model with intelligent class balancing
4. Improved confidence scoring and reasoning

Architecture:
- CSE Whitelist: Immediate classification of legitimate domains
- Binary ML Model: Suspicious vs Phishing classification for non-whitelisted domains
- Rule-based Fallback: Additional heuristics for edge cases
- Confidence Scoring: Transparent reasoning for classifications
"""

import joblib
import json
import logging
from typing import Dict, Any, Tuple
from .phishing_feature_extractor import PhishingFeatureExtractor
from .cse_manager import CSEManager

class UpdatedPhishingDetectionSystem:
    """
    Updated phishing detection system with binary classification approach.
    """
    
    def __init__(self):
        self.feature_extractor = PhishingFeatureExtractor()
        self.cse_manager = CSEManager()
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.label_mapping = {}
        self.load_binary_model()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phishing_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_binary_model(self):
        """Load the improved binary classification model."""
        try:
            model_data = joblib.load('improved_binary_model.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.label_mapping = model_data['label_mapping']
            self.logger.info("✅ Loaded improved binary classification model")
        except Exception as e:
            self.logger.error(f"❌ Failed to load binary model: {e}")
            # Fallback to rule-based classification
            self.model = None
    
    def classify_domain(self, domain: str) -> Dict[str, Any]:
        """
        Classify a domain using the updated 3-tier approach:
        1. CSE Whitelist check
        2. Binary ML classification (Suspicious vs Phishing)
        3. Rule-based fallback
        """
        
        self.logger.info(f"🔍 Analyzing domain: {domain}")
        
        # Tier 1: CSE Whitelist Check
        whitelist_result = self._check_cse_whitelist(domain)
        if whitelist_result['is_whitelisted']:
            return whitelist_result
        
        # Tier 2: Binary ML Classification
        if self.model is not None:
            ml_result = self._binary_ml_classification(domain)
            if ml_result['confidence'] > 0.6:  # High confidence threshold
                return ml_result
        
        # Tier 3: Rule-based Fallback
        return self._rule_based_classification(domain)
    
    def _check_cse_whitelist(self, domain: str) -> Dict[str, Any]:
        """Check if domain is in CSE whitelist."""
        
        is_whitelisted, cse_name = self.cse_manager.is_whitelisted(domain)
        
        if is_whitelisted:
            self.logger.info(f"✅ Domain whitelisted by {cse_name}")
            return {
                'domain': domain,
                'classification': 'Legitimate',
                'confidence': 0.95,
                'is_whitelisted': True,
                'cse_name': cse_name,
                'reasoning': f"Domain is officially verified and whitelisted by {cse_name}",
                'method': 'CSE_Whitelist',
                'details': {
                    'whitelist_source': cse_name,
                    'verification_status': 'Official CSE Domain',
                    'risk_level': 'Very Low'
                }
            }
        
        return {'is_whitelisted': False}
    
    def _binary_ml_classification(self, domain: str) -> Dict[str, Any]:
        """Perform binary ML classification (Suspicious vs Phishing)."""
        
        try:
            self.logger.info("🤖 Performing binary ML classification")
            
            # Extract features
            features = self.feature_extractor.extract_features(domain)
            
            # Convert to model format
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0))
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Predict
            prediction = self.model.predict(feature_vector_scaled)[0]
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            
            # Map prediction to label
            predicted_label = [k for k, v in self.label_mapping.items() if v == prediction][0]
            confidence = max(probabilities)
            
            # Enhanced reasoning based on feature analysis
            reasoning = self._generate_ml_reasoning(features, predicted_label, confidence)
            
            self.logger.info(f"🎯 ML Classification: {predicted_label} (confidence: {confidence:.2f})")
            
            return {
                'domain': domain,
                'classification': predicted_label,
                'confidence': confidence,
                'is_whitelisted': False,
                'reasoning': reasoning,
                'method': 'Binary_ML_Ensemble',
                'details': {
                    'model_type': 'Advanced Weighted Ensemble',
                    'feature_count': len(self.feature_names),
                    'suspicious_probability': probabilities[0],
                    'phishing_probability': probabilities[1],
                    'key_features': self._get_key_features(features)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ ML classification failed: {e}")
            return {'confidence': 0}  # Trigger fallback
    
    def _rule_based_classification(self, domain: str) -> Dict[str, Any]:
        """Rule-based classification fallback."""
        
        self.logger.info("📋 Using rule-based classification")
        
        try:
            features = self.feature_extractor.extract_features(domain)
        except:
            features = {}
        
        suspicious_indicators = []
        phishing_indicators = []
        confidence_factors = []
        
        # Analyze various indicators
        if features.get('url_has_ip_address', 0):
            phishing_indicators.append("Uses IP address instead of domain name")
            confidence_factors.append(0.3)
        
        if features.get('url_length', 0) > 50:
            suspicious_indicators.append("Unusually long URL")
            confidence_factors.append(0.1)
        
        if not features.get('url_uses_https', 1):
            suspicious_indicators.append("Does not use HTTPS")
            confidence_factors.append(0.15)
        
        if features.get('domain_age_days', 365) < 30:
            phishing_indicators.append("Very new domain (less than 30 days)")
            confidence_factors.append(0.25)
        
        if features.get('url_has_suspicious_keywords', 0):
            phishing_indicators.append("Contains suspicious keywords")
            confidence_factors.append(0.2)
        
        # Determine classification
        phishing_score = sum(confidence_factors) if phishing_indicators else 0
        
        if phishing_score > 0.4:
            classification = 'Phishing'
            confidence = min(0.8, 0.5 + phishing_score)
            all_indicators = phishing_indicators
        elif suspicious_indicators or phishing_score > 0.2:
            classification = 'Suspicious'
            confidence = min(0.7, 0.4 + phishing_score)
            all_indicators = suspicious_indicators + phishing_indicators
        else:
            classification = 'Suspicious'  # Conservative default
            confidence = 0.6
            all_indicators = ["Not in verified whitelist, requires manual verification"]
        
        reasoning = f"Rule-based analysis identified {len(all_indicators)} risk indicators: " + \
                   "; ".join(all_indicators)
        
        self.logger.info(f"📋 Rule-based result: {classification} (confidence: {confidence:.2f})")
        
        return {
            'domain': domain,
            'classification': classification,
            'confidence': confidence,
            'is_whitelisted': False,
            'reasoning': reasoning,
            'method': 'Rule_Based_Fallback',
            'details': {
                'phishing_indicators': phishing_indicators,
                'suspicious_indicators': suspicious_indicators,
                'phishing_score': phishing_score,
                'total_indicators': len(all_indicators)
            }
        }
    
    def _generate_ml_reasoning(self, features: Dict, classification: str, confidence: float) -> str:
        """Generate detailed reasoning for ML classification."""
        
        key_features = []
        
        # Analyze key feature contributions
        if features.get('url_has_ip_address', 0):
            key_features.append("IP address usage detected")
        
        if features.get('domain_age_days', 365) < 90:
            key_features.append(f"Domain age: {features.get('domain_age_days', 0)} days (relatively new)")
        
        if not features.get('url_uses_https', 1):
            key_features.append("No HTTPS encryption")
        
        if features.get('url_length', 0) > 40:
            key_features.append(f"Long URL ({features.get('url_length', 0)} characters)")
        
        if features.get('url_has_suspicious_keywords', 0):
            key_features.append("Suspicious keywords present")
        
        if features.get('url_has_suspicious_tld', 0):
            key_features.append("Suspicious top-level domain")
        
        base_reasoning = f"Advanced ensemble model classified as {classification} with {confidence:.1%} confidence. "
        
        if key_features:
            feature_reasoning = "Key factors: " + "; ".join(key_features[:3])
        else:
            feature_reasoning = "Analysis based on comprehensive feature evaluation across 51 domain characteristics"
        
        return base_reasoning + feature_reasoning
    
    def _get_key_features(self, features: Dict) -> Dict[str, Any]:
        """Extract key features for detailed response."""
        
        return {
            'domain_age_days': features.get('domain_age_days', 0),
            'uses_https': bool(features.get('url_uses_https', 0)),
            'has_ip_address': bool(features.get('url_has_ip_address', 0)),
            'url_length': features.get('url_length', 0),
            'has_suspicious_keywords': bool(features.get('url_has_suspicious_keywords', 0)),
            'has_ssl': bool(features.get('domain_has_ssl', 0))
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the detection system."""
        
        return {
            'system_version': '2.0.0',
            'classification_type': 'Binary (Suspicious vs Phishing) + CSE Whitelist',
            'model_loaded': self.model is not None,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'cse_count': len(self.cse_manager.get_all_cses()) if hasattr(self.cse_manager, 'get_all_cses') else 0,
            'detection_methods': ['CSE_Whitelist', 'Binary_ML_Ensemble', 'Rule_Based_Fallback'],
            'improvements': [
                'Binary classification reduces false positives',
                'CSE whitelist handles legitimate domains',
                'Advanced ensemble with 4 models',
                'Intelligent class balancing',
                'Enhanced feature engineering',
                'Transparent confidence scoring'
            ]
        }

# Legacy compatibility
class PhishingDetectionSystem(UpdatedPhishingDetectionSystem):
    """Legacy compatibility wrapper."""
    
    def __init__(self):
        super().__init__()
        self.logger.info("🔄 Using updated binary classification system")
    
    def detect_phishing(self, domain: str) -> Dict[str, Any]:
        """Legacy method name for backward compatibility."""
        return self.classify_domain(domain)
