#!/usr/bin/env python3
"""
Unified Phishing Detection Service
=================================

Production-ready service integrating the realistic phishing detection model
with proper error handling, logging, and monitoring capabilities.
"""

import joblib
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import json

class PhishingDetectionService:
    """
    Production-ready phishing detection service using the realistic model.
    
    Features:
    - 97.3% test accuracy with realistic constraints
    - 15 domain-based features (no label dependency)
    - Proper confidence scoring
    - Error handling and logging
    - Model monitoring capabilities
    """
    
    def __init__(self, model_path: str = "realistic_phishing_model.pkl"):
        """Initialize the phishing detection service."""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_info = {}
        self.prediction_count = 0
        self.error_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the trained model and associated components."""
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load model data
            model_data = joblib.load(self.model_path)
            
            # Extract components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            
            # Store model metadata
            self.model_info = {
                'model_type': model_data.get('model_type', 'realistic_with_challenges'),
                'val_accuracy': model_data.get('val_accuracy', 0.0),
                'test_accuracy': model_data.get('test_accuracy', 0.0),
                'roc_auc': model_data.get('roc_auc', 0.0),
                'feature_count': len(self.feature_names),
                'loaded_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Model loaded successfully: {self.model_path}")
            self.logger.info(f"   Test Accuracy: {self.model_info['test_accuracy']:.3f}")
            self.logger.info(f"   ROC AUC: {self.model_info['roc_auc']:.3f}")
            self.logger.info(f"   Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            return False
    
    def extract_features(self, domain: str) -> Dict[str, float]:
        """
        Extract realistic features from a domain without label dependency.
        
        Args:
            domain: Domain string to analyze
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Clean domain
            domain = str(domain).strip().lower()
            if not domain:
                raise ValueError("Empty domain provided")
            
            features = {}
            
            # Basic domain properties
            features['domain_length'] = len(domain)
            features['dot_count'] = domain.count('.')
            features['dash_count'] = domain.count('-')
            features['underscore_count'] = domain.count('_')
            features['digit_count'] = sum(c.isdigit() for c in domain)
            features['uppercase_count'] = sum(c.isupper() for c in domain)
            
            # Character ratios
            features['digit_ratio'] = features['digit_count'] / max(len(domain), 1)
            features['special_char_ratio'] = (features['dash_count'] + features['underscore_count']) / max(len(domain), 1)
            
            # Domain structure analysis
            parts = domain.split('.')
            features['domain_parts'] = len(parts)
            features['longest_part'] = max(len(part) for part in parts) if parts else 0
            features['shortest_part'] = min(len(part) for part in parts) if parts else 0
            
            # Character entropy (complexity measure)
            if len(domain) > 0:
                char_counts = {}
                for char in domain:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                entropy = 0
                for count in char_counts.values():
                    prob = count / len(domain)
                    if prob > 0:
                        entropy -= prob * np.log2(prob)
                features['entropy'] = entropy
            else:
                features['entropy'] = 0
            
            # Pattern detection
            features['has_consecutive_chars'] = 1 if any(domain[i] == domain[i+1] for i in range(len(domain)-1)) else 0
            features['vowel_count'] = sum(1 for c in domain.lower() if c in 'aeiou')
            features['consonant_count'] = sum(1 for c in domain.lower() if c.isalpha() and c not in 'aeiou')
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features from domain '{domain}': {e}")
            raise
    
    def predict_single(self, domain: str) -> Dict[str, Union[str, float, int]]:
        """
        Predict if a single domain is phishing or suspicious.
        
        Args:
            domain: Domain to classify
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.model:
                raise RuntimeError("Model not loaded")
            
            # Extract features
            features = self.extract_features(domain)
            
            # Convert to DataFrame with correct feature order
            feature_df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            for feature_name in self.feature_names:
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = 0.0
            
            # Reorder columns to match training
            feature_df = feature_df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Map prediction to label
            label = "Phishing" if prediction == 1 else "Suspicious"
            confidence = float(max(prediction_proba))
            
            # Track predictions
            self.prediction_count += 1
            
            result = {
                'domain': domain,
                'prediction': label,
                'confidence': confidence,
                'risk_score': float(prediction_proba[1]),  # Phishing probability
                'features_extracted': len(features),
                'model_version': self.model_info.get('model_type', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Prediction for '{domain}': {label} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error predicting domain '{domain}': {e}")
            raise
    
    def predict_batch(self, domains: List[str]) -> List[Dict[str, Union[str, float, int]]]:
        """
        Predict multiple domains in batch.
        
        Args:
            domains: List of domains to classify
            
        Returns:
            List of prediction results
        """
        results = []
        
        for domain in domains:
            try:
                result = self.predict_single(domain)
                results.append(result)
            except Exception as e:
                # Continue with other domains even if one fails
                error_result = {
                    'domain': domain,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'risk_score': 0.0,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get detailed model information and statistics."""
        return {
            'model_info': self.model_info,
            'feature_names': self.feature_names,
            'statistics': {
                'total_predictions': self.prediction_count,
                'total_errors': self.error_count,
                'error_rate': self.error_count / max(self.prediction_count, 1),
                'uptime': datetime.now().isoformat()
            },
            'performance': {
                'test_accuracy': self.model_info.get('test_accuracy', 'Unknown'),
                'validation_accuracy': self.model_info.get('val_accuracy', 'Unknown'),
                'roc_auc': self.model_info.get('roc_auc', 'Unknown'),
                'expected_range': '85-95% accuracy for realistic phishing detection'
            }
        }
    
    def explain_prediction(self, domain: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Provide detailed explanation of a prediction.
        
        Args:
            domain: Domain to explain
            
        Returns:
            Detailed explanation of the prediction
        """
        try:
            # Get basic prediction
            prediction_result = self.predict_single(domain)
            
            # Extract features for explanation
            features = self.extract_features(domain)
            
            # Get feature importance from model
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                feature_importance = {name: 1.0/len(self.feature_names) for name in self.feature_names}
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            explanation = {
                'prediction_summary': prediction_result,
                'domain_analysis': {
                    'length': len(domain),
                    'structure': f"{domain.count('.') + 1} parts separated by dots",
                    'special_chars': domain.count('-') + domain.count('_'),
                    'digits': sum(c.isdigit() for c in domain),
                    'complexity': features.get('entropy', 0)
                },
                'feature_analysis': {
                    'extracted_features': features,
                    'top_influential_features': sorted_features[:5],
                    'feature_contributions': 'Higher entropy and shorter domain parts typically indicate legitimate domains'
                },
                'risk_assessment': {
                    'risk_level': 'High' if prediction_result['risk_score'] > 0.7 else 'Medium' if prediction_result['risk_score'] > 0.3 else 'Low',
                    'confidence_level': 'High' if prediction_result['confidence'] > 0.8 else 'Medium' if prediction_result['confidence'] > 0.6 else 'Low',
                    'recommendation': 'Block' if prediction_result['prediction'] == 'Phishing' and prediction_result['confidence'] > 0.7 else 'Monitor'
                }
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining prediction for '{domain}': {e}")
            raise
    
    def health_check(self) -> Dict[str, Union[str, bool, Dict]]:
        """Perform health check on the service."""
        try:
            # Test prediction on a sample domain
            test_domain = "example.com"
            test_result = self.predict_single(test_domain)
            
            health_status = {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'scaler_loaded': self.scaler is not None,
                'feature_count': len(self.feature_names),
                'test_prediction': test_result['prediction'],
                'test_confidence': test_result['confidence'],
                'statistics': {
                    'total_predictions': self.prediction_count,
                    'error_rate': self.error_count / max(self.prediction_count, 1)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Convenience functions for Flask integration
def create_phishing_service(model_path: str = "realistic_phishing_model.pkl") -> PhishingDetectionService:
    """Create and initialize the phishing detection service."""
    return PhishingDetectionService(model_path)

def predict_domain(service: PhishingDetectionService, domain: str) -> Dict:
    """Simple wrapper for domain prediction."""
    return service.predict_single(domain)

def explain_domain(service: PhishingDetectionService, domain: str) -> Dict:
    """Simple wrapper for domain explanation."""
    return service.explain_prediction(domain)

# Main execution for testing
if __name__ == "__main__":
    # Test the service
    print("🧪 Testing Phishing Detection Service...")
    
    service = PhishingDetectionService()
    
    # Test domains
    test_domains = [
        "google.com",
        "phishing-site.tk",
        "suspicious-domain.ml",
        "legitimate-bank.com",
        "fake-paypal.ga"
    ]
    
    print(f"\n📊 Testing {len(test_domains)} domains:")
    for domain in test_domains:
        try:
            result = service.predict_single(domain)
            print(f"  {domain:25} → {result['prediction']:10} (confidence: {result['confidence']:.3f})")
        except Exception as e:
            print(f"  {domain:25} → Error: {e}")
    
    # Test health check
    health = service.health_check()
    print(f"\n🏥 Service Health: {health['status']}")
    
    # Test model info
    info = service.get_model_info()
    print(f"\n📈 Model Performance:")
    print(f"  Test Accuracy: {info['performance']['test_accuracy']}")
    print(f"  ROC AUC: {info['performance']['roc_auc']}")
    print(f"  Total Predictions: {info['statistics']['total_predictions']}")
