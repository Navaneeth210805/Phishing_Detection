#!/usr/bin/env python3
"""
Phishing Distribution Detector
==============================

This script uses the learned distributions from phishing URLs to detect
if a new URL is likely to be phishing based on its feature distributions.

It loads the pre-trained distributions and calculates likelihood scores
for new URLs based on how well their features match the phishing patterns.
"""

import json
import pickle
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple
import logging
from phishing_feature_extractor import PhishingFeatureExtractor
import os

logger = logging.getLogger(__name__)

class PhishingDistributionDetector:
    """Detect phishing URLs using learned feature distributions."""
    
    def __init__(self, distributions_dir: str = "distributions"):
        """Initialize the detector with learned distributions."""
        self.distributions_dir = distributions_dir
        self.distributions = {}
        self.feature_stats = {}
        self.feature_extractor = PhishingFeatureExtractor()
        
        self.load_distributions()
    
    def load_distributions(self):
        """Load the pre-trained distributions."""
        try:
            # Try to load pickle file first (more efficient)
            pickle_file = os.path.join(self.distributions_dir, "phishing_distributions.pkl")
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    self.distributions = data['distributions']
                    self.feature_stats = data['feature_stats']
                logger.info(f"Loaded distributions from {pickle_file}")
                return
            
            # Fallback to JSON files
            dist_file = os.path.join(self.distributions_dir, "phishing_distributions.json")
            stats_file = os.path.join(self.distributions_dir, "phishing_feature_stats.json")
            
            if os.path.exists(dist_file):
                with open(dist_file, 'r') as f:
                    self.distributions = json.load(f)
                logger.info(f"Loaded distributions from {dist_file}")
            
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.feature_stats = json.load(f)
                logger.info(f"Loaded feature stats from {stats_file}")
            
        except Exception as e:
            logger.error(f"Error loading distributions: {e}")
            raise
    
    def calculate_feature_likelihood(self, feature_name: str, feature_value: float) -> float:
        """Calculate the likelihood of a feature value given the learned distribution."""
        if feature_name not in self.distributions:
            return 0.5  # Neutral likelihood for unknown features
        
        dist_info = self.distributions[feature_name]
        dist_type = dist_info.get('type', 'unknown')
        
        try:
            if dist_type == 'binary':
                # Bernoulli distribution
                p = dist_info['probability_1']
                if feature_value == 1:
                    return p
                else:
                    return 1 - p
            
            elif dist_type == 'discrete':
                # Poisson or empirical distribution
                if dist_info['distribution'] == 'poisson':
                    lambda_param = dist_info['parameters'][0]
                    return stats.poisson.pmf(int(feature_value), lambda_param)
                else:
                    # Empirical distribution
                    values = dist_info['values']
                    probabilities = dist_info['probabilities']
                    if feature_value in values:
                        idx = values.index(feature_value)
                        return probabilities[idx]
                    else:
                        return 1e-6  # Very small probability for unseen values
            
            elif dist_type == 'continuous':
                # Continuous distribution
                dist_name = dist_info['distribution']
                params = dist_info['parameters']
                
                # Get the scipy distribution
                dist = getattr(stats, dist_name)
                
                # Calculate probability density
                pdf_value = dist.pdf(feature_value, *params)
                
                # Convert to likelihood (normalize by max possible density)
                # This is a heuristic to convert PDF to a probability-like measure
                max_density = dist.pdf(dist.mean(*params), *params) if hasattr(dist, 'mean') else pdf_value
                likelihood = min(pdf_value / max(max_density, 1e-10), 1.0) if max_density > 0 else 0.0
                
                return max(likelihood, 1e-6)  # Ensure minimum likelihood
            
            else:
                return 0.5  # Neutral likelihood for unknown types
                
        except Exception as e:
            logger.debug(f"Error calculating likelihood for {feature_name}={feature_value}: {e}")
            return 0.5
    
    def calculate_phishing_score(self, url: str) -> Dict[str, Any]:
        """Calculate the phishing score for a URL based on feature distributions."""
        try:
            # Extract features from the URL
            features = self.feature_extractor.extract_url_features(url)
            
            # Try to get additional features
            try:
                domain_features = self.feature_extractor.extract_domain_features(url)
                features.update(domain_features)
            except:
                pass
            
            try:
                content_features = self.feature_extractor.extract_content_features(url)
                features.update(content_features)
            except:
                pass
            
            # Calculate likelihood for each feature
            feature_likelihoods = {}
            total_log_likelihood = 0
            valid_features = 0
            
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)) and feature_name in self.distributions:
                    likelihood = self.calculate_feature_likelihood(feature_name, feature_value)
                    feature_likelihoods[feature_name] = {
                        'value': feature_value,
                        'likelihood': likelihood,
                        'log_likelihood': np.log(max(likelihood, 1e-10))
                    }
                    total_log_likelihood += np.log(max(likelihood, 1e-10))
                    valid_features += 1
            
            # Calculate overall phishing score
            if valid_features > 0:
                avg_log_likelihood = total_log_likelihood / valid_features
                # Convert to a 0-1 score (higher = more likely phishing)
                # This is a heuristic mapping from log-likelihood to probability
                phishing_score = 1 / (1 + np.exp(-avg_log_likelihood * 0.1))
            else:
                phishing_score = 0.5  # Neutral score if no valid features
            
            # Calculate confidence based on number of features matched
            confidence = min(valid_features / len(self.distributions), 1.0)
            
            # Adjust score based on specific high-risk indicators
            risk_adjustments = self._calculate_risk_adjustments(features)
            adjusted_score = min(phishing_score + risk_adjustments, 1.0)
            
            return {
                'url': url,
                'phishing_score': float(adjusted_score),
                'confidence': float(confidence),
                'raw_score': float(phishing_score),
                'risk_adjustments': risk_adjustments,
                'valid_features': valid_features,
                'total_features': len(self.distributions),
                'avg_log_likelihood': float(avg_log_likelihood) if valid_features > 0 else 0,
                'feature_analysis': feature_likelihoods,
                'extracted_features': {k: v for k, v in features.items() if isinstance(v, (int, float, str))}
            }
            
        except Exception as e:
            logger.error(f"Error calculating phishing score for {url}: {e}")
            return {
                'url': url,
                'phishing_score': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_risk_adjustments(self, features: Dict[str, Any]) -> float:
        """Calculate additional risk adjustments based on specific indicators."""
        adjustments = 0.0
        
        # High-risk indicators that should increase phishing score
        high_risk_indicators = [
            ('has_ip_address', 1, 0.2),
            ('has_suspicious_keywords', 1, 0.15),
            ('has_url_shortening', 1, 0.1),
            ('uses_https', 0, 0.1),  # Not using HTTPS is suspicious
            ('is_new_domain', 1, 0.1),
            ('expires_soon', 1, 0.1),
        ]
        
        for feature_name, risky_value, adjustment in high_risk_indicators:
            if feature_name in features and features[feature_name] == risky_value:
                adjustments += adjustment
        
        # URL length adjustments
        if 'url_length' in features:
            url_length = features['url_length']
            if url_length > 100:  # Very long URLs are suspicious
                adjustments += min((url_length - 100) / 200, 0.2)
        
        # Subdomain count adjustments
        if 'subdomain_count' in features:
            subdomain_count = features['subdomain_count']
            if subdomain_count > 3:  # Many subdomains are suspicious
                adjustments += min((subdomain_count - 3) * 0.05, 0.15)
        
        return adjustments
    
    def classify_url(self, url: str, threshold: float = 0.7) -> Dict[str, Any]:
        """Classify a URL as phishing or legitimate based on learned distributions."""
        result = self.calculate_phishing_score(url)
        
        phishing_score = result['phishing_score']
        confidence = result['confidence']
        
        # Classification based on score and confidence
        if phishing_score >= threshold and confidence >= 0.5:
            classification = "PHISHING"
            risk_level = "HIGH"
        elif phishing_score >= threshold and confidence >= 0.3:
            classification = "LIKELY_PHISHING" 
            risk_level = "MEDIUM-HIGH"
        elif phishing_score >= 0.5:
            classification = "SUSPICIOUS"
            risk_level = "MEDIUM"
        elif phishing_score >= 0.3:
            classification = "LOW_RISK"
            risk_level = "LOW-MEDIUM"
        else:
            classification = "LEGITIMATE"
            risk_level = "LOW"
        
        result.update({
            'classification': classification,
            'risk_level': risk_level,
            'threshold_used': threshold
        })
        
        return result
    
    def batch_classify(self, urls: List[str], threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Classify multiple URLs."""
        results = []
        for url in urls:
            result = self.classify_url(url, threshold)
            results.append(result)
        return results
    
    def get_top_risk_features(self, url: str, top_n: int = 10) -> List[Tuple[str, float, float]]:
        """Get the top risk features for a URL."""
        result = self.calculate_phishing_score(url)
        feature_analysis = result.get('feature_analysis', {})
        
        # Sort features by likelihood (lower likelihood = higher risk)
        sorted_features = sorted(
            feature_analysis.items(),
            key=lambda x: x[1]['likelihood']
        )
        
        top_features = []
        for feature_name, analysis in sorted_features[:top_n]:
            top_features.append((
                feature_name,
                analysis['value'],
                analysis['likelihood']
            ))
        
        return top_features


def main():
    """Example usage of the distribution detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect phishing URLs using learned distributions')
    parser.add_argument('--url', help='Single URL to analyze')
    parser.add_argument('--file', help='File containing URLs to analyze')
    parser.add_argument('--threshold', type=float, default=0.7, help='Classification threshold')
    parser.add_argument('--distributions-dir', default='distributions', 
                       help='Directory containing learned distributions')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PhishingDistributionDetector(args.distributions_dir)
    
    if args.url:
        # Analyze single URL
        result = detector.classify_url(args.url, args.threshold)
        
        print(f"\nURL: {result['url']}")
        print(f"Classification: {result['classification']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Phishing Score: {result['phishing_score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        # Show top risk features
        top_features = detector.get_top_risk_features(args.url)
        print(f"\nTop Risk Features:")
        for feature, value, likelihood in top_features[:5]:
            risk_score = 1 - likelihood
            print(f"  {feature}: {value} (risk: {risk_score:.3f})")
    
    elif args.file:
        # Analyze URLs from file
        with open(args.file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        results = detector.batch_classify(urls, args.threshold)
        
        print(f"Analyzed {len(results)} URLs:")
        print("-" * 60)
        
        for result in results:
            print(f"{result['classification']:<15} {result['phishing_score']:.3f} {result['url']}")
    
    else:
        print("Please provide either --url or --file argument")


if __name__ == "__main__":
    main()
