#!/usr/bin/env python3
"""
Quick Test Script for Phishing Distribution Learning
===================================================

This script runs a quick test of the distribution learning system
with a limited number of URLs to verify everything works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phishing_distribution_learner import PhishingDistributionLearner
from phishing_distribution_detector import PhishingDistributionDetector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """Run a quick test with limited URLs."""
    print("="*60)
    print("QUICK TEST: Phishing Distribution Learning")
    print("="*60)
    
    try:
        # Step 1: Learn distributions from a subset of phishing URLs
        print("\n1. Learning distributions from phishing URLs...")
        learner = PhishingDistributionLearner("../phishing_links.txt")
        
        # Use only first 50 URLs for quick testing
        learner.run_analysis(max_urls=50)
        
        print("\n✅ Distribution learning completed!")
        
        # Step 2: Test the detector with some sample URLs
        print("\n2. Testing the detector...")
        detector = PhishingDistributionDetector("distributions")
        
        # Test URLs (mix of suspicious and normal patterns)
        test_urls = [
            "http://support-coinbasext-faq.typedream.app/",  # From phishing list
            "https://www.google.com",                        # Legitimate
            "http://amazonzzz.com/",                         # From phishing list
            "https://github.com",                            # Legitimate
            "http://warning-remove-now.vercel.app/",         # From phishing list
        ]
        
        print("\nTesting URLs:")
        print("-" * 80)
        print(f"{'Classification':<15} {'Score':<6} {'Confidence':<10} {'URL'}")
        print("-" * 80)
        
        for url in test_urls:
            try:
                result = detector.classify_url(url, threshold=0.6)
                print(f"{result['classification']:<15} {result['phishing_score']:<6.3f} {result['confidence']:<10.3f} {url}")
            except Exception as e:
                print(f"{'ERROR':<15} {'N/A':<6} {'N/A':<10} {url} - {str(e)}")
        
        print("-" * 80)
        print("\n✅ Testing completed!")
        
        # Step 3: Show some statistics
        print("\n3. Distribution Statistics:")
        if os.path.exists("distributions/phishing_distributions.json"):
            import json
            with open("distributions/phishing_distributions.json", 'r') as f:
                distributions = json.load(f)
            
            dist_types = {}
            for feature, dist_info in distributions.items():
                dist_type = dist_info.get('type', 'unknown')
                dist_types[dist_type] = dist_types.get(dist_type, 0) + 1
            
            print(f"Total features analyzed: {len(distributions)}")
            for dist_type, count in dist_types.items():
                print(f"  {dist_type}: {count} features")
        
        print(f"\nResults saved to: ./distributions/")
        print("\n✅ Quick test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during quick test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
