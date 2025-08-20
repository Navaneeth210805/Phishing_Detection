#!/usr/bin/env python3
"""
Quick Demo Script for Phishing Detection
=========================================

This script demonstrates the phishing detection system by analyzing a single URL.
Use this to test the feature extraction before training the full model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phishing_feature_extractor import PhishingFeatureExtractor
import pandas as pd
import json

def demo_url_analysis(url):
    """Demonstrate URL analysis with the feature extractor."""
    print(f"🔍 Analyzing URL: {url}")
    print("=" * 60)
    
    # Initialize feature extractor
    extractor = PhishingFeatureExtractor()
    
    try:
        # Extract features
        features = extractor.extract_all_features(url)
        
        # Display results
        print(f"\n✅ Feature extraction completed!")
        print(f"📊 Total features extracted: {len(features)}")
        
        # Group features by category
        url_features = {k: v for k, v in features.items() if k.startswith('url_')}
        domain_features = {k: v for k, v in features.items() if k.startswith('domain_')}
        content_features = {k: v for k, v in features.items() if k.startswith('content_')}
        other_features = {k: v for k, v in features.items() 
                         if not any(k.startswith(prefix) for prefix in ['url_', 'domain_', 'content_']) 
                         and k not in ['url', 'subdomain', 'domain', 'suffix', 'fqdn']}
        
        # Display feature categories
        print(f"\n🌐 URL Features ({len(url_features)}):")
        for k, v in list(url_features.items())[:5]:
            print(f"  • {k.replace('url_', '')}: {v}")
        if len(url_features) > 5:
            print(f"  ... and {len(url_features) - 5} more")
        
        print(f"\n🔍 Domain Features ({len(domain_features)}):")
        for k, v in list(domain_features.items())[:5]:
            print(f"  • {k.replace('domain_', '')}: {v}")
        if len(domain_features) > 5:
            print(f"  ... and {len(domain_features) - 5} more")
        
        print(f"\n📄 Content Features ({len(content_features)}):")
        for k, v in list(content_features.items())[:5]:
            print(f"  • {k.replace('content_', '')}: {v}")
        if len(content_features) > 5:
            print(f"  ... and {len(content_features) - 5} more")
        
        print(f"\n🔧 Other Features ({len(other_features)}):")
        for k, v in list(other_features.items())[:3]:
            print(f"  • {k}: {v}")
        if len(other_features) > 3:
            print(f"  ... and {len(other_features) - 3} more")
        
        # Key security indicators
        print(f"\n🛡️ Security Indicators:")
        print(f"  • HTTPS Usage: {'Yes' if features.get('url_uses_https', 0) else 'No'}")
        print(f"  • Has SSL Certificate: {'Yes' if features.get('domain_has_ssl', 0) else 'No'}")
        print(f"  • Domain Age: {features.get('domain_domain_age_days', 'Unknown')} days")
        print(f"  • Suspicious Keywords: {'Yes' if features.get('url_has_suspicious_keywords', 0) else 'No'}")
        print(f"  • IP Address in URL: {'Yes' if features.get('url_has_ip_address', 0) else 'No'}")
        print(f"  • URL Length: {features.get('url_url_length', 0)} characters")
        
        # Save features to JSON for inspection
        output_file = f"demo_features_{url.replace('http://', '').replace('https://', '').replace('/', '_').replace('.', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2, default=str)
        print(f"\n💾 Features saved to: {output_file}")
        
        return features
        
    except Exception as e:
        print(f"❌ Error analyzing URL: {e}")
        return None

def main():
    """Main function for demo."""
    print("🛡️ Phishing Detection System - Demo")
    print("====================================")
    
    # Test URLs
    test_urls = [
        "google.com",
        "github.com",
        "example.com"
    ]
    
    if len(sys.argv) > 1:
        # Use URL from command line argument
        url = sys.argv[1]
        demo_url_analysis(url)
    else:
        # Interactive mode
        print("\nEnter a URL to analyze (or press Enter for demo with google.com):")
        user_url = input("URL: ").strip()
        
        if not user_url:
            user_url = "google.com"
        
        demo_url_analysis(user_url)
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Run feature extraction on training data: python phishing_feature_extractor.py --batch")
    print(f"2. Train the ML models: python train_model.py")
    print(f"3. Start the web interface: python web_app.py")

if __name__ == "__main__":
    main()
