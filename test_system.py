#!/usr/bin/env python3
"""
Quick test of the main phishing detection system
"""

from phishing_detection_system import PhishingDetectionSystem

def test_system():
    print('🚀 Initializing Phishing Detection System...')
    system = PhishingDetectionSystem()
    print('✅ System initialized successfully!')
    
    print('\n📊 System Status:')
    status = system.get_system_status()
    print(f'  • Total CSEs: {status["total_cses"]}')
    print(f'  • Monitoring: {"Active" if status["monitoring_active"] else "Inactive"}')
    print(f'  • Model Loaded: {"Yes" if status["model_loaded"] else "No"}')
    
    print('\n🏢 Available CSEs:')
    cses = system.get_cse_list()
    for i, (cse_name, cse_data) in enumerate(cses.items()):
        if i < 5:  # Show first 5
            print(f'  • {cse_name} ({cse_data["sector"]})')
            print(f'    Domains: {len(cse_data["whitelisted_domains"])}')
    
    # Test domain classification
    print('\n🔍 Testing domain classification...')
    test_domain = 'suspicious-sbi-bank.com'
    result = system.classify_domain(test_domain, 'State Bank of India (SBI)')
    print(f'Domain: {test_domain}')
    print(f'Classification: {result.get("classification", {}).get("classification", "Unknown")}')
    print(f'Confidence: {result.get("confidence", 0):.2f}')
    
    return system

if __name__ == "__main__":
    system = test_system()
    print('\n✅ System test completed successfully!')
