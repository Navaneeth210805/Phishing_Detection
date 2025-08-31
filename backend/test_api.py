#!/usr/bin/env python3
"""
API Test Script
===============

Quick test script to verify all API endpoints are working correctly.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000/api"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test a single API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n🔍 Testing {method} {endpoint}")
    if description:
        print(f"   {description}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {response.status_code}")
            if 'success' in result:
                print(f"   Success: {result['success']}")
            if 'message' in result:
                print(f"   Message: {result['message']}")
            return result
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    """Run all API tests."""
    print("🚀 Starting API Tests...")
    
    # Test 1: Health check
    test_endpoint("GET", "/health", description="Basic health check")
    
    # Test 2: System status
    test_endpoint("GET", "/system/status", description="Get system status")
    
    # Test 3: Get all CSEs
    result = test_endpoint("GET", "/cses", description="Get all CSEs")
    
    # Test 4: Domain classification
    test_data = {"domain": "sbi.co.in"}
    test_endpoint("POST", "/domains/classify", test_data, "Classify legitimate domain")
    
    test_data = {"domain": "sbi-bank-login.com"}
    test_endpoint("POST", "/domains/classify", test_data, "Classify suspicious domain")
    
    # Test 5: Monitoring status
    test_endpoint("GET", "/monitoring/status", description="Get monitoring status")
    
    # Test 6: Dashboard stats
    test_endpoint("GET", "/stats/dashboard", description="Get dashboard statistics")
    
    print("\n🎯 API tests completed!")

if __name__ == "__main__":
    main()
