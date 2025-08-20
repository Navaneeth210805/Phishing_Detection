#!/usr/bin/env python3
"""
Start the Phishing Detection Web Interface
==========================================

This script starts the web interface for the phishing detection system.
"""

import os
import sys
import subprocess

def check_model_exists():
    """Check if the trained model exists."""
    model_path = "phishing_detection_model.pkl"
    return os.path.exists(model_path)

def start_web_interface():
    """Start the Flask web interface."""
    if not check_model_exists():
        print("❌ Trained model not found!")
        print("📝 Please train the model first:")
        print("   python train_model.py")
        print("\n💡 Or run the complete setup:")
        print("   python setup.py")
        return False
    
    print("🚀 Starting Phishing Detection Web Interface...")
    print("🌐 The interface will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "web_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped.")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Change to project directory
    project_dir = "/home/vk/phishing/phishing_detection_project"
    os.chdir(project_dir)
    
    start_web_interface()
