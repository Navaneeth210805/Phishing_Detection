#!/usr/bin/env python3
"""
Enhanced Web Interface for Phishing Detection System
===================================================

This extends the existing web_app.py with comprehensive CSE management,
domain monitoring, and reporting capabilities.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import pandas as pd
import json
import os
import asyncio
import threading
from datetime import datetime, timedelta
import sys

# Import existing modules
from phishing_detection_system import PhishingDetectionSystem
from cse_manager import CSEManager
from web_app import load_model, predict_url

app = Flask(__name__)
app.secret_key = 'phishing_detection_secret_key_2025'

# Global system instance
detection_system = None
monitoring_thread = None

def initialize_system():
    """Initialize the detection system."""
    global detection_system
    try:
        detection_system = PhishingDetectionSystem()
        print("✅ Phishing Detection System initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return False

@app.route('/')
def index():
    """Main dashboard."""
    if not detection_system:
        return render_template('error.html', 
                             error="System not initialized. Please restart the application.")
    
    # Get system status
    status = detection_system.get_system_status()
    
    # Get recent detections
    recent_detections = detection_system.detected_domains[-10:] if detection_system.detected_domains else []
    
    return render_template('dashboard.html', 
                         status=status, 
                         recent_detections=recent_detections)

@app.route('/cse-management')
def cse_management():
    """CSE management page."""
    if not detection_system:
        return redirect(url_for('index'))
    
    cses = detection_system.get_cse_list()
    return render_template('cse_management.html', cses=cses)

@app.route('/api/cses', methods=['GET'])
def get_cses():
    """API endpoint to get all CSEs."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    cses = detection_system.get_cse_list()
    return jsonify(cses)

@app.route('/api/cses', methods=['POST'])
def add_cse():
    """API endpoint to add a new CSE."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    
    required_fields = ['name', 'sector', 'whitelisted_domains']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    success = detection_system.add_cse(
        name=data['name'],
        sector=data['sector'],
        whitelisted_domains=data['whitelisted_domains'],
        keywords=data.get('keywords', []),
        description=data.get('description', '')
    )
    
    if success:
        return jsonify({'message': 'CSE added successfully'})
    else:
        return jsonify({'error': 'Failed to add CSE'}), 500

@app.route('/api/cses/<cse_name>', methods=['DELETE'])
def remove_cse(cse_name):
    """API endpoint to remove a CSE."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    success = detection_system.remove_cse(cse_name)
    
    if success:
        return jsonify({'message': f'CSE {cse_name} removed successfully'})
    else:
        return jsonify({'error': f'Failed to remove CSE {cse_name}'}), 500

@app.route('/domain-discovery')
def domain_discovery():
    """Domain discovery and monitoring page."""
    if not detection_system:
        return redirect(url_for('index'))
    
    return render_template('domain_discovery.html')

@app.route('/api/discover-domains', methods=['POST'])
def discover_domains():
    """API endpoint to discover domains for specific CSEs."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    target_cses = data.get('target_cses', None)
    
    try:
        # Run discovery in a separate thread to avoid blocking
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        discovered = loop.run_until_complete(
            detection_system.discover_phishing_domains(target_cses)
        )
        
        loop.close()
        
        return jsonify({
            'message': f'Discovered {len(discovered)} potential domains',
            'domains': discovered[:50]  # Limit response size
        })
        
    except Exception as e:
        return jsonify({'error': f'Discovery failed: {str(e)}'}), 500

@app.route('/api/classify-domain', methods=['POST'])
def classify_domain():
    """API endpoint to classify a specific domain."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    domain = data.get('domain')
    target_cse = data.get('target_cse')
    
    if not domain:
        return jsonify({'error': 'Domain is required'}), 400
    
    try:
        result = detection_system.classify_domain(domain, target_cse)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

@app.route('/test-url')
def test_url():
    """URL testing page (enhanced version of original)."""
    return render_template('test_url.html')

@app.route('/api/test-url', methods=['POST'])
def api_test_url():
    """API endpoint for URL testing (reuses existing functionality)."""
    data = request.get_json()
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        # Use existing prediction function from web_app.py
        result, error = predict_url(url)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Enhance with CSE mapping if detection_system is available
        if detection_system:
            cse_mapping = detection_system.cse_manager.map_domain_to_cse(url)
            if cse_mapping:
                result['cse_mapping'] = cse_mapping
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/monitoring')
def monitoring():
    """Monitoring control page."""
    if not detection_system:
        return redirect(url_for('index'))
    
    status = detection_system.get_system_status()
    return render_template('monitoring.html', status=status)

@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    """API endpoint to start monitoring."""
    global monitoring_thread
    
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    if detection_system.monitoring_active:
        return jsonify({'message': 'Monitoring already active'})
    
    def run_monitoring():
        """Run monitoring in separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(detection_system.start_monitoring())
        loop.close()
    
    monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
    monitoring_thread.start()
    
    return jsonify({'message': 'Monitoring started successfully'})

@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """API endpoint to stop monitoring."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    detection_system.stop_monitoring()
    return jsonify({'message': 'Monitoring stopped successfully'})

@app.route('/reports')
def reports():
    """Reports page."""
    if not detection_system:
        return redirect(url_for('index'))
    
    # List available reports
    report_files = [f for f in os.listdir('.') if f.startswith('phishing_report_') and f.endswith(('.json', '.csv'))]
    report_files.sort(reverse=True)  # Most recent first
    
    return render_template('reports.html', report_files=report_files)

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """API endpoint to generate a new report."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    format_type = data.get('format', 'json')
    
    try:
        report_file = detection_system.generate_report(
            detection_system.detected_domains, 
            format_type
        )
        return jsonify({
            'message': 'Report generated successfully',
            'filename': report_file
        })
    except Exception as e:
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/api/download-report/<filename>')
def download_report(filename):
    """API endpoint to download a report file."""
    if not os.path.exists(filename):
        return jsonify({'error': 'Report file not found'}), 404
    
    return send_file(filename, as_attachment=True)

@app.route('/api/system-status')
def system_status():
    """API endpoint to get system status."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    status = detection_system.get_system_status()
    return jsonify(status)

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Phishing Detection Web Interface...")
    
    # Initialize the system
    if initialize_system():
        # Also load the original model for URL testing
        load_model()
        
        print("🌐 Web interface available at: http://localhost:5000")
        print("📊 Dashboard: http://localhost:5000")
        print("🏢 CSE Management: http://localhost:5000/cse-management")
        print("🔍 Domain Discovery: http://localhost:5000/domain-discovery")
        print("🧪 URL Testing: http://localhost:5000/test-url")
        print("📈 Monitoring: http://localhost:5000/monitoring")
        print("📄 Reports: http://localhost:5000/reports")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize system. Exiting...")
        sys.exit(1)
