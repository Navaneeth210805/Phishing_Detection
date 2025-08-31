#!/usr/bin/env python3
"""
Flask API Backend for Phishing Detection System
==============================================

RESTful API backend providing endpoints for the Next.js frontend.
Handles CSE management, domain classification, monitoring, and reporting.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import asyncio
import threading
from datetime import datetime, timedelta
import sys
import logging

# Add backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from phishing_detection_system import PhishingDetectionSystem
from cse_manager import CSEManager

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Global variables
detection_system = None
monitoring_thread = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize the phishing detection system."""
    global detection_system
    try:
        detection_system = PhishingDetectionSystem()
        logger.info("✅ Phishing Detection System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing system: {e}")
        return False

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# System status endpoint
@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        status = detection_system.get_system_status()
        return jsonify({
            'success': True,
            'data': status
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

# CSE Management Endpoints
@app.route('/api/cses', methods=['GET'])
def get_all_cses():
    """Get all CSEs with their details."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        cses = detection_system.get_cse_list()
        
        # Add statistics for each CSE
        cse_stats = {}
        for cse_name, cse_data in cses.items():
            cse_stats[cse_name] = {
                **cse_data,
                'domain_count': len(cse_data['whitelisted_domains']),
                'keyword_count': len(cse_data.get('keywords', [])),
            }
        
        return jsonify({
            'success': True,
            'data': cse_stats,
            'total': len(cses)
        })
    except Exception as e:
        logger.error(f"Error getting CSEs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cses', methods=['POST'])
def create_cse():
    """Create a new CSE."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'sector', 'whitelisted_domains']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        success = detection_system.add_cse(
            name=data['name'],
            sector=data['sector'],
            domains=data['whitelisted_domains'],
            keywords=data.get('keywords', []),
            description=data.get('description', '')
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'CSE "{data["name"]}" created successfully'
            })
        else:
            return jsonify({'error': 'Failed to create CSE'}), 500
            
    except Exception as e:
        logger.error(f"Error creating CSE: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cses/<cse_name>', methods=['DELETE'])
def delete_cse(cse_name):
    """Delete a CSE."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        success = detection_system.remove_cse(cse_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'CSE "{cse_name}" deleted successfully'
            })
        else:
            return jsonify({'error': f'Failed to delete CSE "{cse_name}"'}), 500
            
    except Exception as e:
        logger.error(f"Error deleting CSE: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cses/<cse_name>', methods=['GET'])
def get_cse_details(cse_name):
    """Get details for a specific CSE."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        cses = detection_system.get_cse_list()
        
        if cse_name not in cses:
            return jsonify({'error': f'CSE "{cse_name}" not found'}), 404
        
        cse_data = cses[cse_name]
        
        return jsonify({
            'success': True,
            'data': {
                'name': cse_name,
                **cse_data,
                'domain_count': len(cse_data['whitelisted_domains']),
                'keyword_count': len(cse_data.get('keywords', [])),
            }
        })
    except Exception as e:
        logger.error(f"Error getting CSE details: {e}")
        return jsonify({'error': str(e)}), 500

# Domain Classification Endpoints
@app.route('/api/domains/classify', methods=['POST'])
def classify_domain():
    """Classify a domain for phishing detection."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        domain = data.get('domain')
        target_cse = data.get('target_cse')
        
        if not domain:
            return jsonify({'error': 'Domain is required'}), 400
        
        result = detection_system.classify_domain(domain, target_cse)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error classifying domain: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/domains/discover', methods=['POST'])
def discover_domains():
    """Discover potential phishing domains for target CSEs."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        target_cses = data.get('target_cses', None)
        
        # Run discovery in a separate thread
        def run_discovery():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    detection_system.discover_phishing_domains(target_cses)
                )
            finally:
                loop.close()
        
        discovered = run_discovery()
        
        return jsonify({
            'success': True,
            'data': {
                'domains': discovered[:50],  # Limit response size
                'total_found': len(discovered),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error discovering domains: {e}")
        return jsonify({'error': str(e)}), 500

# Monitoring Endpoints
@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    """Start the monitoring system."""
    global monitoring_thread
    
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        if detection_system.monitoring_active:
            return jsonify({
                'success': True,
                'message': 'Monitoring is already active'
            })
        
        def run_monitoring():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(detection_system.start_monitoring())
            finally:
                loop.close()
        
        monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
        monitoring_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Monitoring started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """Stop the monitoring system."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        detection_system.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Monitoring stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/status', methods=['GET'])
def get_monitoring_status():
    """Get monitoring status and statistics."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        status = detection_system.get_system_status()
        
        return jsonify({
            'success': True,
            'data': {
                'active': status['monitoring_active'],
                'detected_domains': status['detected_domains'],
                'last_scan': status.get('last_scan'),
                'config': status['config']['monitoring']
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        return jsonify({'error': str(e)}), 500

# Reports Endpoints
@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """Generate a new report."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        format_type = data.get('format', 'json')
        
        report_file = detection_system.generate_report(
            detection_system.detected_domains,
            format_type
        )
        
        return jsonify({
            'success': True,
            'data': {
                'filename': report_file,
                'format': format_type,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/list', methods=['GET'])
def list_reports():
    """List available reports."""
    try:
        report_files = [
            f for f in os.listdir('.')
            if f.startswith('phishing_report_') and f.endswith(('.json', '.csv'))
        ]
        
        reports = []
        for filename in sorted(report_files, reverse=True):
            try:
                # Extract timestamp from filename
                timestamp_str = filename.replace('phishing_report_', '').split('.')[0]
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                file_stats = os.stat(filename)
                reports.append({
                    'filename': filename,
                    'timestamp': timestamp.isoformat(),
                    'size': file_stats.st_size,
                    'format': filename.split('.')[-1]
                })
            except:
                continue
        
        return jsonify({
            'success': True,
            'data': reports
        })
        
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        return jsonify({'error': str(e)}), 500

# Statistics Endpoints
@app.route('/api/stats/dashboard', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics."""
    if not detection_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        status = detection_system.get_system_status()
        cses = detection_system.get_cse_list()
        
        # Calculate sector distribution
        sectors = {}
        total_domains = 0
        
        for cse_name, cse_data in cses.items():
            sector = cse_data['sector']
            domain_count = len(cse_data['whitelisted_domains'])
            
            if sector not in sectors:
                sectors[sector] = {'count': 0, 'domains': 0}
            
            sectors[sector]['count'] += 1
            sectors[sector]['domains'] += domain_count
            total_domains += domain_count
        
        # Recent detections (last 10)
        recent_detections = detection_system.detected_domains[-10:] if detection_system.detected_domains else []
        
        stats = {
            'overview': {
                'total_cses': len(cses),
                'total_domains': total_domains,
                'detected_domains': len(detection_system.detected_domains),
                'monitoring_active': status['monitoring_active'],
                'model_loaded': status['model_loaded']
            },
            'sectors': sectors,
            'recent_detections': recent_detections,
            'system_config': status['config']
        }
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

if __name__ == '__main__':
    print("🚀 Starting Flask API Backend...")
    
    # Initialize the detection system
    if initialize_system():
        print("✅ Phishing Detection System initialized")
        print("🌐 API Documentation:")
        print("  • Health Check: GET /api/health")
        print("  • System Status: GET /api/system/status")
        print("  • CSE Management: GET/POST/DELETE /api/cses")
        print("  • Domain Classification: POST /api/domains/classify")
        print("  • Domain Discovery: POST /api/domains/discover")
        print("  • Monitoring: POST /api/monitoring/start|stop")
        print("  • Reports: GET/POST /api/reports")
        print("  • Dashboard Stats: GET /api/stats/dashboard")
        print()
        print("🔥 Flask API running on: http://localhost:5000")
        print("🔌 CORS enabled for Next.js frontend")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize system. Exiting...")
        sys.exit(1)
