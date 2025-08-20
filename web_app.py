#!/usr/bin/env python3
"""
Phishing Detection Web Interface
================================

A simple web interface for testing URLs against the trained phishing detection model.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import sys
import traceback
from phishing_feature_extractor import PhishingFeatureExtractor

app = Flask(__name__)

# Global variables
model_data = None
feature_extractor = None

def load_model():
    """Load the trained model."""
    global model_data, feature_extractor
    
    model_path = 'phishing_detection_model.pkl'
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    try:
        model_data = joblib.load(model_path)
        feature_extractor = PhishingFeatureExtractor()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_url(url):
    """Predict if a URL is phishing or legitimate."""
    if model_data is None or feature_extractor is None:
        return None, "Model not loaded"
    
    try:
        # Extract features
        features = feature_extractor.extract_all_features(url)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Remove non-feature columns
        exclude_cols = ['url', 'original_url', 'label', 'cse_name', 'cse_domain', 
                       'subdomain', 'domain', 'suffix', 'fqdn', 'domain_ssl_issuer']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].copy()
        
        # Handle missing values and convert to numeric
        X = X.fillna(-1)
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(-1)
                except:
                    X[col] = 0  # Default for categorical
        
        # Make prediction
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        model_name = model_data['model_name']
        
        if model_name in ['LogisticRegression', 'SVM', 'NaiveBayes']:
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
        else:
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
        
        # Convert prediction back to label
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        
        # Get probability for the predicted class
        prob_phishing = probability[1] if len(probability) > 1 else probability[0]
        prob_legitimate = probability[0] if len(probability) > 1 else 1 - probability[0]
        
        result = {
            'prediction': prediction_label,
            'confidence': max(prob_phishing, prob_legitimate),
            'prob_phishing': prob_phishing,
            'prob_legitimate': prob_legitimate,
            'features': features,
            'model_name': model_name
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a URL for phishing."""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'Please provide a URL'})
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        result, error = predict_url(url)
        
        if error:
            return jsonify({'error': error})
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_data is not None
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Phishing Detection Web Interface...")
        print("Access the interface at: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first:")
        print("python train_model.py")
