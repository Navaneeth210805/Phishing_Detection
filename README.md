# Phishing Detection System

A comprehensive AI-based phishing detection system designed to identify and classify phishing domains targeting Critical Sector Entities (CSEs). This system extracts multiple features from URLs and web content to provide accurate phishing detection.

## Features

- **Advanced Feature Extraction**: 50+ features including URL structure, domain information, content analysis, and security characteristics
- **Multiple ML Models**: Supports Random Forest, Gradient Boosting, SVM, Logistic Regression, and Naive Bayes
- **Web Interface**: User-friendly web interface for real-time URL analysis
- **Batch Processing**: Process multiple URLs from files or datasets
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Detailed model performance metrics and feature importance analysis

## Project Structure

```
phishing_detection_project/
├── venv/                           # Virtual environment
├── templates/                      # Web interface templates
│   └── index.html
├── phishing_feature_extractor.py   # Main feature extraction script
├── train_model.py                  # Model training script
├── web_app.py                      # Web interface application
├── explore_dataset.py              # Dataset exploration utility
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── phishing_features_training.csv  # Extracted features (after running)
├── phishing_detection_model.pkl    # Trained model (after training)
├── feature_importance.png          # Feature importance plot
└── model_comparison.png            # Model comparison plot
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/vk/phishing/phishing_detection_project
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install additional dependencies if needed:**
   ```bash
   pip install flask matplotlib seaborn plotly
   ```

## Usage

### 1. Extract Features from Training Dataset

First, extract features from the training dataset:

```bash
python phishing_feature_extractor.py --batch
```

This will:
- Process the training dataset (`PS02_Training_set.xlsx`)
- Extract 50+ features for each URL
- Save results to `phishing_features_training.csv`

### 2. Train Machine Learning Models

Train and evaluate multiple ML models:

```bash
python train_model.py
```

This will:
- Load the extracted features
- Train multiple models (Random Forest, Gradient Boosting, SVM, etc.)
- Perform hyperparameter tuning
- Generate performance visualizations
- Save the best model to `phishing_detection_model.pkl`

### 3. Analyze Single URLs

Analyze a single URL for phishing characteristics:

```bash
python phishing_feature_extractor.py --url "example.com"
```

### 4. Process URLs from File

Process multiple URLs from a text file:

```bash
python phishing_feature_extractor.py --file urls.txt --output results.csv
```

### 5. Web Interface

Launch the web interface for interactive URL analysis:

```bash
python web_app.py
```

Then open your browser and go to: `http://localhost:5000`

## Extracted Features

The system extracts the following categories of features:

### URL Features (20+ features)
- URL length, domain length, path length
- Number of subdomains, dots, dashes
- Suspicious keywords detection
- IP address usage
- URL shortening service detection
- HTTPS usage
- Character analysis (digit ratio, entropy, special characters)

### Domain Features (15+ features)
- Domain age and expiration
- WHOIS information
- DNS records
- SSL certificate information
- Domain registration details

### Content Features (15+ features)
- HTTP status codes
- HTML structure analysis
- Form detection (password fields, hidden fields)
- Link analysis (internal vs external)
- Image and script analysis
- Suspicious content patterns
- Redirect behavior

## Model Performance

The system evaluates models using:
- **Accuracy**: Overall classification accuracy
- **AUC Score**: Area Under the ROC Curve
- **Cross-validation**: 5-fold cross-validation
- **Feature Importance**: Analysis of most predictive features

Expected performance on the training dataset:
- Random Forest: ~95% accuracy
- Gradient Boosting: ~94% accuracy
- SVM: ~92% accuracy

## Dataset Information

The training dataset contains:
- **1,043 URLs** from the PS02_Training_set.xlsx
- **Labels**: "Phishing" and "Suspected"
- **CSE Organizations**: 19 different critical sector entities
- **Evidence Files**: Screenshots and documentation for phishing sites

## API Usage

### Web API Endpoints

- `GET /`: Main web interface
- `POST /analyze`: Analyze a URL (JSON input: `{"url": "example.com"}`)
- `GET /health`: Health check and model status

### Example API Call

```python
import requests

response = requests.post('http://localhost:5000/analyze', 
                        json={'url': 'suspicious-site.com'})
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Key Files Description

### `phishing_feature_extractor.py`
- Main feature extraction engine
- Extracts URL, domain, and content features
- Supports single URL, batch processing, and file input
- Handles errors gracefully with default values

### `train_model.py`
- Trains multiple ML models
- Performs hyperparameter tuning
- Generates evaluation metrics and visualizations
- Saves the best performing model

### `web_app.py`
- Flask web application
- Provides REST API for URL analysis
- Loads trained model and makes predictions
- Serves the web interface

### `explore_dataset.py`
- Dataset exploration utility
- Displays dataset structure and statistics
- Helps understand the training data

## Security Features

The system identifies multiple security indicators:
- SSL certificate presence and validity
- HTTPS usage
- Suspicious domain patterns
- Phishing keyword detection
- Domain age analysis
- External resource loading
- Form security analysis

## Troubleshooting

### Common Issues

1. **Model not found error:**
   ```bash
   # Train the model first
   python train_model.py
   ```

2. **Features file not found:**
   ```bash
   # Extract features from training dataset
   python phishing_feature_extractor.py --batch
   ```

3. **Network timeouts:**
   - The system includes timeout handling for web requests
   - Some features may return default values for inaccessible sites

4. **Permission errors:**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   ```

### Dependencies

Key Python packages:
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning models
- `requests`: HTTP requests
- `beautifulsoup4`: HTML parsing
- `tldextract`: Domain parsing
- `whois`: Domain information
- `flask`: Web interface

## Contributing

To extend the system:

1. **Add new features**: Modify `PhishingFeatureExtractor` class
2. **Add new models**: Update the `models` dictionary in `train_model.py`
3. **Improve web interface**: Modify `templates/index.html`
4. **Add new endpoints**: Extend `web_app.py`

## Performance Optimization

For large-scale deployment:
- Use caching for WHOIS and DNS lookups
- Implement asynchronous feature extraction
- Add database storage for results
- Use load balancing for the web interface

## License

This project is developed for academic and research purposes related to cybersecurity and phishing detection.

---

**Note**: This system is designed to assist in identifying potential phishing threats. Always exercise caution when visiting unknown websites and verify suspicious sites through multiple sources.
