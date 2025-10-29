Hybrid Phishing Detector - Docker Quickstart

Files in this folder:
- hybrid.py: main script
- Dockerfile: container build recipe
- requirements.txt: Python dependencies
- docker-compose.yml: one-command runner

Prerequisites:
- Docker Desktop installed and running

How to run:
1) Build and run with compose (recommended):
   docker compose up --build

   Environment variables set by compose:
   - DATASET_PATH=/app/backend/dataset/combined_dataset.csv
   - SHORTLIST_DIR=/app/backend/dataset/PS-02_Shortlisting_set

   The compose file mounts this project directory into /app so local datasets are visible.

2) Alternatively, plain Docker:
   docker build -t hybrid-detector .
   docker run --rm -v "%cd%":/app -e DATASET_PATH=/app/backend/dataset/combined_dataset.csv -e SHORTLIST_DIR=/app/backend/dataset/PS-02_Shortlisting_set hybrid-detector

Outputs:
- Submission files will be created under PS-02_AIGR-123456_Submission in the mounted project directory.
# 🛡️ Phishing Detection System

A comprehensive AI-powered phishing detection system designed to identify and classify phishing domains targeting Critical Sector Entities (CSEs). The system combines advanced feature extraction, machine learning models, and real-time monitoring to provide robust security analysis.

## ✨ Features

### 🔍 **Core Detection Capabilities**
- **Whitelist Protection**: Instant recognition of legitimate domains
- **Advanced ML Models**: Random Forest classifier with 99.5% accuracy
- **Real-time Analysis**: Live domain classification and monitoring
- **Multi-factor Analysis**: URL structure, domain age, SSL status, content analysis
- **CSE Auto-detection**: Automatic identification of target entities

### 🎯 **Critical Sector Coverage**
- Banking & Financial Services (SBI, ICICI, HDFC, PNB, BoB)
- Government Services (NIC, RGCCI, IRCTC)
- Telecommunications (Airtel)
- Energy (IOCL)

### 🖥️ **User Interface**
- Modern responsive dashboard
- Real-time domain classification
- Monitoring system controls
- CSE management interface
- Recent detections tracking

## 🏗️ Project Architecture

```
Phishing_Detection/
├── backend/                    # Flask API Backend
│   ├── app.py                 # Main Flask application
│   ├── phishing_detection_system.py  # Core detection logic
│   ├── phishing_feature_extractor.py # Feature extraction engine
│   ├── cse_manager.py         # CSE management
│   ├── domain_discovery.py    # Domain discovery engine
│   ├── train_model.py         # Model training
│   ├── quick_train_model.py   # Quick setup training
│   ├── cse_whitelist.json     # CSE configuration
│   ├── requirements.txt       # Python dependencies
│   └── models/                # Trained ML models
├── frontend/                   # Next.js Frontend
│   ├── src/
│   │   ├── app/              # App router pages
│   │   ├── components/ui/    # Shadcn/ui components
│   │   └── lib/              # API client & utilities
│   ├── package.json          # Node.js dependencies
│   └── next.config.ts        # Next.js configuration
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **Git** for cloning

### 📥 Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/Navaneeth210805/Phishing_Detection.git
cd Phishing_Detection
```

#### 2. Backend Setup (Flask API)
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt

# Train the initial model
cd backend
python quick_train_model.py
cd ..
```

#### 3. Frontend Setup (Next.js)
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Build the application
npm run build

# Return to root directory
cd ..
```

### 🏃‍♂️ Running the Application

#### Option 1: Development Mode
```bash
# Terminal 1: Start Backend API
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
cd backend
python app.py

# Terminal 2: Start Frontend Dev Server
cd frontend
npm run dev
```

#### Option 2: Production Mode
```bash
# Start Backend
.venv\Scripts\activate    # Windows
cd backend
python app.py

# Start Frontend (production build)
cd frontend
npm start
```

### 🌐 Access the Application
- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Documentation**: Available at startup in backend terminal

## 🔧 Configuration

### Environment Variables
Create `.env` files in respective directories:

**Backend (.env)**
```env
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000
```

**Frontend (.env.local)**
```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:5000
```

### CSE Management
Edit `backend/cse_whitelist.json` to modify Critical Sector Entities:
```json
{
  "Your Bank Name": {
    "sector": "BFSI",
    "whitelisted_domains": ["yourbank.com", "online.yourbank.com"],
    "keywords": ["yourbank", "online banking"],
    "description": "Your Bank Description"
  }
}
```

## 🎯 Usage Guide

### 🔍 Domain Classification
1. **Navigate to Dashboard**: Open http://localhost:3000
2. **Enter Domain**: Type domain without protocol (e.g., `sbi.co.in`)
3. **Select CSE**: Choose target entity or use Auto-detect
4. **Classify**: Click button or press Enter
5. **Review Results**: Check classification, confidence, and reasoning

### 📊 Monitoring System
1. **Start Monitoring**: Click "Start" in Monitoring Status
2. **View Progress**: Monitor real-time domain discovery
3. **Check Results**: Review detected suspicious domains
4. **Stop/Restart**: Control monitoring as needed

### 🏢 CSE Management
1. **View Entities**: Navigate to CSE Management tab
2. **Add CSE**: Click "Add New CSE" and fill details
3. **Manage Domains**: Edit whitelisted domains
4. **Update Keywords**: Modify detection keywords

## 🧪 Testing

### Quick Tests
```bash
# Test Backend API
curl http://localhost:5000/api/health

# Test Domain Classification
curl -X POST http://localhost:5000/api/domains/classify \
  -H "Content-Type: application/json" \
  -d '{"domain": "sbi.co.in", "target_cse": "State Bank of India (SBI)"}'
```

### Sample Domains for Testing
- **Legitimate**: `sbi.co.in`, `icicibank.com`, `hdfcbank.com`
- **Suspicious**: `sbi-bank-login.com`, `icicbank.com`
- **Custom**: Your own portfolio/test domains

## 🛠️ Development

### Model Retraining
```bash
cd backend
python train_model.py  # Full training with custom dataset
python quick_train_model.py  # Quick synthetic data training
```

### API Development
- **Base URL**: `http://localhost:5000/api`
- **Health Check**: `GET /health`
- **Domain Classification**: `POST /domains/classify`
- **System Status**: `GET /system/status`
- **CSE Management**: `GET/POST/DELETE /cses`

### Frontend Development
```bash
cd frontend
npm run dev        # Development server
npm run build      # Production build
npm run lint       # Code linting
npm run type-check # TypeScript checking
```

## 📋 Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install --force-reinstall -r backend/requirements.txt

# Check port availability
netstat -an | findstr :5000  # Windows
lsof -i :5000  # macOS/Linux
```

#### Frontend Build Errors
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json  # macOS/Linux
# rmdir /s node_modules & del package-lock.json  # Windows
npm install

# Check Node.js version
node --version  # Should be 18+
```

#### Model Loading Issues
```bash
cd backend
# Retrain model
python quick_train_model.py

# Check model file
ls -la *.pkl  # Should see phishing_detection_model.pkl
```

#### CORS Issues
- Ensure backend CORS is configured for frontend URL
- Check API client configuration in `frontend/src/lib/api.ts`
- Verify environment variables

### Performance Optimization
- **Backend**: Use production WSGI server (gunicorn)
- **Frontend**: Use `npm run build` for production
- **Database**: Add Redis for caching (future enhancement)
- **Monitoring**: Implement proper logging and metrics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** for machine learning capabilities
- **Next.js** for the modern frontend framework
- **Flask** for the lightweight backend API
- **Shadcn/ui** for beautiful UI components
- **Critical Sector Entities** for security requirements

---

**🚨 Security Notice**: This system is designed for educational and research purposes. For production deployment, ensure proper security hardening, authentication, and monitoring.

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
