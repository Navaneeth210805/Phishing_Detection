Phishing Detection System - Technical Documentation
---

## 1. Participant Details and Skill Sets

### Team Information
- **Team Name:** Shradha
- **Category:** Student Group
- **Institution/Organization:** Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram
- **Team Size:** 3

### Skill Sets and Expertise

**Machine Learning & AI:**
- Deep Learning with PyTorch framework
- Neural Network Architecture Design
- Feature Engineering and Selection
- Classification Algorithms and Ensemble Methods
- Model Training, Validation, and Hyperparameter Tuning

**Software Development:**
- Python Programming and Data Science Libraries
- Web Application Development (Flask, HTML/CSS/JavaScript)
- Database Design and Management
- API Development and Integration

**Data Analysis & Visualization:**
- Large-scale Data Processing (Pandas, NumPy)
- Statistical Analysis and Performance Metrics
- Data Visualization and Reporting
- Excel/CSV Data Manipulation
- Batch Processing and Performance Optimization

---

## 2. Detailed Problem Statement

### 2.1 Challenge Overview

The challenge involves developing an AI/ML-based solution to detect phishing domains and URLs targeting 10 Critical Sector Entities (CSEs) in India. The system must process large-scale datasets (up to 1 million domains) and accurately classify domains as "Phishing" or "Suspected" based on their similarity to legitimate CSE infrastructure.

### 2.2 Target Critical Sector Entities

The system focuses on detecting threats against these 10 CSEs:

**Banking Sector:**
1. State Bank of India (SBI)
2. ICICI Bank
3. HDFC Bank
4. Punjab National Bank (PNB)
5. Bank of Baroda (BoB)

**Government & Infrastructure:**
6. National Informatics Centre (NIC)
7. Registrar General and Census Commissioner of India (RGCCI)
8. Indian Railway Catering and Tourism Corporation (IRCTC)
9. Indian Oil Corporation Limited (IOCL)

**Telecommunications:**
10. Airtel

### 2.3 Technical Challenges

**Scale and Volume:**
- Processing datasets containing 1+ million domains efficiently
- Maintaining high throughput (10,000-100,000 domains/hour)
- Memory-efficient batch processing without resource exhaustion

**Classification Complexity:**
- Distinguishing between legitimate domains, suspicious domains, and active phishing
- Handling subtle typosquatting techniques and homograph attacks
- Managing false positive rates to avoid overwhelming investigators

**Multi-Entity Targeting:**
- Supporting diverse naming conventions across different CSE sectors
- Adapting to rebranding, new services, and domain variations
- Balancing detection sensitivity per CSE without cross-contamination

**Evidence and Reporting:**
- Collecting actionable evidence including visual proof and registration metadata
- Generating legally defensible reports for incident response teams
- Maintaining chain of custody for detected threats

---

## 3. Proposed Approach and Scope

### 3.1 Hybrid Detection Methodology

Our solution employs a **two-stage hybrid approach** combining pattern-based filtering with machine learning classification to achieve both high accuracy and computational efficiency.

### 3.2 Stage 1: Pattern-based CSE Mapping

**Objective:** Filter and map domains to target CSE organizations using pattern recognition

**Key Algorithms:**
- **Keyword Matching:** Multi-tier scoring system evaluating exact matches, prefix/suffix matches, and substring detection
- **Typosquatting Detection:** Levenshtein distance similarity scoring against official CSE domains
- **Confidence Scoring:** Weighted combination of keyword and similarity scores with threshold filtering

**Benefits:**
- Reduces computational load by filtering irrelevant domains early (only ~0.5-2% proceed to ML)
- Provides explainable mapping rationale for investigation teams
- Handles known attack patterns with deterministic patterns

### 3.3 Stage 2: Machine Learning Classification

**Objective:** Classify CSE-mapped domains as "Phishing" or "Suspected" using deep learning

**Key Components:**
- **Feature Engineering:** 51 numerical features across 5 categories (basic, entropy, TLD, CSE patterns, lexical)
- **Neural Network:** 4-layer feedforward architecture with batch normalization and dropout
- **Training Strategy:** Weighted sampling for class imbalance, stratified train/test split, learning rate scheduling

**Benefits:**
- Provides probabilistic confidence scores for prioritization
- Adapts through retraining on new labeled examples

### 3.4 Detection Logic and Research Foundation

**Shannon Entropy Analysis:**
Quantifies randomness in domain strings to detect algorithmically generated phishing domains
```
H(X) = -Σ p(x) × log₂(p(x))
```

**Levenshtein Similarity:**
Measures edit distance between suspicious and legitimate domains for typosquatting detection
```
similarity = 1 - distance(s1, s2) / max(len(s1), len(s2))
```

**Jaccard Similarity:**
Character-level similarity for CSE pattern matching
```
J(A,B) = |A ∩ B| / |A ∪ B|
```

### 3.5 Scope and Limitations

**In Scope:**
- Domain-based phishing detection and classification
- Evidence collection (screenshots and WHOIS data)
- Batch processing of large datasets
- Comprehensive reporting and submission generation
- Continuous monitoring capability through re-evaluation

**Out of Scope:**
- Real-time URL analysis during web browsing
- Content analysis of live phishing pages
- Email-based phishing detection
- Mobile application phishing detection
- Advanced evasion techniques requiring dynamic analysis

---

## 4. Architecture

### 4.1 System Architecture Overview

The Hybrid Phishing Detection System follows a modular pipeline architecture designed for scalability, maintainability, and performance optimization.

### 4.2 Core Component Architecture

#### 4.2.1 CSE Mapper Module
- **Function:** Pattern-based domain-to-organization mapping
- **Algorithm:** Multi-tier keyword scoring + Levenshtein similarity
- **Input:** Domain name string
- **Output:** CSE name and confidence score (0-100)

#### 4.2.2 Feature Extractor Module  
- **Function:** Convert domain strings to numerical feature vectors
- **Algorithm:** Statistical and lexical analysis across 5 feature categories
- **Input:** Domain name string
- **Output:** Dictionary of 51 normalized features

#### 4.2.3 Neural Network Classifier
- **Function:** Binary classification (Phishing vs Suspected)
- **Architecture:** 4-layer feedforward network (51→128→64→32→16→2)
- **Input:** 51-dimensional feature vector
- **Output:** 2-class probability distribution

#### 4.2.4 Evidence Collection Framework
- **Function:** Automated evidence gathering for detected threats
- **Components:** Screenshot capture (Playwright) + WHOIS queries
- **Input:** Detected domain list
- **Output:** PDF evidence files with metadata

### 4.3 Data Flow Architecture

![Data Flow Diagram](data_flow_architecture.png)
*Figure 1: Data flow diagram showing processing pipeline from domain input to evidence output*

**Processing Pipeline:**
1. **Data Ingestion** → Load shortlisting dataset (Excel/CSV)
2. **CSE Mapping** → Apply Pattern-based filtering (reduces 1M → ~5K domains)
3. **Feature Extraction** → Convert domains to 51-dimensional vectors
4. **ML Classification** → Neural network inference with confidence scoring
5. **Evidence Collection** → Screenshot capture and WHOIS queries
6. **Report Generation** → Excel/CSV submission files with metadata

### 4.4 Deployment Architecture


**Scalability Features:**
- **Horizontal Scaling:** Batch processing supports domain list partitioning
- **Vertical Scaling:** Memory-efficient processing handles million+ domains
- **GPU Acceleration:** CUDA support for neural network inference
- **Distributed Storage:** Evidence files organized by CSE for parallel access

---

## 5. Implementation Details

### 5.1 Technologies and Frameworks

**Core Technologies:**
- **Python 3.9+:** Primary development language
- **PyTorch 1.8+:** Deep learning framework for neural network implementation
- **Pandas & NumPy:** Data manipulation and numerical computation
- **Scikit-learn:** Data preprocessing, scaling, and evaluation metrics

**Specialized Libraries:**
- **tldextract:** Domain parsing and TLD extraction
- **python-Levenshtein:** Fast string similarity computation
- **Playwright:** Browser automation for screenshot capture
- **python-whois:** Domain registration data extraction
- **joblib:** Model serialization and persistence

### 5.2 Feature Engineering Implementation

#### 5.2.1 Feature Categories (51 Total Features)

**Category 1: Basic Features (15 features)**
```python
# Domain structural analysis
domain_length, dot_count, dash_count, underscore_count
digit_count, uppercase_count, special_char_ratio
domain_parts, longest_part, shortest_part, avg_part_length
vowel_count, consonant_count, has_consecutive_chars
```

**Category 2: Entropy Features (10 features)**
```python
# Randomness quantification using Shannon entropy
char_entropy = -Σ(p(x) × log₂(p(x)))
bigram_entropy, transition_entropy, unique_char_ratio
trigram_count, unique_trigram_ratio, vowel_consonant_ratio
part_length_variance, part_length_std, consonant_clusters
```

**Category 3: TLD Features (5 features)**
```python
# Top-level domain classification
tld_length, is_common_tld, is_country_tld
is_suspicious_tld, is_indian_tld
```

**Category 4: CSE Pattern Features (10 features)**
```python
# Similarity to legitimate CSE infrastructure
legitimate_domain_similarity  # Jaccard similarity
legitimate_keyword_count, contains_legitimate_keyword
brand_keyword_count, finance_keywords, gov_keywords
telecom_keywords, tech_keywords
```

**Category 5: Lexical Features (11 features)**
```python
# Advanced pattern detection
char_repetition_score, has_year_pattern
starts_with_number, ends_with_number, max_numeric_sequence
suspicious_pattern_count, has_homograph, complexity_score
dictionary_word_count, has_subdomain, subdomain_count
```

### 5.3 Neural Network Implementation

#### 5.3.1 Architecture Design

![Neural Network Architecture](neural_network_architecture.png)
*Figure 2: Neural network architecture showing layer dimensions and activation functions*

```python
class PhishingNet(nn.Module):
    def __init__(self, input_size=51, hidden_sizes=[128, 64, 32, 16], num_classes=2):
        super(PhishingNet, self).__init__()
        
        # Input layer: 51 features → 128 neurons
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        
        # Hidden layers with batch normalization and dropout
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  # 128 → 64
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])  # 64 → 32
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])  # 32 → 16
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        
        # Output layer: 16 → 2 classes
        self.fc5 = nn.Linear(hidden_sizes[3], num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
```

#### 5.3.2 Training Configuration

**Hyperparameters:**
- **Epochs:** 300 (with early stopping)
- **Batch Size:** 32
- **Learning Rate:** 0.001 with ReduceLROnPlateau scheduler
- **Optimizer:** Adam with weight decay (1e-4)
- **Loss Function:** Weighted CrossEntropyLoss for class imbalance

**Data Preparation:**
- **Train/Test Split:** 80/20 with stratification
- **Feature Normalization:** StandardScaler (mean=0, std=1)
- **Class Balancing:** WeightedRandomSampler during training

### 5.4 CSE Mapping Algorithm

#### 5.4.1 Multi-tier Scoring System

```python
def map_domain_to_cse(self, domain):
    """
    Maps domain to CSE using keyword matching + typosquatting detection
    Returns: (cse_name, confidence_score) or (None, 0)
    """
    
    # Step 1: Parse domain components
    extracted = tldextract.extract(domain)
    domain_part = extracted.domain.lower()
    
    # Step 2: Keyword scoring for each CSE
    for cse_name, cse_info in self.cse_data.items():
        score = 0
        
        # Keyword matching with weighted scoring
        for keyword in cse_info['keywords']:
            if domain_part == keyword:  # Exact match
                score += 30
            elif domain_part.startswith(keyword) or domain_part.endswith(keyword):
                score += 25  # Prefix/suffix match
            elif keyword in domain_part:
                score += 20  # Substring match
        
        # Step 3: Typosquatting detection using Levenshtein similarity
        for official_domain in cse_info['domains']:
            similarity = Levenshtein.ratio(domain_part, official_domain.split('.')[0])
            if similarity >= 0.75:
                score += int(similarity * 30)
    
    # Return highest scoring CSE if above threshold
    return max_cse if max_score >= 20 else (None, 0)
```

### 5.5 Evidence Collection Implementation

#### 5.5.1 Screenshot Capture System

```python
async def capture_screenshot(self, domain, cse_name):
    """
    Automated screenshot capture using Playwright browser automation
    """
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Try HTTPS first, fallback to HTTP
        try:
            await page.goto(f"https://{domain}", timeout=30000)
        except:
            await page.goto(f"http://{domain}", timeout=30000)
        
        # Capture full-page screenshot
        screenshot = await page.screenshot(full_page=True)
        
        # Convert to PDF with metadata overlay
        return self.create_evidence_pdf(screenshot, domain, cse_name)
```

#### 5.5.2 WHOIS Data Collection

```python
def collect_whois(self, domain):
    """
    Extract domain registration metadata with timeout protection
    """
    import whois
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(whois.whois, domain)
            whois_data = future.result(timeout=5)  # 5-second timeout
            
            return {
                'registration_date': whois_data.creation_date,
                'registrar': whois_data.registrar,
                'registrant': whois_data.name,
                'country': whois_data.country,
                'name_servers': whois_data.name_servers
            }
    except (TimeoutError, Exception):
        return {}  # Return empty dict if WHOIS fails
```

### 5.6 Datasets and Training Data

#### 5.6.1 Training Dataset Structure

**File:** `[your_path]/dataset/combined_dataset.csv`
- **Total Samples:** 4,630 labeled domains
- **Columns:** `['url', 'label']`
- **Labels:** 'Phishing' (3,860 samples, 83.4%) and 'Suspected' (770 samples, 16.6%)
- **Source:** Combination of known phishing domains and suspicious CSE-related domains

#### 5.6.2 Shortlisting Dataset

**File:** `[your_path]/dataset/PS-02_Shortlisting_set/`
- **Format:** Excel (.xlsx) or CSV (.csv)
- **Size:** Up to 1 million domains
- **Content:** Single column containing domain names for classification
- **Purpose:** Target dataset for phishing detection

---

## 6. Scalability of the Solution

### 6.1 Horizontal Scaling Capabilities

**Component-Level Parallelization:**
- **CSE Mapping:** Domain list partitioning across multiple workers
- **Feature Extraction:** Embarrassingly parallel across domain batches
- **Model Inference:** Batch processing with GPU acceleration
- **Evidence Collection:** Concurrent screenshot and WHOIS queries

**Distributed Processing Framework:**
```python
# Example: Distributed domain processing
from multiprocessing import Pool
import numpy as np

def process_domain_batch(domain_batch):
    """Process batch of domains through full pipeline"""
    results = []
    for domain in domain_batch:
        cse_name, cse_score = cse_mapper.map_domain_to_cse(domain)
        if cse_name:
            features = feature_extractor.extract_features(domain)
            classification, confidence = model.predict(features)
            results.append((domain, classification, confidence, cse_name))
    return results

# Parallel execution across CPU cores
domain_batches = np.array_split(domain_list, num_cores)
with Pool(processes=num_cores) as pool:
    batch_results = pool.map(process_domain_batch, domain_batches)
```

### 6.2 Data Volume Scalability

**Current Performance Metrics:**
- **Processing Speed:** 4,496 domains/second on standard hardware
- **Memory Usage:** ~500MB for 1 million domains
- **Throughput:** 10,000-100,000 domains/hour depending on evidence collection

**Scaling Projections:**
- **10M domains:** ~2-4 hours processing time
- **100M domains:** Distributed across 10 workers, ~24 hours
- **1B domains:** Enterprise deployment with 100+ workers

### 6.3 CSE Expansion Scalability

**Current Support:** 10 CSEs with 6-8 keywords each
**Scaling Capability:** 100+ CSEs through configuration expansion

```python
# Example: Adding new CSE to system
new_cse_data = {
    "Reserve Bank of India (RBI)": {
        "keywords": ["rbi", "reserve", "bank", "india", "monetary", "policy"],
        "domains": ["rbi.org.in", "rbidocs.rbi.org.in"],
        "confidence_threshold": 25
    }
}

# Automatic integration without code changes
cse_mapper.cse_data.update(new_cse_data)
```

### 6.4 Continuous Monitoring Scalability

**State Management:**
- Stateless design enables re-evaluation without storage overhead
- External scheduling supports flexible monitoring intervals
- Evidence versioning through timestamped file naming

**Monitoring Workflow:**
```python
# Periodic re-evaluation of detected domains
def continuous_monitoring(detected_domains, interval_days=30):
    """
    Re-evaluate previously detected domains for status changes
    """
    for domain in detected_domains:
        # Re-run classification pipeline
        current_classification = detector.predict_phishing(domain)
        
        # Compare with historical classification
        if classification_changed(domain, current_classification):
            generate_alert(domain, current_classification)
            update_evidence(domain)
```

### 6.5 Integration Scalability

**Multi-Source Data Integration:**
- Certificate Transparency log feeds
- Passive DNS databases
- Threat intelligence platforms
- Domain registration monitors

**API Integration Points:**
```python
# Example: Integration with external threat feeds
class ThreatFeedIntegrator:
    def __init__(self, feed_urls):
        self.feed_urls = feed_urls
    
    def fetch_new_domains(self):
        """Fetch domains from multiple threat intelligence sources"""
        new_domains = []
        for feed_url in self.feed_urls:
            domains = self.parse_threat_feed(feed_url)
            new_domains.extend(domains)
        return new_domains
    
    def process_threat_feed(self, domains):
        """Process new domains through detection pipeline"""
        return detector.test_on_shortlisting(domains)
```

---

## 7. Resources Used for Detection

### 7.1 Computing Resources

**Hardware Requirements:**

*Minimum Configuration:*
- **CPU:** 4-core processor (Intel i5 or AMD Ryzen 5)
- **RAM:** 4GB available memory
- **Storage:** 10GB disk space for datasets and models
- **Network:** Internet connection for WHOIS queries and screenshots

*Recommended Configuration:*
- **CPU:** 8-core processor (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM:** 8GB+ for large dataset processing
- **Storage:** 50GB SSD for faster I/O operations
- **GPU:** NVIDIA GPU with CUDA support (optional, for training acceleration)
- **Network:** Stable broadband connection (100+ Mbps)

*Enterprise Configuration:*
- **CPU:** 16-32 core server processors
- **RAM:** 32GB+ for distributed processing
- **Storage:** 500GB+ NVMe SSD storage
- **GPU:** Multiple NVIDIA Tesla/A100 GPUs for training
- **Network:** High-bandwidth connection (1+ Gbps)

### 7.2 Software Dependencies

**Core Python Libraries:**
```bash
# Essential dependencies
pip install pandas==1.5.0 numpy==1.24.0 torch==2.0.0 
pip install scikit-learn==1.3.0 tldextract==3.4.0
pip install python-Levenshtein==0.20.0 joblib==1.3.0
```

**Evidence Collection Dependencies:**
```bash
# Optional for evidence collection
pip install playwright==1.30.0 python-whois==0.8.0
pip install Pillow==9.5.0 reportlab==4.0.0 openpyxl==3.1.0

# Browser installation for screenshots
playwright install chromium
```

### 7.3 Data Sources and Access

**Training Data Sources:**
- **PhishTank Database:** Known phishing URLs
- **VirusTotal Intelligence:** Malicious domain feeds
- **CSE Official Domains:** Legitimate domain references
- **Manual Labeling:** Expert-verified suspicious domains

**External API Access:**
- **WHOIS Services:** Domain registration metadata
- **DNS Resolution:** Domain-to-IP mapping
- **Certificate Transparency:** SSL certificate logs
- **Threat Intelligence Feeds:** Real-time malicious domain updates

### 7.4 Processing Performance Metrics

**Current Benchmarks (1M domain dataset):**
- **Total Processing Time:** ~4 minutes
- **CSE Mapping Rate:** 780 domains (0.07%)
- **Classification Speed:** 4,496 domains/second
- **Memory Peak Usage:** 512MB
- **Evidence Collection:** +30 seconds per domain (optional)

---

## 8. How to Setup and Run the Solution

### 8.1 Installation Instructions

#### 8.1.1 Prerequisites

**System Requirements:**
- Operating System: Linux, macOS, or Windows 10/11
- Python Version: 3.8 or higher
- Available RAM: 4GB minimum, 8GB recommended
- Disk Space: 10GB minimum for installation and datasets

#### 8.1.2 Environment Setup

**Step 1: Clone Repository**
```bash
# On Master Branch
git clone https://github.com/Navaneeth210805/Phishing_Detection.git
cd Phishing_Detection
git checkout master
```

**Step 2: Create Virtual Environment**
```bash
# Using conda (recommended)
conda create -n phishing_detection python=3.9
conda activate phishing_detection

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

**Step 3: Install Dependencies**
```bash
# Install core dependencies
pip install -r requirements.txt

# Install additional dependencies for evidence collection
pip install playwright python-whois Pillow reportlab openpyxl

# Install browser for screenshot capture
playwright install chromium
```

### 8.2 Configuration Setup

#### 8.2.1 Directory Structure Verification

```bash
# Verify required directories exist
mkdir -p your_folder/dataset/
mkdir -p your_folder/
mkdir -p PS-02_AIGR-123456_Submission/
```

#### 8.2.2 Dataset Preparation

**Training Dataset:**
```bash
# Place training data in correct location
cp your_training_data.csv yourfolder/combined_dataset.csv

# Verify format: CSV with columns ['url', 'label']
head yourfolder/combined_dataset.csv
```

**Shortlisting Dataset:**
```bash
# Place shortlisting data in dataset directory
cp PS-02_Shortlisting_set.xlsx yourfolder/PS-02_Shortlisting_set/
# Or: cp PS-02_Shortlisting_set.csv yourfolder/PS-02_Shortlisting_set/
```

### 8.3 Running the Solution

#### 8.3.1 Complete Pipeline Execution

**Full Training and Detection Pipeline:**
```bash
# Navigate to hybrid detection script
cd [your_folder]/

# Run complete pipeline (training + detection)
python main.py

# Expected output:
# Starting training phase...
# Loading dataset from: [your_folder]/dataset/combined_dataset.csv
# Training neural network for 300 epochs...
# Model saved to: yourfolder/hybrid_model.pkl
# Starting detection phase...
# Processing shortlisting dataset...
# Results saved to: PS-02_AIGR-123456_Submission/
```

#### 8.3.2 Step-by-Step Execution

**Training Only:**
```python
from hybrid import HybridPhishingDetector

# Initialize detector
detector = HybridPhishingDetector(application_id="AIGR-123456")

# Train model on combined dataset
detector.train_model(
    dataset_path="yourfolder/combined_dataset.csv",
    epochs=300
)
print("Training completed. Model saved to yourfolder/hybrid_model.pkl")
```

**Detection Only (using pre-trained model):**
```python
# Load pre-trained model
detector = HybridPhishingDetector(application_id="AIGR-123456")
detector.load_model('yourfolder/hybrid_model.pkl')

# Run detection on shortlisting data
detector.test_on_shortlisting(
    shortlisting_dir="yourfolder/PS-02_Shortlisting_set"
)
print("Detection completed. Results in PS-02_AIGR-123456_Submission/")
```

#### 8.3.3 Single Domain Testing

**Interactive Testing:**
```python
# Test individual domains
detector = HybridPhishingDetector()
detector.load_model('yourfolder/hybrid_model.pkl')

# Test domain classification
domain = "sbibank-login.com"
cse_name, cse_score = detector.cse_mapper.map_domain_to_cse(domain)

if cse_name:
    classification, confidence = detector.predict_phishing(domain)
    print(f"Domain: {domain}")
    print(f"CSE Target: {cse_name} (score: {cse_score})")
    print(f"Classification: {classification} (confidence: {confidence:.2%})")
else:
    print(f"Domain {domain} not targeting any CSE")
```

### 8.4 Configuration Options

#### 8.4.1 Modifying System Parameters

**Application ID and Output Paths:**
```python
# Customize submission identifier
detector = HybridPhishingDetector(application_id="CUSTOM-ID-789")

# Custom output directory
detector.submission_folder = "custom_output_folder"
```

**Training Parameters:**
```python
# Adjust training configuration
detector.train_model(
    dataset_path="path/to/dataset.csv",
    epochs=500,           # Increase training epochs
    batch_size=64,        # Larger batch size
    learning_rate=0.0005  # Lower learning rate
)
```

**Evidence Collection Toggle:**
```python
# Enable/disable evidence collection in hybrid.py
COLLECT_SCREENSHOTS = True   # Set to False for faster processing
COLLECT_WHOIS = True        # Set to False to skip WHOIS queries
```

You can may refer to the README file for Docker usage

### 8.5 Output Interpretation

#### 8.5.1 Generated Files

**Submission Files:**
```
PS-02_AIGR-123456_Submission/
├── PS-02_AIGR-123456_Submission_Set.xlsx    # Main submission (Excel)
├── PS-02_AIGR-123456_Submission_Set.csv     # Main submission (CSV)
├── all_detected.csv                          # All detections summary
└── PS-02_AIGR-123456_Evidences/             # Evidence folder
    ├── SBI/                                  # Per-CSE evidence
    ├── ICICI/
    └── [other CSE folders]/
```

**Model Files:**
```
your_folder/
├── hybrid_model.pkl        # Complete model with scaler and metadata
└── hybrid_weights.pth      # PyTorch state dict only
```

#### 8.5.2 Understanding Results

**Submission File Columns:**
- **Application_ID:** Unique submission identifier
- **Identified_Domain:** Detected phishing/suspected domain
- **CSE_Name:** Target Critical Sector Entity
- **Classification:** "Phishing" or "Suspected"
- **Registration_Date:** Domain creation date (from WHOIS)
- **Evidence_File:** Screenshot filename reference
- **Remarks:** ML confidence % and CSE mapping score

### 8.6 Troubleshooting

#### 8.6.1 Common Issues

**Memory Errors:**
```bash
# Reduce batch size for large datasets
# Edit hybrid.py: batch_size = 16 (instead of 32)

# Or process in chunks
# Split large shortlisting files into smaller pieces
```

**Screenshot Failures:**
```bash
# Install browser dependencies
sudo apt-get install -y libnss3 libatk1.0 libdrm2 libxkbcommon0

# Verify Playwright installation
playwright --version
```

**WHOIS Timeouts:**
```bash
# Increase timeout in hybrid.py
# Edit collect_whois method: future.result(timeout=10)
```

#### 8.6.2 Performance Optimization

**GPU Acceleration:**
```python
# Verify CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

**Parallel Processing:**
```python
# Enable multiple workers for data loading
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4  # Increase for faster loading
)
```

---

## 9. Results

### 9.1 Model Performance Metrics

#### 9.1.1 Training Results

**Training Configuration Summary:**
- **Total Samples:** 4,630 labeled domains
- **Training Samples:** 3,704 domains (80%)
- **Test Samples:** 926 domains (20%)
- **Training Duration:** 300 epochs (~2 hours on CPU)
- **Final Architecture:** 51 → 128 → 64 → 32 → 16 → 2

**Class Distribution:**
- **Phishing:** 3,860 samples (83.4%) 
- **Suspected:** 770 samples (16.6%)
- **Class Weights Applied:** [Suspected: 3.006, Phishing: 0.600]

#### 9.1.2 Classification Performance

**Overall Metrics:**
- **Test Accuracy:** 90.6%
- **ROC-AUC Score:** 0.9752

**Detailed Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Suspected** | 0.65 | 0.97 | 0.77 | 154 |
| **Phishing** | 0.99 | 0.89 | 0.94 | 772 |
| **Accuracy** | - | - | **0.91** | 926 |
| **Macro Avg** | 0.82 | 0.93 | 0.86 | 926 |
| **Weighted Avg** | 0.93 | 0.91 | 0.91 | 926 |

**Confusion Matrix:**

|  | Predicted Suspected | Predicted Phishing |
|---|--------------------|--------------------|
| **Actual Suspected** | 149 (TN) | 5 (FP) |
| **Actual Phishing** | 82 (FN) | 690 (TP) |

**Performance Analysis:**
- **High Phishing Precision (99%):** When model predicts Phishing, it's correct 99% of the time
- **Excellent Suspected Recall (97%):** Catches 97% of all Suspected domains
- **Low False Positive Rate:** Only 5 Suspected domains misclassified as Phishing
- **Acceptable False Negative Rate:** 82 Phishing domains classified as Suspected (conservative approach)

### 9.2 Large-Scale Detection Results

#### 9.2.1 Shortlisting Dataset Analysis

**Dataset Statistics:**
- **Total Domains Processed:** 1,088,266 domains
- **Processing Time:** ~4 minutes
- **Processing Rate:** 4,496 domains/second
- **CSE-Mapped Domains:** 780 domains (0.07%)

#### 9.2.2 Detection Classification Summary

**Final Classification Results:**

| Classification | Count | Percentage |
|----------------|-------|------------|
| **Phishing** | 364 | 46.67% |
| **Suspected** | 416 | 53.33% |
| **Total Detected** | 780 | 100% |

#### 9.2.3 CSE-Wise Detection Analysis

**Detection Distribution by Target CSE:**

| Rank | CSE Name | Domains | Percentage | Risk Level |
|------|----------|---------|------------|------------|
| 1 | **State Bank of India (SBI)** | 383 | 49.10% | Critical |
| 2 | **National Informatics Centre (NIC)** | 169 | 21.67% | High |
| 3 | **Airtel** | 46 | 5.90% | Medium |
| 4 | **Indian Oil Corporation (IOCL)** | 44 | 5.64% | Medium |
| 5 | **Bank of Baroda (BoB)** | 38 | 4.87% | Medium |
| 6 | **Punjab National Bank (PNB)** | 33 | 4.23% | Medium |
| 7 | **HDFC Bank** | 26 | 3.33% | Medium |
| 8 | **ICICI Bank** | 26 | 3.33% | Medium |
| 9 | **RGCCI** | 8 | 1.03% | Low |
| 10 | **IRCTC** | 7 | 0.90% | Low |

**Sector-wise Analysis:**
- **Banking Sector:** 506 domains (64.9%) - SBI, ICICI, HDFC, PNB, BoB
- **Government Sector:** 177 domains (22.7%) - NIC, RGCCI  
- **Infrastructure/Telecom:** 97 domains (12.4%) - Airtel, IOCL, IRCTC

### 9.3 Domain Examples and Case Studies

#### 9.3.1 High-Confidence Phishing Examples

**State Bank of India (SBI) Targets:**
```
sbibank-online.com          → Phishing (confidence: 94%)
sbi-netbanking.in          → Phishing (confidence: 91%)
statebankofindia-login.com → Phishing (confidence: 88%)
```

**ICICI Bank Targets:**
```
icicibank-secure.net       → Phishing (confidence: 92%)
icici-netbanking.org       → Phishing (confidence: 89%)
icicidirect-login.com      → Phishing (confidence: 86%)
```

#### 9.3.2 Suspected Classification Examples

**Government Entity Targets:**
```
nic-gov-india.tk           → Suspected (confidence: 67%)
nicgov.ml                  → Suspected (confidence: 61%)
nationalinformatics.click  → Suspected (confidence: 59%)
```

#### 9.3.3 Feature Analysis Examples

**Example: `sbibank-login.com`**
- **CSE Mapping:** State Bank of India (score: 45)
- **Key Features:** High CSE similarity, financial keywords, suspicious TLD
- **Classification:** Phishing (confidence: 91%)

**Example: `nicgov.tk`**
- **CSE Mapping:** National Informatics Centre (score: 23)  
- **Key Features:** Government keywords, suspicious TLD (.tk), short domain
- **Classification:** Suspected (confidence: 64%)

### 9.4 Evidence Collection Results

#### 9.4.1 Screenshot Evidence Quality

**Successfully Captured:**
- **Total Screenshots:** 780 domains (when enabled)
- **Success Rate:** 89% (693 successful captures)
- **Failure Reasons:** Unreachable domains (8%), timeout errors (3%)


#### 9.4.2 WHOIS Data Analysis

**Registration Patterns:**
- **Recent Registrations:** 67% registered within last 6 months
- **Suspicious Registrars:** 43% using known bulk registrars
- **Geographic Distribution:** 78% registered in non-Indian countries

**Common Registration Indicators:**
```
Registration Date: 2025-10-15 (recent)
Registrar: GoDaddy.com (bulk registrar)
Registrant Country: US (non-Indian)
Name Servers: ns1.parkingcrew.net (parking service)
```

### 9.5 System Performance Analysis

#### 9.5.1 Computational Efficiency

**Processing Breakdown:**
- **CSE Mapping:** 0.15 seconds for 1M domains
- **Feature Extraction:** 2.1 seconds for 780 CSE-mapped domains  
- **ML Classification:** 0.8 seconds for 780 domains
- **Evidence Collection:** 15-30 seconds per domain (optional)


#### 9.5.2 Accuracy vs. Speed Trade-offs

**Configuration Options:**
- **Fast Mode:** CSE mapping only (~1M domains/hour)
- **Standard Mode:** CSE + ML classification (~100K domains/hour)
- **Evidence Mode:** Full pipeline with screenshots (~1K domains/hour)

---

## 10. Conclusion

### 10.1 Key Findings

#### 10.1.1 Technical Achievements

**High Classification Accuracy:**
Our hybrid phishing detection system achieved **90.6% test accuracy** with **0.9752 ROC-AUC**, demonstrating excellent discrimination between Phishing and Suspected domains. The **99% precision** for Phishing classification ensures high confidence in threat alerts while maintaining **97% recall** for Suspected domains, providing comprehensive coverage.

**Scalable Processing Performance:**
The system successfully processed **1,088,266 domains in 4 minutes** (4,496 domains/second), proving its capability for large-scale real-world deployment. The two-stage hybrid approach efficiently filtered 99.93% of irrelevant domains through pattern-based CSE mapping, enabling focused ML classification on only 780 relevant domains.

**Comprehensive Threat Coverage:**
Detection results revealed **SBI as the primary target** (49.1% of detections), highlighting the need for enhanced protection of major banking infrastructure. The system identified threats across all 10 target CSEs, with **banking sector comprising 64.9%** of total detections, indicating the critical nature of financial services protection.

#### 10.1.2 Practical Impact

**Operational Efficiency:**
The automated evidence collection framework captured screenshots and WHOIS metadata for 89% of detected domains, providing actionable intelligence for incident response teams. The structured reporting format enables seamless integration with existing security operations workflows.

**Early Threat Detection:**
Analysis of registration patterns showed 67% of detected domains were registered within the last 6 months, demonstrating the system's capability to identify emerging threats before they become active phishing campaigns.

**Multi-Sector Protection:**
The system's modular CSE mapping architecture successfully protected diverse sectors including banking (SBI, ICICI, HDFC), government (NIC, RGCCI), and critical infrastructure (IRCTC, IOCL, Airtel), proving its adaptability across different organizational types.

### 10.2 Limitations and Challenges

#### 10.2.1 Technical Limitations

**Domain-Based Detection Scope:**
The current system focuses exclusively on domain-level analysis and cannot detect content-based phishing techniques such as compromised legitimate websites hosting phishing pages. This limitation requires complementary content analysis for comprehensive protection.

**False Positive Management:**
While the 99% precision minimizes false positives, the conservative classification approach results in some legitimate domains being flagged as Suspected. Manual review workflows are necessary to distinguish between genuine threats and false positives.

**Real-Time Processing Constraints:**
Evidence collection (screenshots and WHOIS) introduces significant latency (~30 seconds per domain), making real-time analysis challenging for high-volume scenarios. Current architecture optimized for batch processing rather than stream processing.

#### 10.2.2 Dataset Dependencies

**Training Data Quality:**
Model performance directly depends on the quality and representativeness of the training dataset. Limited labeled examples for some CSEs may result in reduced detection accuracy for those specific targets.

**Feature Engineering Assumptions:**
The 51-feature model assumes current phishing techniques remain consistent. Novel evasion techniques or completely different attack vectors may require feature engineering updates and model retraining.

#### 10.2.3 Operational Constraints

**Network Dependencies:**
WHOIS queries and screenshot captures require reliable internet connectivity and are subject to rate limiting by external services. Offline operation limited to domain-based classification only.

**Browser Compatibility:**
Screenshot capture depends on Playwright browser automation, which may fail for domains using anti-automation measures or requiring specific browser configurations.

### 10.3 Future Improvements

#### 10.3.1 Technical Enhancements

**Deep Learning Architecture Expansion:**
- **Transformer Models:** Implement attention-based models for better sequence understanding of domain strings
- **Ensemble Methods:** Combine multiple ML models (Random Forest, SVM, XGBoost) with neural networks for improved accuracy
- **Active Learning:** Implement human-in-the-loop feedback mechanisms for continuous model improvement

**Advanced Feature Engineering:**
- **DNS-Based Features:** Incorporate DNS record analysis, TTL patterns, and authoritative server characteristics
- **Certificate Analysis:** Integrate SSL certificate metadata, validity periods, and issuer patterns
- **Behavioral Features:** Add temporal patterns, traffic analysis, and domain lifecycle characteristics

**Content Analysis Integration:**
```python
# Future enhancement: Content-based detection
class ContentAnalyzer:
    def analyze_webpage_content(self, domain):
        """
        Analyze webpage content for visual similarity to legitimate CSE sites
        """
        # Image similarity using perceptual hashing
        # HTML structure analysis
        # Logo detection and brand similarity
        # Form field analysis for credential harvesting
        pass
```

#### 10.3.2 Scalability Improvements

**Real-Time Stream Processing:**
- **Apache Kafka Integration:** Enable real-time domain feed processing
- **Microservices Architecture:** Decompose system into independent scalable services
- **Container Orchestration:** Kubernetes deployment for auto-scaling and fault tolerance

**Distributed Computing Framework:**
```python
# Future enhancement: Distributed processing
from dask.distributed import Client, as_completed

def distributed_detection(domain_list, num_workers=10):
    """
    Distribute phishing detection across multiple workers
    """
    client = Client(f'scheduler-address:{port}')
    
    # Scatter data across workers
    scattered_domains = client.scatter(domain_list)
    
    # Submit parallel detection tasks
    futures = client.map(detect_phishing, scattered_domains)
    
    # Collect results
    results = client.gather(futures)
    return results
```

**Cloud-Native Deployment:**
- **Auto-Scaling Groups:** Automatic resource scaling based on workload
- **Load Balancing:** Distribute detection requests across multiple instances
- **Managed Database:** Store detection results and historical data in cloud databases

#### 10.3.3 Extended Functionality

**Multi-Language Support:**
- **Internationalized Domain Names (IDN):** Enhanced Unicode domain analysis
- **Regional CSE Support:** Expand to CSEs in other countries and regions
- **Cross-Language Phishing:** Detect phishing targeting non-English speaking users

**Advanced Evidence Collection:**
- **Live Content Monitoring:** Periodic re-evaluation of detected domains for content changes
- **Social Media Integration:** Monitor social media platforms for phishing link propagation  
- **Email Campaign Detection:** Integrate with email security systems for comprehensive protection

**Threat Intelligence Integration:**
```python
# Future enhancement: Threat intelligence feeds
class ThreatIntelligenceIntegrator:
    def __init__(self):
        self.feeds = [
            'virustotal_api', 'shodan_api', 'alienvault_otx',
            'phishtank_api', 'openphish_feed'
        ]
    
    def enrich_detection(self, domain):
        """
        Enrich detection with external threat intelligence
        """
        threat_score = 0
        for feed in self.feeds:
            score = self.query_threat_feed(feed, domain)
            threat_score += score
        
        return threat_score
```

---

## 11. References

### 11.1 Software Libraries

1. **PyTorch:** Paszke, A., et al. "PyTorch: An imperative style, high-performance deep learning library." NeurIPS, 2019.

2. **Scikit-learn:** Pedregosa, F., et al. "Scikit-learn: Machine learning in Python." JMLR, 2011.

3. **Pandas:** McKinney, W. "Data structures for statistical computing in Python." SciPy, 2010.

4. **tldextract:** Domain parsing and TLD extraction library. GitHub Repository.

5. **Playwright:** Microsoft. Browser automation framework. Documentation.

### 11.2 Datasets and Tools

6. **PhishTank:** Collaborative clearing house for phishing data. Online Database.

7. **WHOIS Services:** Domain registration metadata providers.

---
