# Phishing Distribution Learning System

This system learns statistical distributions from known phishing URLs to create a robust phishing detection model. It analyzes various features of phishing URLs and learns their statistical patterns for future detection.

## Overview

The distribution learning system consists of three main components:

1. **`phishing_distribution_learner.py`** - Learns statistical distributions from phishing URLs
2. **`phishing_distribution_detector.py`** - Uses learned distributions to detect phishing URLs
3. **`quick_test_distributions.py`** - Quick test script to verify the system

## How It Works

### 1. Feature Extraction
The system extracts 50+ features from each phishing URL, including:
- **URL-based features**: length, character patterns, suspicious keywords
- **Domain-based features**: age, registration info, DNS records
- **Content-based features**: HTML structure, links, scripts, forms

### 2. Distribution Learning
For each feature, the system:
- Determines if the feature is binary, discrete, or continuous
- Fits the best statistical distribution (Normal, Gamma, Beta, etc.)
- Saves distribution parameters for future use

### 3. Phishing Detection
Uses the learned distributions to:
- Calculate likelihood scores for new URLs
- Combine feature likelihoods into an overall phishing score
- Classify URLs as phishing, suspicious, or legitimate

## Usage

### Learning Distributions from Phishing URLs

```bash
# Learn from all phishing URLs
python phishing_distribution_learner.py

# Learn from first 100 URLs (for testing)
python phishing_distribution_learner.py --max-urls 100

# Use custom phishing file
python phishing_distribution_learner.py --phishing-file /path/to/phishing_urls.txt
```

### Detecting Phishing URLs

```bash
# Analyze a single URL
python phishing_distribution_detector.py --url "http://suspicious-site.com"

# Analyze URLs from a file
python phishing_distribution_detector.py --file test_urls.txt

# Use custom threshold
python phishing_distribution_detector.py --url "http://example.com" --threshold 0.8
```

### Quick Test

```bash
# Run a quick test with 50 URLs
python quick_test_distributions.py
```

## Output Files

The system generates several output files in the `distributions/` directory:

### Core Distribution Files
- **`phishing_distributions.json`** - Human-readable distribution parameters
- **`phishing_distributions.pkl`** - Python pickle file for fast loading
- **`phishing_feature_stats.json`** - Statistical summaries for each feature

### Analysis Reports
- **`phishing_distribution_report.txt`** - Detailed analysis report
- **`plots/`** - Visualization plots for feature distributions

## Distribution Types

The system automatically determines the appropriate distribution type for each feature:

### Binary Features (Bernoulli Distribution)
- Features with only 0/1 values
- Examples: `has_ip_address`, `uses_https`, `has_suspicious_keywords`

### Discrete Features (Poisson/Empirical Distribution)
- Integer-valued features with limited range
- Examples: `subdomain_count`, `form_count`, `redirect_count`

### Continuous Features (Various Distributions)
- Real-valued features
- Automatically fits best distribution: Normal, Gamma, Beta, Log-normal, etc.
- Examples: `url_length`, `entropy`, `digit_ratio`

## Example Output

```
PHISHING DISTRIBUTION LEARNING SUMMARY
URLs processed: 50
Features analyzed: 52
Distributions learned: 52

Distribution types:
  continuous: 7 features
  binary: 40 features
  discrete: 5 features
```

## Detection Results

The detector provides detailed analysis for each URL:

```python
{
    'url': 'http://suspicious-site.com',
    'classification': 'PHISHING',
    'phishing_score': 0.873,
    'confidence': 0.924,
    'risk_level': 'HIGH',
    'feature_analysis': {
        'url_length': {'value': 156, 'likelihood': 0.023},
        'has_suspicious_keywords': {'value': 1, 'likelihood': 0.892},
        # ... more features
    }
}
```

## Classification Levels

- **PHISHING** (score ≥ 0.7, confidence ≥ 0.5): High-risk phishing
- **LIKELY_PHISHING** (score ≥ 0.7, confidence ≥ 0.3): Probable phishing
- **SUSPICIOUS** (score ≥ 0.5): Suspicious patterns
- **LOW_RISK** (score ≥ 0.3): Some risk indicators
- **LEGITIMATE** (score < 0.3): Likely legitimate

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- pandas, numpy, scipy - Data processing and statistics
- scikit-learn - Machine learning utilities
- matplotlib, seaborn - Visualization
- requests, beautifulsoup4 - Web scraping
- tldextract, python-whois - Domain analysis

## Performance Notes

- **Training time**: ~1-2 minutes for 50 URLs, ~30-60 minutes for 900 URLs
- **Detection time**: ~2-5 seconds per URL (including web requests)
- **Memory usage**: ~50MB for loaded distributions
- **Accuracy**: High precision on phishing URLs, some false positives on unusual legitimate sites

## Advanced Usage

### Custom Feature Selection

You can modify the feature extractor to focus on specific features:

```python
from phishing_distribution_learner import PhishingDistributionLearner

learner = PhishingDistributionLearner()
# Customize feature extraction here
learner.run_analysis()
```

### Integration with Other Systems

```python
from phishing_distribution_detector import PhishingDistributionDetector

detector = PhishingDistributionDetector()
result = detector.classify_url("http://example.com")
phishing_score = result['phishing_score']
```

### Batch Processing

```python
urls = ["http://site1.com", "http://site2.com"]
results = detector.batch_classify(urls)
```

## Troubleshooting

### Common Issues

1. **SSL/TLS Errors**: Some phishing sites have invalid certificates (expected)
2. **DNS Resolution Failures**: Phishing domains often get taken down
3. **Connection Timeouts**: Use reasonable timeout values (10-30 seconds)
4. **Memory Usage**: For large datasets, process in batches

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Files Created

After running the system, you'll have:

```
distributions/
├── phishing_distributions.json          # Human-readable distributions
├── phishing_distributions.pkl           # Python pickle file
├── phishing_feature_stats.json          # Feature statistics
├── phishing_distribution_report.txt     # Analysis report
└── plots/                               # Visualization plots
    ├── feature_distributions_summary.png
    ├── url_length_distribution.png
    ├── domain_length_distribution.png
    └── ... (individual feature plots)
```

## Next Steps

1. **Collect more phishing data** for better distribution accuracy
2. **Add legitimate URL data** for comparative analysis
3. **Implement ensemble methods** combining multiple detection approaches
4. **Create web API** for real-time phishing detection
5. **Add machine learning models** using the distributions as features

---

This distribution learning approach provides a robust foundation for phishing detection by understanding the statistical patterns that characterize phishing URLs.
