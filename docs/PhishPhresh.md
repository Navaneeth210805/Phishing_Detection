# Phishing Detection Model 🎣

A lightweight, high-performance deep learning classifier for detecting phishing URLs using a multi-modal approach (HTML + URL + Text features).

## 📊 Data Overview
- **Source**: `phreshphish/phreshphish` (Hugging Face)
- **Total Samples Used**: 10,000 (streaming subset of the full dataset)
- **Train/Test Split**: 80% Training (8,000) / 20% Testing (2,000)
- **Class Balance**: Validated on a mix of legitimate and phishing sites.

## 🧠 Model Architecture
The solution uses a **Multi-Modal Neural Network (`PhishingNet`)** implemented in PyTorch that fuses three distinct feature sets:

1. **HTML Features** (17 dims): Structural tags (forms, inputs, iframes), hidden elements, etc.
2. **URL Features** (19 dims): Domain entropy, special characters, TLD analysis, etc.
3. **Text Features** (1500 dims): TF-IDF/Hashing vectorizer on URL and HTML content.

These branches are processed independently via dense layers (with BatchNorm & Dropout) and fused into a final classification layer.

## 📈 Performance Results

### Training Metrics
- **Best Validation Accuracy**: 97.45%
- **Final Test Accuracy**: 97.35%
- **AUC**: ~0.9942

### Confusion Matrix & Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Benign** | 0.98 | 0.99 | 0.98 | 1353 |
| **Phishing** | 0.97 | 0.95 | 0.96 | 647 |

**Summary**:
- Total Test Samples: 2000
- Correct Predictions: 1947
- Incorrect Predictions: 53

## 🧪 Real-World Examples
| URL | Type | Confidence | Prediction |
|-----|------|------------|------------|
| `https://secure-verify-account.tk/login` | **Phishing** | 100.00% | ✅ Correct |
| `https://www.google.com/search` | **Benign** | 0.12% (Phishing prob) | ✅ Correct |

## 🚀 Usage
The model automatically saves best weights to `phishing_model.pth`. Test results with individual confidence scores are exported to `test_results.csv`.
