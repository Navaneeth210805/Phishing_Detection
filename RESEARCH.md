# URLPhishNet — Research References

All papers cited or relevant to this project, with links and notes on how each relates to our work.

---

## Core Papers We Compare Against

### 1. Sabir et al. (2020) — The Adversarial URL Benchmark Paper
- **Title**: Exposing Malicious URLs: A Comprehensive Adversarial Analysis of Phishing Detectors
- **arXiv**: https://arxiv.org/abs/2005.08454
- **Experiments 3, 4, 5** directly compare against this paper.
- Tested 50 models (RF, XGB, LGBM, URLNet, LSTM) on 3 URL-mutation attacks: domain, path, TLD adversary.
- Best normal accuracy in paper: **98.58%** (Basic Lexical + XGBoost)
- Best adversarially-trained accuracy: **97.71%** (Bigram + RF)
- We beat both: **99.18% normal (Exp 4)**, **99.25% adv-trained (Exp 5)**

### 2. AlEroud & Karabatis (2020) — The GAN Attack Paper
- **Title**: Bypassing Detection of URL-based Phishing Attacks Using Generative Adversarial Deep Neural Networks
- **DOI**: https://dl.acm.org/doi/10.1145/3375708.3380315
- **Experiments 8, 9** directly implement and then defend against this attack.
- **Datasets used in paper**:
  - PhishTank (phishing URLs) — community-verified phishing URL repository
  - Alexa Top Sites (legitimate URLs) — top 1M websites as benign baseline
- **PDs tested in paper** (all single-stream, feature-based):
  - Random Forest
  - Neural Network
  - SVM (Support Vector Machine)
- **What the GAN does**: takes 67-feature vector → binarizes to 134-dim → Generator flips bits
  to make phishing look benign to the PD. All 3 PDs were successfully fooled.
- **Evasion achieved in paper**: Near-complete evasion on all 3 PDs.
- Our replication (Exp 8) on FeatureMLP: **100% evasion confirmed**.

---

## Architecture References

### 3. Kim (2014) — TextCNN (basis for CharCNN)
- **Title**: Convolutional Neural Networks for Sentence Classification
- **arXiv**: https://arxiv.org/abs/1408.5882
- Our CharCNN with parallel kernels k=3,5,7 follows this design.
- Multiple parallel convolutional filters + global max pooling over sequence.

### 4. Le et al. (2018) — URLNet
- **Title**: URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection
- **arXiv**: https://arxiv.org/abs/1802.03162
- Dual-channel: char embeddings + word embeddings. No handcrafted features.
- We differ: char CNN + handcrafted 67 features (not word embeddings).

---

## Related Dual-Stream / Fusion Architectures (2023-2025)

### 5. CNN-Fusion (2023)
- **Title**: CNN-Fusion: An effective and lightweight phishing detection method based on multi-variant ConvNet
- **Link**: https://www.sciencedirect.com/science/article/abs/pii/S0020025523002281
- Multi-scale parallel char CNN. 99%+ accuracy. **No handcrafted feature branch** — single stream only.
- Closest char CNN architecture to ours; our novelty is adding the 67-feature MLP stream.

### 6. Multi-channel TCN Fusion (2025)
- **Title**: Comprehensive phishing detection: A multi-channel approach with variants TCN fusion leveraging URL and HTML features
- **Link**: https://www.sciencedirect.com/science/article/abs/pii/S1084804525000670
- 3 TCN variants on URL char embeddings + handcrafted features. 99.81% accuracy.
- Most similar architecture to ours in 2025. Uses TCN not CharCNN; adds HTML features.

### 7. D-PhishNet (2025)
- **Title**: Dual-branch network for URL/HTML features with GNN and Transformer
- **Link**: https://www.sciencedirect.com/science/article/abs/pii/S1389128625006152
- 99.53% accuracy. Uses HTML features in addition to URL. GNN-based, not CNN.

### 8. URL2Graph++ (2025)
- **Title**: Unified Semantic-Structural-Character Learning
- **arXiv**: https://arxiv.org/abs/2509.10287
- BERT + dual-grained graph (subword + char) + GCN. Completely different approach; no handcrafted features.

---

## Adversarial Robustness — Defense Methods

### 9. SpacePhish (2022)
- **Title**: SpacePhish: The Evasion-space of Phishing Attacks against ML Detectors
- **arXiv**: https://arxiv.org/abs/2210.13660
- Systematic study of the full evasion space for phishing attacks on ML detectors.
- Good framing paper for why adversarial robustness matters.

### 10. urlBERT + Virtual Adversarial Training (2024)
- **Title**: Continuous Multi-Task Pre-training for URL Representation Learning
- **arXiv**: https://arxiv.org/abs/2402.11495
- Combines semantic pre-training with Virtual Adversarial Training (VAT).
- VAT adds noise to learned representations and minimizes KL divergence — improves robustness
  WITHOUT needing explicit adversarial examples. Domain-agnostic.

### 11. Certified Adversarial Robustness via (De)Randomized Smoothing — Malware (2024)
- **Title**: Certified Adversarial Robustness for Malware Detection via (De)Randomized Smoothing
- **arXiv**: https://arxiv.org/abs/2405.00392
- Applies randomized smoothing to feature-based malware detector — directly analogous to
  defending our FeatureMLP. No published equivalent for phishing URL features yet (gap we can claim).

### 12. PWDGAN (2021)
- **Title**: PWDGAN: Phishing Website Detection using Generative Adversarial Network
- **IEEE**: https://ieeexplore.ieee.org/document/9690540/
- Follow-up to AlEroud's approach. Also achieves ~100% evasion on ML classifiers.
- Confirms the vulnerability is general, not specific to AlEroud's exact setup.

---

## Character-Level Adversarial Robustness

### 13. Every Character Counts (2025)
- **Title**: Every Character Counts: Adversarial Robustness of Character-Level Phishing Detection
- **arXiv**: https://arxiv.org/abs/2509.20589
- Compares CharCNN vs CharGRU vs CharBiLSTM under character-level adversarial attacks.
- **Key finding**: CharGRU/CharBiLSTM (98.5% F1 with adv training) outperform CharCNN (95.2%)
  under char-level attacks because RNNs capture long-range dependencies. CharCNN's local
  convolutions miss these. **Relevant to our architecture choice — potential future improvement.**

### 14. URLTran (2021)
- **Title**: URLTran: Improving Phishing URL Detection Using Transformers
- **arXiv**: https://arxiv.org/abs/2106.05256
- BERT/RoBERTa transformer on URL characters. 86.8% TPR at 0.01% FPR.
- Transformer only, no feature engineering branch.

---

## GAN Defense Surveys

### 15. Adversarial Defense Systematic Review: GANs for Threat Detection (2025)
- **arXiv**: https://arxiv.org/abs/2509.20411
- Reviews 85 papers (2021–2025) on GAN-based defenses in threat detection.
- Confirms: **detection of GAN-generated adversarial examples is underexplored** — a gap our
  Exp 8+9 story can claim novelty in for the phishing domain.

### 16. From ML to LLM: Robustness Evaluation (2024)
- **arXiv**: https://arxiv.org/abs/2407.20361
- Broad robustness evaluation across ML and LLM-based phishing detectors.
- Useful for framing our adversarial robustness claims in the larger picture.

---

## Dataset References

### 17. phishphresh (Phreshphresh)
- **HuggingFace**: https://huggingface.co/datasets/phreshphresh/phreshphresh
- 666,315 URLs: 367,904 benign + 298,411 phishing across 50 brands + benign + other_phishing.
- Used in all our experiments except Exp 4/5 (which use Sabir's datasets).
- Cached in `phishphresh/multiclass_brand/data/` as .npz chunks.

### 18. Sabir et al. Replication Package (phishing adversarial datasets)
- Leg_Training.csv, Phish_Training.csv — paper's original training data
- DomainAdversary.csv, PathAdversary.csv, TLDAdversary.csv — adversarial attack datasets
- Used in Exp 4, 5, and for adversarial evaluation in Exp 2, 3.
- Stored locally in `Replication_Package/`

### 19. PhishTank
- **Link**: https://phishtank.org/
- Community-verified phishing URLs. Used in AlEroud & Karabatis (2020).
- Not used directly in our experiments (we use phishphresh instead).

### 20. Alexa Top Sites (now Tranco)
- **Tranco (successor)**: https://tranco-list.eu/
- Top legitimate websites used as benign baseline. Used in AlEroud (2020).
- Not used directly in our experiments.

---

## Our Experiments — Quick Reference

| Exp | Script | Output Dir | What it does |
|-----|--------|-----------|--------------|
| 1 | `train_multiclass_brand.py` | `phishphresh/multiclass_brand/` | 52-class CharCNN+MLP |
| 2 | `train_binary.py` | `binary_phishing/` | Binary CharCNN+MLP |
| 3 | `train_binary_adv.py` | `binary_phishing_adv/` | Binary + Sabir URL adversarial training |
| 4 | `train_binary_paper_data.py` | `binary_paper_data/` | Binary on Sabir's dataset |
| 5 | `train_binary_paper_data_adv.py` | `binary_paper_data_adv/` | Binary + Sabir adv on paper dataset |
| 6 | `train_binary_feat_only.py` | `binary_feat_only/` | Ablation: MLP only, binary |
| 7 | `train_multiclass_feat_only.py` | `multiclass_feat_only/` | Ablation: MLP only, 52-class |
| 8 | `train_binary_gan_adv.py` | `binary_phishing_adv_gan/` | GAN attack eval (100% evasion) |
| 9 | `train_binary_gan_hardened.py` | `binary_gan_hardened/` | GAN adversarial training defense (0% evasion) |
