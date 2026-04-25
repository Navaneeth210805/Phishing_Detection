# Defense Research — Catching GAN Adversarial Vectors Without Retraining

> **The core question**: Can we detect GAN-generated adversarial phishing feature vectors
> out-of-the-box, without ever training on GAN-generated data?
>
> **Short answer**: Yes — multiple proven approaches exist. One of them exploits a
> fundamental statistical weakness in AlEroud's specific GAN that makes its adversarial
> vectors trivially distinguishable from real URL features.

---

## The Key Weakness in AlEroud's GAN (Our Discovery)

AlEroud's GAN works via a binarize → perturb → decode pipeline:

```
67 real features
    → binarize to 134 bits (Q1/Median thresholds)
    → GAN flips bits
    → decode back to 67 continuous values via from_binary()
```

The `from_binary()` decode step (train_binary_gan_adv.py line 235–238) maps each feature to
**exactly one of 3 discrete midpoint values**:

```python
(feat_min + Q1) / 2.0      ← "malicious" region midpoint
(Q1 + Median)  / 2.0       ← "suspicious" region midpoint
(Median + feat_max) / 2.0  ← "benign" region midpoint
```

**Real URL features have continuous distributions** — values spread across the full range with
natural variance. GAN-generated adversarial vectors have every single feature pinned to one of
exactly 3 discrete values. This is an unnatural, artificial fingerprint.

**Implication**: A simple statistical check (does this vector's features cluster at the 3 midpoints?)
could detect all of AlEroud's adversarial vectors with near-100% accuracy — no adversarial training needed.

> Note: This weakness is specific to binarization-based GANs. A more sophisticated GAN operating
> directly in continuous feature space would not have this artifact.

---

## What We Can Do — Ranked by Strength

---

### Option 1 — Quantization Artifact Detector (Custom, Tier 1)

**Idea**: Check whether each feature value in an input vector falls suspiciously close to one of
the 3 known midpoints derived from training data. Real phishing features never cluster at exactly
these values; GAN-generated ones always do.

**How to implement**:
```python
# Fit thresholds from training phishing data (same as GAN training)
q1, med = compute_thresholds(X_phish_train)
midpoints = [
    (feat_min + q1) / 2,     # malicious midpoint
    (q1 + med) / 2,          # suspicious midpoint
    (med + feat_max) / 2,    # benign midpoint
]

# For each input vector, count how many features are within epsilon of a midpoint
def is_gan_generated(x, midpoints, epsilon=0.01):
    hit = 0
    for i in range(67):
        for mp in [midpoints[0][i], midpoints[1][i], midpoints[2][i]]:
            if abs(x[i] - mp) < epsilon:
                hit += 1
                break
    return hit / 67  # ratio — real URLs will have ~0, GAN vectors will have ~1.0
```

**Expected result**: Near-perfect separation between real and GAN-generated features.

**Limitation**: Only catches AlEroud-style binarization GANs. Continuous-space GANs bypass this.

**No papers needed** — this is a novel contribution we can make based on our own analysis.

---

### Option 2 — Mahalanobis Distance OOD Detection (General, Tier 2)

**Idea**: Fit a multivariate Gaussian on the real training feature distribution. Score any input
by its Mahalanobis distance to this Gaussian. GAN-generated vectors, which lie in an artificially
constrained subspace, will have high distance (they are out-of-distribution).

**How it works**:
```
1. Fit mean μ and covariance Σ on real training phishing features
2. For new input x: score = (x - μ)ᵀ Σ⁻¹ (x - μ)   (Mahalanobis distance)
3. If score > threshold → flag as anomalous / adversarial
```

**No adversarial training needed** — only real data is used to fit μ and Σ.

**Evidence**:
- **Lee et al. (NeurIPS 2018)** — "A Simple Unified Framework for Detecting Out-of-Distribution Samples"
  Proves Mahalanobis distance is highly effective for OOD detection on arbitrary feature spaces.
  Source: https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d94cc2-Paper.pdf

- **Miyaguchi et al. (UDL @ NeurIPS 2021)** — "Relative Mahalanobis Distance" (RMD)
  Improves stability on near-OOD samples by subtracting background Gaussian.
  Source: http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-007.pdf

- **Mahalanobis++ (arXiv:2505.18032, 2025)** — L2-normalization before Mahalanobis improves
  high-dimensional performance.
  Source: https://arxiv.org/html/2505.18032v1

**Paper claim we could make**: "Mahalanobis OOD detection detects AlEroud GAN adversarial vectors
with X% accuracy without ever seeing GAN-generated data during training" — no paper has done
this for phishing feature vectors yet.

---

### Option 3 — Isolation Forest (General, Tier 2)

**Idea**: Train an Isolation Forest on real URL features only. Anomalous inputs (GAN-generated
vectors that occupy an unnatural region of feature space) are "easier to isolate" → lower anomaly
score → flagged.

**No adversarial training needed** — unsupervised, trained on real data only.

**Evidence**:
- **Apruzzese et al. (arXiv:2502.16044, 2025)** — "Multi-Scale Isolation Forest for Adversarial Detection"
  Applied Isolation Forest to adversarial detection in feature streams. Robust, low overhead.
  Source: https://arxiv.org/html/2502.16044

- **Tormanen & Pietiläinen (Pattern Recognition 2023)** — Isolation Forest + Autoencoder combined
  for OOD detection. Source: https://www.sciencedirect.com/science/article/abs/pii/S0031320323008245

- **Nature Scientific Reports (2025)** — Isolation Forest vs One-Class SVM on security feature data.
  Both achieved ~95% adversarial detection without adversarial training.
  Source: https://www.nature.com/articles/s41598-025-20445-4

**Advantage**: sklearn implementation, fits in 5 lines of code, no hyperparameter tuning needed.

---

### Option 4 — Energy-Based OOD Detection (No Retraining, Uses Existing Model, Tier 2)

**Idea**: Run the already-trained FeatureMLP (from Exp 8). Instead of using its softmax output,
use the raw logits to compute an "energy score". In-distribution inputs (real phishing/benign) have
low energy; OOD inputs (GAN-generated) have high energy.

```python
def energy_score(logits):
    return -torch.logsumexp(logits, dim=1)  # lower = more in-distribution

# No retraining — applies post-hoc to the already-trained Exp 8 FeatureMLP
```

**No adversarial training needed** — wraps around the existing trained model.

**Evidence**:
- **Liu et al. (NeurIPS 2020)** — "Energy-based Out-of-Distribution Detection"
  18% FPR reduction vs softmax confidence. Model-agnostic.
  Source: https://arxiv.org/pdf/2010.03759
  GitHub: https://github.com/weitliu/energy_ood

- **Choi et al. (CVPR 2023)** — "Balanced Energy Regularization Loss for OOD Detection"
  Source: https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_Balanced_Energy_Regularization_Loss_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf

**Advantage**: Zero setup — just compute `-logsumexp(logits)` on the Exp 8 model. Instant experiment.

---

### Option 5 — Feature Coherence / Correlation Invariant Check (Novel, Tier 2)

**Idea**: Real URL features have natural correlations that must hold (e.g., URL length correlates
with path length, entropy correlates with URL complexity). GAN bit-flipping can break these
natural correlations. Fit a correlation matrix on real training data; flag vectors where correlation
deviation is abnormally high.

**Evidence**:
- **Rahman et al. (ACM TOPS 2024)** — "Level Up with ML Vulnerability Identification:
  Leveraging Domain Constraints in Feature Space for Robust Android Malware Detection"
  Built a secondary model checking feature dependencies. Achieved 89.6% adversarial detection
  without adversarial training. Directly analogous to our feature space.
  Source: https://dl.acm.org/doi/full/10.1145/3711899

- **Sabir et al. (2020)** — Their "feature removal" strategy showed certain features are disproportionately
  perturbed by attacks, making feature-level deviation analysis possible.

**How to implement**:
```python
# Fit correlation on real training features
C_real = np.corrcoef(X_train.T)   # 67×67 correlation matrix

# For a batch of GAN vectors:
C_gan = np.corrcoef(X_gan.T)

# Frobenius norm distance — large value = correlations broken = adversarial
deviation = np.linalg.norm(C_real - C_gan, 'fro')
```

**Paper claim**: "Feature correlation invariants distinguish GAN adversarial from real phishing
vectors without adversarial training" — no published paper has done this for phishing yet.

---

### Option 6 — One-Class SVM / Deep SVDD (General, Tier 2)

**Idea**: Train a One-Class SVM (OCSVM) or Support Vector Data Description (SVDD) on real URL
features. Learn a tight hypersphere around real data. GAN vectors fall outside the sphere → rejected.

**No adversarial training needed** — one-class learning from real data only.

**Evidence**:
- **Ruff et al. (ICML 2018)** — "Deep One-Class Classification" (Deep SVDD)
  Source: https://github.com/lukasruff/Deep-SVDD

- **Nature Scientific Reports (2025)** — One-Class SVM on security data, ~95% detection rate.
  Source: https://www.nature.com/articles/s41598-025-20445-4

- **Sudharsan & Pilosof (OpenReview 2018)** — Unsupervised adversarial detection via One-Class SVM.
  Source: https://openreview.net/forum?id=BJgd7m0xRZ

---

### Option 7 — Randomized Smoothing (Certified, Mathematical Guarantee, Tier 3)

**Idea**: At inference time, add Gaussian noise to input features, classify N noisy copies,
take majority vote. Provides a **provable certificate**: for perturbations within L2 radius R,
the classification is guaranteed correct. No adversarial training needed.

**No adversarial training needed** — wraps any existing classifier at inference time.

**Evidence**:
- **Cohen et al. (ICML 2019)** — "Certified Adversarial Robustness via Randomized Smoothing"
  Original framework. Source: https://arxiv.org/abs/1902.02918

- **arXiv:2405.00392 (2024)** — "Certified Adversarial Robustness of ML-based Malware Detectors
  via (De)Randomized Smoothing" — **Extended to feature-based security detectors** (tabular, not images).
  This is our most direct analog. Source: https://arxiv.org/abs/2405.00392

- **arXiv:2402.15267 (2024)** — Earlier variant on tabular security data.
  Source: https://arxiv.org/html/2402.15267v1

- **NeurIPS 2022** — De-Randomized Smoothing for tree-based classifiers on tabular data.
  Source: https://proceedings.neurips.cc/paper_files/paper/2022/file/146b4bab3f8536a07905f25d367b4924-Paper-Conference.pdf

**Advantage**: Mathematical guarantee, not just empirical. Strong for paper claims.
**Disadvantage**: Slows inference (need N forward passes per sample).

---

### What Does NOT Work Well (Avoid)

**Autoencoder reconstruction error** — recent 2025 finding shows adversarial inputs can occupy
the latent space close to normal data, making reconstruction error unreliable as a standalone detector.
- Source: Bouman & Heskes (ICLR 2025) — https://arxiv.org/html/2501.13864v1
- **Use only as secondary signal**, not primary detection.

---

## Proposed Exp 10 — Two-Layer OOD Defense

```
                  Input feature vector (67-dim)
                           ↓
          ┌────────────────────────────────────────┐
          │     Layer 1: OOD Anomaly Detector      │
          │  (Mahalanobis + Isolation Forest       │
          │   trained on REAL data only)           │
          │                                        │
          │  Anomaly score > threshold?            │
          │  YES → flag as ADVERSARIAL (no GAN     │
          │         training data used!)           │
          │  NO  → pass to classifier              │
          └────────────────────────────────────────┘
                           ↓ (if not flagged)
          ┌────────────────────────────────────────┐
          │     Layer 2: FeatureMLP Classifier     │
          │  (normal phishing detection)           │
          └────────────────────────────────────────┘
```

**What to measure**:
- How many of the 238,729 GAN adversarial vectors (from Exp 8) are caught by Layer 1 alone?
- Normal accuracy on real phishing/benign — does Layer 1 introduce false positives?
- Compare to Exp 9 (adversarial training): same evasion rate? better normal accuracy?

**Expected outcome**: Layer 1 catches most/all GAN adversarial vectors (because they're OOD by
the 3-point quantization artifact) while preserving normal accuracy on real URL features.

**Paper contribution**: First paper to show that AlEroud-style GAN attacks on phishing detectors
can be caught by OOD detection without any adversarial training — exploiting a fundamental
statistical weakness in the binarization-based feature perturbation approach.

---

## Summary Table

| Approach | Paper | Needs GAN data? | Works on Tabular? | Strength |
|----------|-------|----------------|-------------------|----------|
| Quantization artifact check | Novel (us) | No | Yes (custom) | Perfect for AlEroud's GAN |
| Mahalanobis Distance OOD | Lee et al. NeurIPS 2018 | No | Yes | Strong, proven |
| Isolation Forest | Apruzzese et al. 2025 | No | Yes | Strong, easy to implement |
| Energy-Based OOD | Liu et al. NeurIPS 2020 | No | Yes (model-agnostic) | Good, zero setup |
| Feature Coherence Check | Rahman et al. ACM TOPS 2024 | No | Yes | Novel for phishing |
| One-Class SVM / SVDD | Ruff et al. ICML 2018 | No | Yes | Solid baseline |
| Randomized Smoothing | Cohen et al. ICML 2019 + arXiv:2405.00392 | No | Yes (malware proven) | Certified guarantee |
| Autoencoder | Geiger et al. 2023 | No | Unreliable alone | Avoid as primary |

---

## All Source Links

| # | Paper | Link |
|---|-------|------|
| 1 | Lee et al. (NeurIPS 2018) — Mahalanobis OOD | https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d94cc2-Paper.pdf |
| 2 | Miyaguchi et al. (NeurIPS 2021) — Relative Mahalanobis | http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-007.pdf |
| 3 | Mahalanobis++ (arXiv 2025) | https://arxiv.org/html/2505.18032v1 |
| 4 | Liu et al. (NeurIPS 2020) — Energy-based OOD | https://arxiv.org/pdf/2010.03759 |
| 5 | Choi et al. (CVPR 2023) — Balanced Energy OOD | https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_Balanced_Energy_Regularization_Loss_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf |
| 6 | Rahman et al. (ACM TOPS 2024) — Domain Constraints for Adversarial Detection | https://dl.acm.org/doi/full/10.1145/3711899 |
| 7 | Grosse et al. (arXiv 2017) — Statistical Detection of Adversarial Examples | https://ar5iv.labs.arxiv.org/html/1702.06280 |
| 8 | Cohen et al. (ICML 2019) — Certified Robustness via Randomized Smoothing | https://arxiv.org/abs/1902.02918 |
| 9 | arXiv:2405.00392 (2024) — Certified Robustness for Malware Detectors | https://arxiv.org/abs/2405.00392 |
| 10 | arXiv:2402.15267 (2024) — De-Randomized Smoothing (Security) | https://arxiv.org/html/2402.15267v1 |
| 11 | NeurIPS 2022 — De-Randomized Smoothing for Trees on Tabular Data | https://proceedings.neurips.cc/paper_files/paper/2022/file/146b4bab3f8536a07905f25d367b4924-Paper-Conference.pdf |
| 12 | Apruzzese et al. (arXiv 2025) — Isolation Forest for Adversarial Detection | https://arxiv.org/html/2502.16044 |
| 13 | Tormanen & Pietiläinen (Pattern Recognition 2023) — Isolation Forest + Autoencoder | https://www.sciencedirect.com/science/article/abs/pii/S0031320323008245 |
| 14 | Nature Scientific Reports (2025) — OC-SVM vs Isolation Forest on Security Data | https://www.nature.com/articles/s41598-025-20445-4 |
| 15 | Ruff et al. (ICML 2018) — Deep SVDD | https://github.com/lukasruff/Deep-SVDD |
| 16 | Bouman & Heskes (ICLR 2025) — Autoencoders Unreliable for Anomaly Detection | https://arxiv.org/html/2501.13864v1 |
| 17 | Geiger et al. (Computers & Security 2023) — ATDAD Tabular Anomaly Detection | https://www.sciencedirect.com/science/article/abs/pii/S0167404823003590 |
| 18 | Roy et al. (NeurIPS 2021) — Qu-ANTI-zation: Quantization Artifacts are Detectable | https://arxiv.org/abs/2110.13541 |
| 19 | SpacePhish (arXiv 2022) — Evasion-space of phishing attacks | https://arxiv.org/abs/2210.13660 |
| 20 | GAN Defense Systematic Review 2021-2025 (arXiv:2509.20411) | https://arxiv.org/abs/2509.20411 |
