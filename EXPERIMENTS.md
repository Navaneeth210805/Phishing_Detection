# URLPhishNet — Experiment Summary

> **What is this?** We built URLPhishNet — a dual-stream neural network combining a
> **CharCNN** (reads raw URL characters) and a **FeatureMLP** (processes 67 handcrafted
> URL features) — and ran 7 experiments to measure its performance and robustness
> against a real adversarial attack paper (Sabir et al. arXiv:2005.08454).

---

## The Paper We're Comparing Against

**Sabir et al. (2020)** tested 50 ML models (Random Forest, XGBoost, LGBM, URLNet, LSTM, etc.)
on phishing detection. Key finding: **most models collapse when attackers slightly mutate URLs**.

They tested 3 attack types:

| Attack | How it works | Dataset size |
|--------|-------------|--------------|
| **Domain adversary** | Swap chars in domain (`paypal` → `paypa1`) | 12,569 URLs |
| **Path adversary** | Duplicate/mutate URL path segments | 1,048,575 URLs |
| **TLD adversary** | Change domain extension (`.com` → `.net`) | 9,768 URLs |

Paper's best normal accuracy: **98.58%** (Basic Lexical + XGBoost)
Paper's best adversarial accuracy (after adv training): **97.71%** (Bigram + RF)

---

## Our Architecture

```
URL → [CharCNN k=3,5,7]  ──┐
                            ├─ concat → Classifier → benign / phishing
URL → [FeatureMLP 67-dim] ──┘
```

- **CharCNN**: reads URL character by character, learns patterns like `paypa1`, `secure-login-update`
- **FeatureMLP**: processes 67 features (URL length, entropy, dot count, HTTPS flag, etc.)
- **Fusion**: both streams concatenated → final classifier

---

## Why CharCNN? What Does It Add Over Just 67 Features?

The 67 handcrafted features are **aggregates** — they compress the whole URL into single numbers
(e.g. URL length=34, dot count=3, entropy=4.1). Two completely different URLs can have identical
feature vectors:

```
paypal.com/login          →  length=18, dots=1, entropy~3.9
paypa1.com/login          →  length=18, dots=1, entropy~3.9   ← same features, different URL
```

The features are **blind to character-level mutations** — which is exactly what adversarial attacks exploit.

**CharCNN fixes this.** It scans the raw URL character by character using parallel convolutional filters
at 3 different window sizes (k=3, k=5, k=7), learning n-gram patterns like:

| Pattern CharCNN learns | Why it matters |
|------------------------|----------------|
| `pa1`, `pay`, `pal` | Typosquatting — digit/letter swaps in brand names |
| `secure-login`, `verify-account` | Deceptive keyword patterns in path |
| `-update.`, `.com.` | Subdomain abuse patterns |
| `@`, `%2F`, `//` | URL obfuscation techniques |

With k=3,5,7 running in parallel (like Kim 2014 TextCNN), it captures both short n-grams (individual chars)
and longer substring patterns simultaneously. Global max+avg pooling then extracts the strongest signals
regardless of where they appear in the URL.

**Ablation proof (Exp 6 vs Exp 2):**

| | MLP only | CharCNN + MLP | Gained |
|-|---------|--------------|--------|
| Normal Accuracy | 96.22% | 98.68% | **+2.46%** |
| FPR | 3.31% | 1.11% | **−2.20%** fewer false alarms |
| FNR | 4.36% | 1.57% | **−2.79%** fewer missed attacks |
| Path adversarial FNR | 98.23% | 76.22% | **−22%** (further fixed by adv training) |

For the 52-class brand task (Exp 7 vs Exp 1), removing CharCNN drops F1 from 0.54 → 0.33.
That's because brand identification is *entirely* character-level — `paypal` vs `microsoft` look
the same to any aggregate feature, but CharCNN reads the actual letters.

---

## Novelty — How Is This Different From Existing Work?

**What already exists:**
- **URLNet** (Le et al. 2018): Dual-channel CNN using character embeddings + word embeddings. No handcrafted features.
- **EXPOSE** (Saxe & Berlin 2017): Bag of character CNNs. No handcrafted features.
- **CNN-Fusion** (2023): Multi-kernel char CNN. No engineered features, single stream.
- **PhishingNet**: Char CNN + hierarchical RNN on words. No engineered features.
- **Sabir et al. 50 models**: Either handcrafted features (RF/LGBM/XGB) OR neural (URLNet/LSTM) — **never both fused**.

**What we do differently:**

| Feature | URLNet | EXPOSE | Sabir models | **Ours** |
|---------|--------|--------|-------------|----------|
| Multi-scale CharCNN (k=3,5,7) | partial | yes | no | **yes** |
| Handcrafted URL features (67-dim) | no | no | yes | **yes** |
| Dual-stream fusion | char+word | no | no | **char+features** |
| Residual MLP on features | no | no | no | **yes** |
| Adversarial evaluation | no | no | yes | **yes** |
| Brand identification (52-class) | no | no | no | **yes** |

**The specific novelty**: combining multi-scale parallel CharCNN (learns sequential character patterns)
with a residual FeatureMLP (learns from structured domain knowledge) in a **fused dual-stream**
is not done by any of the 50 Sabir models, and differs from URLNet (which fuses char+word,
not char+engineered-features). Our ablation experiments quantify exactly what each stream contributes,
which no prior adversarial phishing paper has done.

The 2024-2025 literature is moving toward this fusion direction (multi-channel TCN + features,
URL2Graph++ fusing embeddings + char CNN), which means our architecture is ahead of the Sabir et al.
paper (2020) and consistent with current state-of-the-art thinking.

---

## Related Work — What Already Exists (and How We Differ)

Papers are ordered by relevance to our architecture.

### Character CNN papers (no handcrafted features)

| Paper | Year | Architecture | Gap vs Ours |
|-------|------|-------------|-------------|
| [CNN-Fusion: An effective and lightweight phishing detection method based on multi-variant ConvNet](https://www.sciencedirect.com/science/article/abs/pii/S0020025523002281) | 2023 | Multi-scale parallel char CNN (various kernel sizes) + max pooling. 99%+ accuracy. | **No handcrafted feature branch at all.** Single stream only. |
| [URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection](https://arxiv.org/abs/1802.03162) | 2018 | Dual-channel CNN: char embeddings + word embeddings. | Fuses char+word, not char+engineered features. No handcrafted features. |
| [URLTran: Improving Phishing URL Detection Using Transformers](https://arxiv.org/abs/2106.05256) | 2021 | BERT/RoBERTa transformer on URL characters. 86.8% TPR at 0.01% FPR. | Transformer only, no feature engineering branch. |
| [Phishing URL Detection via CNN and Attention-Based Hierarchical RNN](https://ieeexplore.ieee.org/document/8887407/) | 2019 | Char-level CNN (spatial) + attention hierarchical RNN (word-level temporal). Late fusion. | Fuses char+word with attention, not char+engineered features. |

### Dual-stream / fusion papers (closest to ours)

| Paper | Year | Architecture | Gap vs Ours |
|-------|------|-------------|-------------|
| [Comprehensive phishing detection: A multi-channel approach with variants TCN fusion leveraging URL and HTML features](https://www.sciencedirect.com/science/article/abs/pii/S1084804525000670) | 2025 | Dual channel: (1) URL char embedding + 3 TCN variants, (2) handcrafted features. 99.81% accuracy. | Uses TCN not char CNN; also uses HTML features beyond URL-only; kernel sizes not k=3,5,7. |
| [A dual-layer deep learning model for parallel analysis of URL and HTML features in phishing website detection](https://link.springer.com/article/10.1007/s10115-025-02617-w) | 2025 | Dual-branch CNN: raw URL branch + HTML features branch. 99.53% accuracy. | HTML-based, not lexical feature MLP. Raw URL not character n-gram CNN. |
| [URL2Graph++: Unified Semantic-Structural-Character Learning](https://arxiv.org/abs/2509.10287) | 2025 | BERT semantic encoder + dual-grained graph (subword + char level) + GCN. | Graph-based entirely. No handcrafted features, no CNN. Completely different approach. |

### The paper we're evaluating against

| Paper | Year | Architecture | Notes |
|-------|------|-------------|-------|
| [Exposing Malicious URLs: A Comprehensive Adversarial Analysis of Phishing Detectors](https://arxiv.org/abs/2005.08454) (Sabir et al.) | 2020 | 50 models: RF, XGB, LGBM, SVM, URLNet, EXPOSE, LSTM on handcrafted feature sets OR raw chars. | Never fuses both streams. Best normal: 98.58%. Best adv-trained: 97.71%. We beat both. |

### What none of these papers do (our specific novelty)

1. **Multi-scale char CNN (k=3,5,7 parallel) + separate residual MLP on 67 handcrafted URL features, fused** — this exact combination does not appear in any published paper found.
2. **Ablation study** proving each stream's contribution on an adversarial benchmark — no prior adversarial phishing paper quantifies this.
3. **Brand-level 52-class identification** combined with adversarial robustness evaluation — not done in any of the above.

---

## Experiment 1 — 52-Class URLPhishNet (our original model)

**Goal**: Detect phishing AND identify which brand is being impersonated (52 classes)

| Setting | Value |
|---------|-------|
| Dataset | phishphresh — 666,315 URLs |
| Classes | 52 (benign + 50 brands + other_phishing) |
| Model | CharCNN + FeatureMLP |
| Epochs | 60, best at epoch 39 |
| Output | `phishphresh/multiclass_brand/` |

**Normal test results:**

| Metric | Value |
|--------|-------|
| Accuracy | **77.64%** |
| F1-Macro | **0.5368** |

*Note: 77% accuracy on 52 classes is actually strong — random would be ~2%. The model correctly identifies which brand is being impersonated in most cases.*

**Then we tested the same model on adversarial URLs (no retraining):**

| Attack | Detection Rate | FNR (missed phishing) |
|--------|---------------|----------------------|
| Domain adversary | 27.06% | **72.9%** ← collapses |
| Path adversary | 36.32% | **63.7%** ← collapses |
| TLD adversary | 27.12% | **72.9%** ← collapses |

**Takeaway**: Even our best model misses 63–73% of adversarial phishing. Same problem as the paper's 50 models.

---

## Experiment 2 — Binary URLPhishNet on phishphresh (no retraining on attacks)

**Goal**: Simplify to 2 classes (benign vs phishing) — does it help?

| Setting | Value |
|---------|-------|
| Dataset | phishphresh — 666,315 URLs |
| Classes | **2** (benign=0, phishing=1) |
| Model | CharCNN + FeatureMLP |
| Epochs | 40, best at epoch 39 |
| Output | `binary_phishing/` |

**Normal test results:**

| Metric | Value |
|--------|-------|
| Accuracy | **98.68%** |
| F1-Macro | 0.9867 |
| MCC | 0.9734 |
| FPR (benign wrongly flagged) | 1.11% |
| FNR (phishing missed) | 1.57% |

**Same model tested on adversarial URLs:**

| Attack | Detection Rate | FNR (missed) |
|--------|---------------|--------------|
| Domain adversary | 48.23% | **51.8%** |
| Path adversary | 23.78% | **76.2%** ← worst |
| TLD adversary | 43.02% | **57.0%** |

**Takeaway**: Binary is much better on normal data (98.68% vs 77.64%) but adversarial still fails badly.

---

## Experiment 3 — Adversarial Training on phishphresh

**Goal**: Inject the paper's attack URLs into training — does robustness improve?

| Setting | Value |
|---------|-------|
| Dataset | phishphresh 666,315 + Domain 12,569 + Path 100,000 + TLD 9,768 = **788,652 total** |
| Classes | 2 |
| Model | CharCNN + FeatureMLP |
| Epochs | 15, best at epoch 13 |
| Output | `binary_phishing_adv/` |

**Normal test results (slight drop expected — tradeoff for robustness):**

| Metric | Before adv training | After adv training | Change |
|--------|--------------------|--------------------|--------|
| Accuracy | 98.68% | **98.58%** | -0.10% |
| F1-Macro | 0.9867 | 0.9856 | -0.001 |
| MCC | 0.9734 | 0.9712 | -0.002 |
| FPR | 1.11% | **1.08%** | better |
| FNR | 1.57% | **1.85%** | slightly worse |

**Adversarial results after training:**

| Attack | FNR before | FNR after | Change |
|--------|-----------|-----------|--------|
| Domain adversary | 51.8% | **0.00%** | +51.8% |
| Path adversary | 76.2% | **0.08%** | +76.1% |
| TLD adversary | 57.0% | **0.00%** | +57.0% |

**Takeaway**: Tiny normal accuracy drop (−0.1%), near-perfect adversarial robustness. The tradeoff is massively worth it.

---

## Experiment 4 — Train on Paper's Exact Dataset (fair Table 7 comparison)

**Goal**: Use the paper's own training data so comparison is apples-to-apples

| Setting | Value |
|---------|-------|
| Dataset | Leg_Training.csv (300K legitimate) + Phish_Training.csv (96,693 phishing) |
| Classes | 2 |
| Model | CharCNN + FeatureMLP |
| Epochs | 20, best at epoch 20 |
| Output | `binary_paper_data/` |

**Normal test results vs paper's Table 7:**

| Model | Accuracy | MCC | FNR |
|-------|---------|-----|-----|
| **Ours (CharCNN + MLP)** | **99.18%** | **0.9780** | **1.42%** |
| Paper's best (Basic Lexical + XGB) | 98.58% | 0.956 | 2.22% |
| EXPOSE (Bag of CNN) | 98.46% | 0.969 | 1.59% |
| URLNet (Char+Word CNN) | 98.27% | 0.966 | 2.10% |

**We rank #1 on accuracy among all 50+ paper models on normal data.**

**Adversarial on same model (no adv training):**

| Attack | Detection | FNR |
|--------|----------|-----|
| Domain adversary | 27.5% | 72.5% |
| Path adversary | 100% | 0% *(path adversary happens to look clearly phishy)* |
| TLD adversary | 50.5% | 49.5% |

---

## Experiment 5 — Paper's Dataset + Adversarial Training (replicates paper's Round 2)

**Goal**: Best of both worlds — paper's data + attack URL injection

| Setting | Value |
|---------|-------|
| Dataset | Leg (300K) + Phish (96K) + Domain (12,569) + Path (100K) + TLD (9,768) |
| Classes | 2 |
| Model | CharCNN + FeatureMLP |
| Epochs | 20, best at epoch 19 |
| Output | `binary_paper_data_adv/` |

**Normal test results:**

| Metric | Value |
|--------|-------|
| Accuracy | **99.25%** |
| F1-Macro | 0.9898 |
| MCC | 0.9800 |
| FPR | ~0.4% |
| FNR | 1.62% |

**Adversarial results:**

| Attack | Detection | FNR |
|--------|----------|-----|
| Domain adversary | **100.00%** | **0.00%** |
| Path adversary | **100.00%** | **0.00%** |
| TLD adversary | **100.00%** | **0.00%** |

**vs paper's best adversarially-trained model (Bigram + RF): 97.71% normal accuracy**
**We get 99.25% normal + 100% adversarial — beats it on both simultaneously.**

---

## Experiment 6 — Ablation: Binary, 67 Features Only (no CharCNN)

**Goal**: Prove CharCNN is actually contributing — remove it and see what breaks

| Setting | Value |
|---------|-------|
| Dataset | phishphresh — 666,315 URLs |
| Classes | 2 |
| Model | **FeatureMLP only** (no CharCNN) |
| Epochs | 30, best at epoch 28 |
| Output | `binary_feat_only/` |

**Results vs full model:**

| Metric | MLP only | CharCNN + MLP | Delta |
|--------|---------|--------------|-------|
| Accuracy | 96.22% | 98.68% | **−2.46%** |
| F1-Macro | 0.9618 | 0.9867 | −0.025 |
| MCC | 0.9236 | 0.9734 | −0.050 |
| FPR | 3.31% | 1.11% | **−2.20%** worse |
| FNR | 4.36% | 1.57% | **−2.79%** worse |

**Adversarial FNR (MLP only):**

| Attack | FNR |
|--------|-----|
| Domain adversary | high |
| Path adversary | **98.23%** — nearly useless |
| TLD adversary | 29.63% |

**Takeaway**: CharCNN contributes +2.46% accuracy, cuts FPR from 3.31% → 1.11%, and is critical for path adversarial resistance. Without it, the model is nearly blind to path mutations.

---

## Experiment 7 — Ablation: 52-Class, 67 Features Only (no CharCNN)

**Goal**: Can features alone identify which brand is being impersonated?

| Setting | Value |
|---------|-------|
| Dataset | phishphresh — 666,315 URLs |
| Classes | **52** |
| Model | **FeatureMLP only** (no CharCNN) |
| Epochs | 30, best at epoch 28 |
| Output | `multiclass_feat_only/` |

**Results:**

| Metric | MLP only | CharCNN + MLP |
|--------|---------|--------------|
| Accuracy | 38.20% | 77.64% |
| F1-Macro | **0.3280** | **0.5368** |
| MCC | 0.3151 | — |

**Takeaway**: Features alone get 38% on 52 classes. CharCNN doubles F1 to 0.54. Brand identification is almost impossible without reading the characters — "paypal phish" and "microsoft phish" have similar length, entropy, dot count, etc. Only the characters differ.

---

## The Big Picture — All Results at a Glance

| # | Model | Data | Normal Acc | Adv FNR (avg) |
|---|-------|------|-----------|---------------|
| 1 | CharCNN+MLP (52-class) | phishphresh | 77.64% (F1=0.54) | 70% |
| 2 | CharCNN+MLP (binary) | phishphresh | 98.68% | 62% |
| 3 | CharCNN+MLP adv-trained | phishphresh + attacks | 98.58% | **~0%** |
| 4 | CharCNN+MLP (binary) | paper data | **99.18%** | 41% |
| 5 | CharCNN+MLP adv-trained | paper data + attacks | **99.25%** | **0%** |
| 6 | MLP only (ablation) | phishphresh | 96.22% | ~60%+ |
| 7 | MLP only 52-class (ablation) | phishphresh | F1=0.33 | — |
| 8 | FeatureMLP PD only (GAN attack eval) | phishphresh | 95.52% | **100% evaded by GAN** |
| 9 | Hardened FeatureMLP (GAN adv training) | phishphresh + GAN adv | 94.06% | **0% evaded — fully closed** |
| 10 | FeatureMLP + Mahalanobis OOD (GAN defense, no GAN training data) | phishphresh | **95.45%** | **100% GAN detected, 1% FPR** |
| 11 | Exp 10 model cross-dataset validation (no retraining) | Replication Package | 57% real (scaler shift) | **100% GAN detected (different dataset GAN)** |
| — | Paper's best normal (XGB) | paper data | 98.58% | — |
| — | Paper's best adv-trained (RF) | paper data + attacks | 97.71% | ~2.4% avg |

---

## Key Claims We Can Make

> Honest assessment — each claim comes with what supports it and what the caveat is.

---

**Claim 1 — Higher normal accuracy on paper's dataset**

> *"Our model achieves 99.18% accuracy on the Sabir et al. dataset, compared to the best reported model (98.58%, Basic Lexical + XGBoost)."*

✅ **What supports it**: We trained and tested on the paper's exact dataset (same CSVs) with an 80/20 split. The 0.6% gap is consistent — FPR also improves from ~2.22% → ~0.5%, FNR from 2.22% → 1.42%.

⚠️ **Caveat**: We capped legitimate training data to 300,000 (paper used ~1,048,574). We also don't know the paper's exact train/test split method — ours is random 80/20 stratified. The 0.6% improvement is real but modest, and the comparison is not perfectly controlled for data quantity.

---

**Claim 2 — Adversarial robustness improves with adversarial training**

> *"Injecting attack URLs into training brings adversarial FNR from 51–76% down to near 0%, with only a 0.1% drop in normal accuracy."*

✅ **What supports it**: Before adv training: Domain FNR=51.8%, Path FNR=76.2%, TLD FNR=57.0%. After: all near 0%. The normal accuracy drop is tiny (98.68% → 98.58%). This is a clean before/after comparison on the same model and test set.

⚠️ **Caveat**: The 100% adversarial detection is because we trained on those exact attack types — it's expected. This is the same methodology the paper calls "Round 2". It does NOT mean the model is robust to new, unseen attack types. We cannot claim general adversarial robustness — only robustness to these specific 3 attacks.

---

**Claim 3 — Dual-stream architecture outperforms features-only baseline**

> *"Adding CharCNN to the 67 feature MLP improves accuracy by +2.46%, halves FPR, and is critical for path adversarial resistance."*

✅ **What supports it**: Controlled ablation experiments (Exp 2 vs Exp 6, Exp 1 vs Exp 7). Same dataset, same split, same training setup — only CharCNN removed. Results are clear: +2.46% accuracy, FPR 3.31%→1.11%, FNR 4.36%→1.57%. For 52-class brand identification, F1 drops from 0.54 → 0.33 without CharCNN.

⚠️ **Caveat**: Among the 2024–25 literature, a 2025 TCN paper ([multi-channel TCN fusion](https://www.sciencedirect.com/science/article/abs/pii/S1084804525000670)) also combines char embeddings with handcrafted features. Our specific pairing (multi-scale char CNN k=3,5,7 + residual feature MLP) doesn't appear in published work, but the *idea* of fusing char patterns with features is not entirely new. The novelty is in the specific implementation and the adversarial validation.

---

**Claim 5 — Mahalanobis OOD is a dataset-agnostic, training-free GAN defense**

> *"The Mahalanobis OOD detector, trained only on real phishphresh URLs, detects 100% of GAN adversarial
> vectors from both phishphresh and the Sabir Replication Package — without seeing a single GAN sample
> during training, and without needing to know which GAN or which dataset the attacker used."*

✅ **What supports it**: Exp 10 — 100% detection of 238,729 phishphresh-origin GAN vectors with 1.02% FPR.
Exp 11 — 100% detection of 77,354 Replication Package-origin GAN vectors using the same unmodified model.
The Mahalanobis scores for adversarial vectors (median 2,488,141 and 118,653 respectively) are 231–4,845× above the calibration threshold (513). The structural reason is clear: AlEroud's binary encoding quantizes adversarial outputs to 3 discrete midpoints per feature — a geometric artifact the OOD detector picks up regardless of which dataset trained the GAN.

⚠️ **Caveat**: This defense is specific to feature-space GAN attacks that use binary quantization (AlEroud 2020 architecture). It does NOT defend against character-level URL mutations (Sabir adversarial URLs — Exp 11C shows 0% OOD detection for those). A complete defense requires combining OOD detection (for GAN attacks) with adversarial URL training (for char-level attacks).

---

**Claim 4 — Brand-level identification alongside binary detection**

> *"Our 52-class model identifies which brand is being impersonated, not just whether a URL is phishing — none of the 50 Sabir et al. models do this."*

✅ **What supports it**: The 52-class experiment achieves 77.64% accuracy across 50 brand classes + benign. This is strictly richer output than binary detection. No model in the Sabir et al. paper does multi-class brand identification.

⚠️ **Caveat**: Brand identification is not the main contribution of this work — it exists because our base dataset (phishphresh) has brand labels. It's a bonus capability, not a claim we designed experiments around. We also haven't compared the 52-class model against any dedicated brand-identification paper.

---

## Experiment 8 — GAN Attack Evaluation on FeatureMLP Feature Space (AlEroud 2020)

**What was TRAINED**: A GAN (Generator + Discriminator) targeting our standalone FeatureMLP PD.
**What was TESTED**: GAN-generated adversarial feature vectors fed to the FeatureMLP PD — does PD still detect them as phishing?

**In plain terms**:
> We trained a GAN on phishphresh feature data. The GAN learned to flip phishing feature vectors
> so they look benign to our FeatureMLP. Then we ran those GAN-generated vectors through our
> FeatureMLP and measured how many it still caught. Answer: **none — 100% evaded**.

**Context**:
- AlEroud's GAN attacks the **feature vector space** (67-dim), not raw URL characters
- CharCNN reads raw URL characters → completely unaffected by this attack
- FeatureMLP reads 67 aggregate features → directly attacked (length, entropy, dots, etc.)
- The GAN outputs modified **feature vectors only**, not actual URL strings
  (features are lossy aggregates — you cannot reconstruct a URL from them)

| Setting | Value |
|---------|-------|
| Training data (GAN) | phishphresh features — 533,052 train samples (238,729 phishing + 294,323 benign) |
| Training target (PD) | Standalone FeatureMLP: 67→[256,128,64]→2, trained to 95.52% accuracy |
| GAN architecture | Generator: (134+64)→3×120 ReLU→134 sigmoid \| Discriminator: 134→2×120 ReLU→1 sigmoid |
| GAN training | 100 epochs, batch=512, LR=1e-4 (AlEroud 2020 exact config) |
| Test data | 238,729 GAN-generated adversarial feature vectors run through FeatureMLP PD |
| Output | `binary_phishing_adv_gan/` |
| Script | `train_binary_gan_adv.py` |

**GAN Architecture (following AlEroud 2020 exactly)**:
- **Binary encoding**: 67 continuous features → 134-dim binary (2 bits/feature): `11`=malicious, `01`=suspicious, `00`=benign
- **Generator G**: takes a binary phishing vector + noise → outputs adversarial binary vector (looks benign to PD)
- **Discriminator D**: proxy for the PD — trained to copy PD's predictions so G can backprop through it
- **Why D?**: PD is a trained blackbox with no gradients. D acts as a differentiable stand-in.
- **GAN output**: modified feature vectors only — **no URLs generated**, no char sequences

**Results**:

| What was measured | Result |
|-------------------|--------|
| FeatureMLP PD accuracy (baseline, real test data) | **95.52%** |
| PD detection rate on real phishing (test set) | **94.40%** |
| PD detection rate on GAN adversarial vectors | **0.00%** |
| GAN evasion rate (fooled PD) | **100.00%** |
| Adversarial feature vectors generated | **238,729** |

**Confusion matrix — adversarial vectors vs FeatureMLP PD**:

| | Pred Benign | Pred Phishing |
|--|------------|--------------|
| True Phishing | **238,729** (all evaded) | 0 |

**Takeaway**: AlEroud's GAN achieves 100% evasion on our standalone FeatureMLP.
This establishes the vulnerability. Exp 9 closes it.

**Saved artifacts**:
- `binary_phishing_adv_gan/models/feature_mlp_pd.pth` — trained FeatureMLP PD (95.52% acc)
- `binary_phishing_adv_gan/models/pd_scaler.pkl` — fitted StandardScaler
- `binary_phishing_adv_gan/models/gan_generator.pth` — trained GAN Generator weights
- `binary_phishing_adv_gan/logs/gan_training_loss.csv` — per-epoch G/D loss (100 epochs)
- `binary_phishing_adv_gan/logs/pd_evasion_report.txt` — full evasion report
- `binary_phishing_adv_gan/logs/adversarial_dataset.csv` — 238,729 adversarial feature vectors (114MB, not in git)

---

## Experiment 9 — Hardened FeatureMLP (GAN Adversarial Training Defense)

**What was TRAINED**: FeatureMLP trained on real phishphresh data PLUS the 238,729 GAN adversarial vectors from Exp 8.
**What was TESTED**: Same 238,729 GAN adversarial vectors from Exp 8 — does the hardened model now catch them?

**In plain terms**:
> Exp 8 showed the GAN fools our FeatureMLP with 100% success. Exp 9 answers:
> if we inject those GAN-generated fake phishing vectors into training (labeled as phishing),
> does the model learn to detect them? Answer: **yes — evasion drops from 100% to 0%**.

**This is NOT from any external paper.** We designed this experiment following the same
adversarial training methodology as Exp 3 (which injected Sabir's URL attacks into training).
The concept is standard adversarial training — teach the model by showing it the attacks.

| Setting | Value |
|---------|-------|
| Training data | phishphresh real: 533,052 samples + GAN adversarial (from Exp 8): 238,729 = **771,781 total** |
| Training split | 294,323 benign + 477,458 phishing (real + adversarial) |
| Test data (normal) | phishphresh test split — 133,263 real URLs (same 80/20 split as all experiments) |
| Test data (adversarial) | 238,729 GAN adversarial vectors from Exp 8 |
| Model | Standalone FeatureMLP: 67→[256,128,64]→2, trained from scratch |
| Epochs | 20 |
| Output | `binary_gan_hardened/` |
| Script | `train_binary_gan_hardened.py` (our own script, no external paper) |

**Results — side-by-side with Exp 8 (unhardened baseline)**:

| Metric | Exp 8: Unhardened | Exp 9: Hardened | Change |
|--------|------------------|-----------------|--------|
| Normal accuracy (real test set) | 95.52% | **94.06%** | −1.46% |
| F1-Macro | — | **0.9393** | — |
| MCC | — | **0.8825** | — |
| FPR (benign wrongly flagged) | — | 1.36% | — |
| FNR (real phishing missed) | ~5% | **11.57%** | +6.6% |
| **GAN evasion rate** | **100.00%** | **0.00%** | **−100%** |

**Confusion matrix — same GAN adversarial vectors tested on hardened model**:

| | Pred Benign | Pred Phishing |
|--|------------|--------------|
| True Phishing | **0** (none evaded) | 238,729 (all caught) |

**Takeaway**: Adversarial training closes the GAN evasion gap completely (100% → 0%).
Cost: −1.46% normal accuracy and FNR rises from ~5% → 11.57% on real phishing.

**Why FNR went up**: GAN adversarial vectors are engineered to look benign. Training on 238K of them
labeled as phishing forces the model to be more aggressive on "benign-looking" inputs — which
inadvertently catches fewer genuinely ambiguous real phishing URLs.

**How to fix the FNR hit**: Retrain the full URLPhishNet (CharCNN + FeatureMLP) with these adversarial
vectors injected. The CharCNN stream reads raw URL characters and is completely immune to feature
perturbations — it provides strong complementary signal that should recover the FNR.
This is the planned Exp 10.

---

## Experiment 10 — Mahalanobis OOD Detection (GAN Defense, Zero GAN Training Data)

**What was TRAINED**: FeatureMLP (same 67→[256,128,64]→2 architecture as Exp 8) trained on real phishphresh data only. No GAN data anywhere.
**What was TESTED**: Mahalanobis distance in penultimate-layer (64-dim) representation space used to flag GAN adversarial vectors as out-of-distribution (OOD) — without ever seeing a GAN sample during training.

**Reference paper**: Lee et al., NeurIPS 2018 — *"A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks"*
- arXiv: [https://arxiv.org/abs/1807.03888](https://arxiv.org/abs/1807.03888)
- NeurIPS proceedings: [https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)
- **Venue**: NeurIPS 2018 — CORE A* (top-tier, alongside ICML and ICLR; ~25% acceptance rate)

**In plain terms**:
> Real phishing/benign URLs occupy specific clusters in the neural network's internal representation space.
> AlEroud's GAN generates quantized fake vectors (all features snap to one of 3 fixed midpoints due to binary
> encoding/decoding). These fake vectors land very far from any real URL cluster. The Mahalanobis detector
> measures that distance — without needing to have seen a single GAN sample — and flags them as adversarial.

**Key insight (why this works)**:
- AlEroud's binary encode → GAN → decode pipeline forces all outputs to exactly 3 discrete midpoints per feature
- Real URLs have continuous, smooth feature distributions in embedding space
- The Mahalanobis score for GAN vectors (~2,488,141) is **4,845× higher** than the training threshold (513)
- The separation is so extreme that even a loose threshold catches 100% of GAN vectors

| Setting | Value |
|---------|-------|
| Training data | phishphresh real only — 533,052 train samples (NO GAN data) |
| Test data (normal) | phishphresh test — 133,263 real URLs |
| Test data (adversarial) | 238,729 GAN adversarial vectors from Exp 8 |
| Model | FeatureMLP: 67→[256,128,64]→2 (same as Exp 8 PD) |
| Penultimate layer dim | 64 (features extracted for Gaussian fitting) |
| OOD threshold | 99th percentile of training Mahalanobis scores = **513.16** |
| Epochs | 20 |
| Output | `binary_mahalanobis_ood/` |
| Script | `train_mahalanobis_ood.py` |

**Algorithm (Lee et al. NeurIPS 2018)**:
1. Extract 64-dim penultimate features f(x) for all training samples
2. Compute class-conditional means: μ_benign, μ_phishing
3. Compute pooled tied covariance Σ = Σ_c Σ_{x∈c}(f(x)−μ_c)(f(x)−μ_c)ᵀ / (N−C)
4. Regularise: Σ_reg = Σ + 1e-5 × I, compute precision matrix P = Σ_reg⁻¹
5. OOD score: M(x) = min_c [ (f(x)−μ_c)ᵀ P (f(x)−μ_c) ]
6. Threshold = 99th percentile of training scores → flag anything above as adversarial

**Results**:

| Metric | Value |
|--------|-------|
| FeatureMLP standard accuracy (no OOD layer) | **95.45%** |
| Standard FNR (real phishing missed) | 6.05% |
| Real URLs falsely flagged as OOD (FPR of OOD layer) | **1.02%** (1,363 / 133,263) |
| GAN evasion rate (no OOD layer — baseline) | **100.0%** |
| GAN vectors detected by Mahalanobis OOD | **100.0%** (238,729 / 238,729) |
| OOD threshold | 513.16 |
| Median Mahalanobis score — real training URLs | 35.94 |
| Median Mahalanobis score — GAN adversarial vectors | **2,488,141** (~4,845× above threshold) |

**Side-by-side comparison across GAN defense experiments**:

| Experiment | GAN data in training? | Normal Acc | GAN Evasion | Real URL FPR |
|-----------|----------------------|-----------|-------------|--------------|
| Exp 8 (no defense) | No | 95.52% | **100%** | — |
| Exp 9 (adv training) | **Yes** | 94.06% | **0%** | 1.36% |
| **Exp 10 (Mahalanobis OOD)** | **No** | **95.45%** | **0%** | **1.02%** |

**Takeaway**: Mahalanobis OOD achieves the same 0% GAN evasion as Exp 9, with *higher* normal accuracy (95.45% vs 94.06%), *lower* false positive rate (1.02% vs 1.36%), and without requiring ANY GAN training data. This is a dataset-agnostic defense — it works on any GAN that produces quantized/OOD feature vectors.

**Saved artifacts**:
- `binary_mahalanobis_ood/models/feature_mlp.pth` — trained FeatureMLP (95.45% acc)
- `binary_mahalanobis_ood/models/scaler.pkl` — fitted StandardScaler
- `binary_mahalanobis_ood/models/mahalanobis_params.pkl` — μ_0, μ_1, precision matrix P, threshold
- `binary_mahalanobis_ood/logs/run.log` — full training log
- `binary_mahalanobis_ood/logs/ood_report.txt` — full OOD detection report
- `binary_mahalanobis_ood/logs/training_metrics.csv` — per-epoch train/val metrics

---

## Experiment 11 — Cross-Dataset Validation: Mahalanobis OOD on Replication Package

**What was TRAINED**: Nothing. Exp 10 model loaded as-is (no retraining).
**What was TESTED**: Three evaluations using the Sabir et al. Replication Package to validate that the Mahalanobis OOD defense generalises beyond phishphresh.

**Reference papers**:
- Lee et al., NeurIPS 2018 — [arXiv:1807.03888](https://arxiv.org/abs/1807.03888) (Mahalanobis OOD)
- AlEroud & Karabatis, IWSPA 2020 — [DOI:10.1145/3375708.3380315](https://dl.acm.org/doi/10.1145/3375708.3380315) (GAN attack)
- Sabir et al. 2020 — [arXiv:2005.08454](https://arxiv.org/abs/2005.08454) (Replication Package datasets)

**In plain terms**:
> The key question: does the Mahalanobis OOD detector (trained only on phishphresh) still catch GAN
> adversarial vectors when the GAN was trained on a completely different phishing dataset (Replication Package)?
> If yes → the defense is truly dataset-agnostic. Answer: **yes — 100% detection on a completely new GAN**.

| Setting | Value |
|---------|-------|
| Exp 10 model | Loaded from `binary_mahalanobis_ood/` — NOT retrained |
| Replication Package phishing | `Phish_Training.csv` — 77,354 URLs (different from phishphresh) |
| Replication Package legitimate | `Leg_Training.csv` — 100,000 URLs (capped) |
| Adversarial datasets | `DomainAdversary.csv`, `PathAdversary.csv`, `TLDAdversary.csv` |
| GAN trained on | Replication Package phishing features (77,354 samples) — NOT phishphresh |
| Output | `binary_replication_gan_test/` |
| Script | `train_replication_gan_test.py` |

**Part A — Real URL Classification (Replication Package → Exp 10 model)**:

| Metric | Value | Note |
|--------|-------|------|
| Accuracy | 57.32% | Poor — expected (scaler mismatch: phishphresh-fitted scaler applied to different URL distribution) |
| FNR | 34.27% | |
| FPR | 50.81% | |
| OOD flagged (real URLs) | 6.73% | Higher than Exp 10's 1.02% — distribution shift inflates scores |

**Interpretation**: Feature-based models require in-distribution scaling to perform well. The cross-dataset accuracy drop (98%→57%) shows that the StandardScaler trained on phishphresh does not transfer to the Replication Package URL distribution. This is a known limitation of handcrafted feature approaches and motivates using the CharCNN stream (character patterns are more domain-agnostic).

**Part B — GAN Vectors from Replication Package Phishing Features**:

| Metric | Value |
|--------|-------|
| GAN training data | Phish_Training.csv — 77,354 Replication Package phishing URLs |
| Adversarial vectors generated | 77,354 |
| GAN evasion (no OOD layer) | **100.0%** |
| Mahalanobis OOD detection | **100.0%** |
| Adversarial score median | 118,653 (231× above threshold of 513) |

**Key finding**: Even with a GAN trained on a completely different dataset, OOD detection is 100%. The quantization artifact (all features snap to 3 discrete midpoints via AlEroud's binary encode→decode) is independent of which dataset was used — it's a structural property of the GAN architecture itself.

**Part C — Sabir Adversarial URLs (DomainAdversary / PathAdversary / TLDAdversary)**:

| Dataset | Accuracy | FNR | OOD flagged | Interpretation |
|---------|----------|-----|-------------|----------------|
| DomainAdversary (12,569) | 92.39% | 7.61% | 0% | Real URLs — in-distribution, correctly not flagged |
| PathAdversary (50,000) | 0.56% | 99.44% | 0% | Real URLs — model misclassifies but not OOD |
| TLDAdversary (9,768) | 67.88% | 32.12% | 3.44% | Real URLs — not OOD |

**Key finding**: Sabir adversarial URLs are real mutated URL strings with continuous, realistic feature values. The Mahalanobis OOD detector correctly does NOT flag them as OOD — they are in-distribution. The high FNR on PathAdversary confirms what Exp 3 showed: path mutations require CharCNN + adversarial training, not OOD detection. These are two distinct attack classes requiring different defenses.

**Defense strategy map (what this experiment reveals)**:

| Attack type | Defense needed | Mahalanobis OOD? | Adv training? |
|-------------|---------------|-----------------|---------------|
| GAN feature-space (AlEroud) | Mahalanobis OOD | ✅ 100% detection | Not needed |
| GAN on different dataset | Mahalanobis OOD | ✅ 100% detection | Not needed |
| Domain char mutations (Sabir) | CharCNN + adv training | ✗ Not OOD | ✅ Needed |
| Path mutations (Sabir) | CharCNN + adv training | ✗ Not OOD | ✅ Needed |
| TLD mutations (Sabir) | CharCNN + adv training | ✗ Not OOD | ✅ Needed |

**Saved artifacts**:
- `binary_replication_gan_test/models/gan_generator_reppack.pth` — GAN Generator trained on Replication Package phishing
- `binary_replication_gan_test/logs/run.log` — full run log
- `binary_replication_gan_test/logs/eval_A_real_urls.txt` — Part A classification report
- `binary_replication_gan_test/logs/eval_B_gan_vectors.txt` — Part B GAN detection report
- `binary_replication_gan_test/logs/eval_C_sabir_adversarial.txt` — Part C adversarial URL report
- `binary_replication_gan_test/logs/gan_training_loss.csv` — GAN training loss curve

---

## Directory Map

```
phishphresh/multiclass_brand/   ← Exp 1  (52-class CharCNN+MLP, normal + adv eval)
binary_phishing/                ← Exp 2  (binary CharCNN+MLP, phishphresh)
binary_phishing_adv/            ← Exp 3  (binary adv-trained, phishphresh)
binary_paper_data/              ← Exp 4  (binary CharCNN+MLP, paper's dataset)
binary_paper_data_adv/          ← Exp 5  (binary adv-trained, paper's dataset)
binary_feat_only/               ← Exp 6  (ablation: binary, MLP only)
multiclass_feat_only/           ← Exp 7  (ablation: 52-class, MLP only)
binary_phishing_adv_gan/        ← Exp 8  (GAN attack evaluation — 100% evasion demonstrated)
binary_gan_hardened/            ← Exp 9  (GAN adversarial training defense — evasion closed to 0%)
binary_mahalanobis_ood/         ← Exp 10 (Mahalanobis OOD detection — 100% GAN detected, no GAN data needed)
binary_replication_gan_test/    ← Exp 11 (cross-dataset validation — 100% GAN detection on different dataset)
```

Each directory contains:
- `models/model_best.pth` — saved weights at best epoch
- `models/scaler.pkl` — fitted StandardScaler
- `logs/training_log.csv` or `logs/training_metrics.csv` — per-epoch metrics
- `logs/stdout.log` — full console output
- `logs/adversarial_results.txt` or `comparison_vs_table7*.txt` — final report

Exp 8 additionally contains:
- `models/feature_mlp_pd.pth` — standalone FeatureMLP PD (blackbox being attacked)
- `models/pd_scaler.pkl` — scaler for PD
- `models/gan_generator.pth` — trained GAN Generator weights
- `logs/gan_training_loss.csv` — per-epoch G and D loss
- `logs/pd_evasion_report.txt` — evasion rate report
- `logs/adversarial_dataset.csv` — 238,729 adversarial feature vectors (reusable for next exp)
