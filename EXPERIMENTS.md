# URLPhishNet — Experiment Tracker

Each row is one experiment. Results are from the best epoch unless noted.
`FPR` = false positive rate (benign flagged as phishing). `FNR` = false negative rate (phishing missed).

---

## Legend

| Symbol | Meaning |
|--------|---------|
| CharCNN + MLP | Dual-stream: parallel k=3,5,7 char convolutions fused with 67-feature MLP |
| MLP only | 67 handcrafted features → residual MLP, **no CharCNN** (ablation) |
| phishphresh | Our own dataset — 666,315 URLs, 52 brand classes |
| paper data | Sabir et al. dataset — Leg_Training.csv + Phish_Training.csv |
| adv-trained | Paper's attack URLs injected into the training set |

---

## Experiment Table

| # | Script | Dataset | Model | Classes | Epochs | Output Dir | Normal Acc | F1-Macro | MCC | FPR | FNR |
|---|--------|---------|-------|---------|--------|-----------|-----------|----------|-----|-----|-----|
| 1 | `train_multiclass_brand.py` | phishphresh (666K) | CharCNN + MLP | 52 | 60 | `phishphresh/multiclass_brand/` | — | — | — | — | — |
| 2 | `run_52class_adversary.py` | adversarial test only | 52-class model from #1 | 52→binary eval | — | `phishphresh/multiclass_brand/logs/` | — | — | — | — | Dom=72.9% Path=63.7% TLD=72.9% |
| 3 | `train_binary.py` | phishphresh (666K) | CharCNN + MLP | **2** | 40 | `binary_phishing/` | **98.68%** | 0.9867 | 0.9734 | 1.11% | 1.57% |
| 4 | `run_binary_adversary` (within #3) | adversarial test | binary model from #3 | 2 | — | `binary_phishing/logs/` | — | — | — | — | Dom=51.8% Path=76.2% TLD=57.0% |
| 5 | `train_binary_adv.py` | phishphresh + adv URLs (655K train) | CharCNN + MLP, **adv-trained** | **2** | 15 | `binary_phishing_adv/` | **98.58%** | 0.9856 | 0.9712 | 1.08% | 1.85% |
| 6 | adversarial eval within #5 | adversarial test | adv-trained model from #5 | 2 | — | `binary_phishing_adv/logs/` | — | — | — | — | Dom=**0.00%** Path=**0.08%** TLD=**0.00%** |
| 7 | `train_binary_paper_data.py` | paper data (300K leg + 96K phish) | CharCNN + MLP | **2** | 20 | `binary_paper_data/` | **99.18%** | 0.9890 | 0.9780 | ~0.5% | 1.42% |
| 8 | adversarial eval within #7 | adversarial test | model from #7 | 2 | — | `binary_paper_data/logs/` | — | — | — | — | Dom=72.5% Path=0.0% TLD=49.5% |
| 9 | `train_binary_paper_data_adv.py` | paper data + adv URLs | CharCNN + MLP, **adv-trained** | **2** | 20 | `binary_paper_data_adv/` | **99.25%** | 0.9898 | 0.9800 | ~0.4% | Dom=0% Path=0% TLD=0% |
| 10 | `train_binary_feat_only.py` | phishphresh (666K) | **MLP only** (ablation) | **2** | 30 | `binary_feat_only/` | 96.22% | 0.9618 | 0.9236 | 3.31% | Dom=? Path=98.2% TLD=29.6% |
| 11 | `train_52class_feat_only.py` | phishphresh (666K) | **MLP only** (ablation) | **52** | 30 | `multiclass_feat_only/` | 38.20% (F1) | 0.3280 | 0.3151 | — | FNR=0.63% |

---

## Key Findings So Far

### Normal accuracy ranking (our models vs paper Table 7)
```
#7  paper data  + CharCNN+MLP    → 99.18%   ← BEST, beats all 50 paper models
#3  phishphresh + CharCNN+MLP    → 98.68%
#5  phishphresh + CharCNN+MLP adv→ 98.58%   (small drop from adv training, expected)
--- Paper's best model (Basic Lexical+Ext+XGB) → 98.58% ---
```

### Adversarial robustness (FNR = phishing missed on attack URLs)
```
Before adv training (#4):  Dom=51.8%  Path=76.2%  TLD=57.0%  ← collapses
After adv training  (#6):  Dom=0.00%  Path=0.08%  TLD=0.00%  ← near-perfect
Paper best adv model (Basic Lexical+Ext+LR): ~2.42% avg FNR
```

### Novelty vs. paper's 50 models
Our architecture (CharCNN k=3,5,7 + residual FeatureMLP, dual-stream fusion) does not appear in any of the 50 paper models. The paper's neural models are URLNet (Char+Word CNN), EXPOSE (Bag of CNN), and LSTM — none combine multi-scale character CNN with a handcrafted-feature residual MLP in a fusion head.

---

## Adversarial Dataset Description (Sabir et al. arXiv:2005.08454)

Three attack types, all URLs labeled phishing=1:

| Attack | File | Size | Method |
|--------|------|------|--------|
| Domain adversary | `DomainAdversary.csv` | 12,569 | Character substitution in domain (e.g. `paypa1`, `micosoft`) |
| Path adversary | `PathAdversary.csv` | 1,048,575 | Path segment duplication/mutation |
| TLD adversary | `TLDAdversary.csv` | 9,768 | Swap TLD (e.g. `.com` → `.net`) |

Paper's claim: most ML models achieve 95-98% on clean data, but drop to 55-67% on these attacks.
Our model confirms this (before adv training). After injecting attack URLs into training, our model reaches ~100% on all three.

---

## Directory Map

```
phishphresh/multiclass_brand/   ← Exp #1 (52-class CharCNN+MLP)
binary_phishing/                ← Exp #3 (binary CharCNN+MLP, phishphresh)
binary_phishing_adv/            ← Exp #5 (binary adv-trained, phishphresh)
binary_paper_data/              ← Exp #7 (binary CharCNN+MLP, paper data)
binary_paper_data_adv/          ← Exp #9 (binary adv-trained, paper data) DONE
binary_feat_only/               ← Exp #10 (binary MLP-only ablation)       DONE
multiclass_feat_only/           ← Exp #11 (52-class MLP-only ablation)      DONE
```

Each directory has:
- `models/model_best.pth` — best checkpoint
- `models/scaler.pkl` — fitted StandardScaler
- `logs/training_log.csv` — per-epoch metrics
- `logs/stdout.log` — full console output
- `logs/adversarial_results.txt` — final comparison report
