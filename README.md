# URLPhishNet Repository Guide

This repository contains three related but distinct code paths:

1. A hybrid phishing detection and submission pipeline for CSV and shortlist inputs.
2. A URLPhishNet research pipeline covering multi-class, binary, adversarial-training, GAN, and OOD-detection experiments.
3. Auxiliary DOM, HTML, and visual-similarity experiments for phishing analysis.

It is not a single linear application. Different scripts expect different datasets, produce different artifact folders, and depend on outputs from earlier experiments. This README documents every tracked file in the repository, what it is for, how it fits into the project, and how to run or reuse it.

## Repository at a glance

- Hybrid detector path:
  `main.py`, `train_main.py`, `test_main.py`, `Dockerfile`, `docker-compose.yml`
- URLPhishNet research path:
  `train_multiclass_brand.py`, `train_binary.py`, `train_binary_adv.py`, `train_binary_paper_data.py`, `train_binary_paper_data_adv.py`, `train_binary_feat_only.py`, `train_52class_feat_only.py`, `train_binary_gan_adv.py`, `train_binary_gan_hardened.py`, `train_binary_gan_exact_paper.py`, `train_mahalanobis_ood.py`, `train_charcnn_mahalanobis_ood.py`, `train_replication_gan_test.py`, `run_52class_adversary.py`, `generate_predictions.py`, `analyze_feature_importance.py`
- Alternative multimodal experiments:
  `content.py`, `train_phreshphish_url.py`, `dom_tree_builder.py`, `visual_similarity_detector.py`
- Reference and research notes:
  `RESEARCH.md`, `DEFENSE_RESEARCH.md`, `EXPERIMENTS.md`, `docs/PhishPhresh.md`

## Environment setup

Recommended baseline setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additional packages are needed for some optional scripts but are not pinned in `requirements.txt`:

```bash
pip install playwright python-whois pillow reportlab transformers opencv-python psutil
playwright install chromium
```

Notes:

- `main.py` can use Playwright, WHOIS, Pillow, and ReportLab if they are installed. It will still start without some of them, but evidence collection will be reduced.
- `content.py` uses `psutil`.
- `visual_similarity_detector.py` needs `transformers`, `opencv-python`, `Pillow`, and `playwright`.
- `dom_tree_builder.py` depends on `crawl4ai`, which is listed in `requirements.txt`, but it also needs a working browser automation environment.

## External inputs and expected paths

All example commands below assume:

```bash
export REPO_ROOT=/home/vk/url-only-worktree
cd "$REPO_ROOT"
```

The major inputs are:

- Hybrid training CSV:
  `${REPO_ROOT}/backend/dataset/combined_dataset.csv`
  Used by `main.py` and `train_main.py` through `DATASET_PATH`.
- Hybrid shortlist directory:
  `${REPO_ROOT}/backend/dataset/PS-02_Shortlisting_set`
  Used by `main.py` and `test_main.py` through `SHORTLIST_DIR`.
- Hugging Face dataset:
  `phreshphish/phreshphish`
  Used by `train_multiclass_brand.py`, `content.py`, and `train_phreshphish_url.py`.
- Sabir replication package:
  `${REPO_ROOT}/Replication_Package/Replication_Package/Datasets/...`
  Required by `run_52class_adversary.py`, `train_binary.py`, `train_binary_adv.py`, `train_binary_paper_data.py`, `train_binary_paper_data_adv.py`, and `train_replication_gan_test.py`.

Important practical detail:

- Many research scripts have no CLI flags. They use repo-relative constants inside the file. That means the easiest way to run them is to place the expected datasets in the exact folder layout they reference, then execute `python <script>.py`.

## Recommended execution order

### Hybrid submission workflow

Run the full hybrid pipeline:

```bash
DATASET_PATH="${REPO_ROOT}/backend/dataset/combined_dataset.csv" \
SHORTLIST_DIR="${REPO_ROOT}/backend/dataset/PS-02_Shortlisting_set" \
python main.py
```

Train only:

```bash
DATASET_PATH="${REPO_ROOT}/backend/dataset/combined_dataset.csv" \
python train_main.py
```

Test only:

```bash
SHORTLIST_DIR="${REPO_ROOT}/backend/dataset/PS-02_Shortlisting_set" \
python test_main.py
```

### URLPhishNet research workflow

Recommended order if you want to recreate the research outputs:

1. Generate the phishphresh cache and 52-class base model:
   `HF_TOKEN=hf_xxx python train_multiclass_brand.py`
2. Evaluate or branch from the cached chunks:
   `python run_52class_adversary.py`
   `python train_binary.py`
   `python train_binary_adv.py`
   `python train_binary_feat_only.py`
   `python train_52class_feat_only.py`
   `python generate_predictions.py`
3. Run the feature-space GAN attack:
   `python train_binary_gan_adv.py`
4. Run OOD and hardening experiments that depend on the GAN output:
   `python train_mahalanobis_ood.py`
   `python train_binary_gan_hardened.py`
   `python train_charcnn_mahalanobis_ood.py`
5. Run follow-on analysis:
   `python train_binary_gan_exact_paper.py`
   `python analyze_feature_importance.py`
   `python train_replication_gan_test.py`
6. If you also have the Sabir replication package training data, run the paper-data replications:
   `python train_binary_paper_data.py`
   `python train_binary_paper_data_adv.py`

### Alternative multimodal workflow

Run the DOM/HTML/URL model:

```bash
python content.py
```

Run the phreshphish URL-only trainer that reuses `main.py` feature extraction:

```bash
python train_phreshphish_url.py
```

## File-by-file catalog

In the catalog below, some entries group multiple file paths together when they belong to the same output set and share the same significance and usage pattern. Every tracked file is still named explicitly.

### Root-level source, configuration, and documentation

- `.gitignore`: Git ignore rules for Python bytecode, virtual environments, large cached `.npz` files, generated adversarial CSVs, compressed CSVs, `.env`, IDE files, and auto-generated stdout logs. Not runnable.
- `README.md`: The main repository guide. Not runnable.
- `requirements.txt`: Core Python dependency list for the baseline ML and DOM pipelines. Install with `pip install -r requirements.txt`.
- `Dockerfile`: Minimal Python 3.10 container recipe for the hybrid detector path. Build with `docker build -t hybrid-detector .`.
- `docker-compose.yml`: Compose definition for the hybrid train and test services. Run with `docker compose up --build` or `docker compose run --rm train`.
- `RESEARCH.md`: Literature map and experiment context for URLPhishNet, Sabir et al., GAN attacks, and robustness comparisons. Not runnable.
- `DEFENSE_RESEARCH.md`: Focused note on GAN-adversarial-vector detection strategies, especially quantization-artifact and OOD defenses. Not runnable.
- `EXPERIMENTS.md`: Narrative summary of the architecture, novelty claim, and main experiment results. Not runnable.
- `docs/PhishPhresh.md`: Short note describing the multimodal `content.py` pipeline and its earlier results. Not runnable.

### Root-level runnable scripts

- `main.py`: Primary hybrid submission pipeline. It trains a 51-feature binary classifier on a local CSV, maps domains to CSE targets, optionally captures screenshots and WHOIS records, and writes final submission files. Run:
  `DATASET_PATH="${REPO_ROOT}/backend/dataset/combined_dataset.csv" SHORTLIST_DIR="${REPO_ROOT}/backend/dataset/PS-02_Shortlisting_set" python main.py`
- `train_main.py`: Train-only wrapper around `main.py`. It skips shortlist processing and only trains and saves the hybrid detector. Run:
  `DATASET_PATH="${REPO_ROOT}/backend/dataset/combined_dataset.csv" python train_main.py`
- `test_main.py`: Test-only wrapper around `main.py`. It loads `submission/hybrid_model.pkl` if available, then runs shortlist detection and submission generation. Run:
  `SHORTLIST_DIR="${REPO_ROOT}/backend/dataset/PS-02_Shortlisting_set" python test_main.py`
- `content.py`: Alternative multimodal phishing classifier using DOM features, URL features, and hashing-based text features. It streams `phreshphish/phreshphish`, writes checkpoint `.npz` files, trains a model, saves `phishing_model.pth`, and writes `test_results.csv`. Run:
  `python content.py`
  Note: the script currently contains an embedded Hugging Face token string instead of reading `HF_TOKEN` from the environment.
- `train_phreshphish_url.py`: URL-only training pipeline that reuses `main.py`'s `FeatureExtractor`, `PhishingDataset`, and `PhishingNet`, but trains on `phreshphish/phreshphish`. It caches 51-feature chunks under `phishphresh/urls/`. Run:
  `python train_phreshphish_url.py`
  Note: `HF_TOKEN` is an empty constant in the file, so adjust the script or dataset access method if your environment needs authentication.
- `dom_tree_builder.py`: Crawl4AI and BeautifulSoup utility for building a DOM tree, extracting HTML-structure features, saving tree JSON, and comparing tree structures. Its built-in example crawls `https://sbi.bank.in/` and writes `example_dom_tree.json`. Run:
  `python dom_tree_builder.py`
- `visual_similarity_detector.py`: Screenshot-based phishing similarity detector combining DINOv2 semantic similarity with pixel-based comparison. The built-in example captures SBI and ICICI screenshots and compares them. Run:
  `python visual_similarity_detector.py`
- `train_multiclass_brand.py`: Core 52-class URLPhishNet trainer. It streams and caches phishphresh chunks, builds the dual-stream CharCNN + FeatureMLP model, and writes the primary `phishphresh/multiclass_brand/` artifacts used by many later scripts. Run:
  `HF_TOKEN=hf_xxx python train_multiclass_brand.py`
- `run_52class_adversary.py`: Loads the trained 52-class model and evaluates it against the Sabir `DomainAdversary.csv`, `PathAdversary.csv`, and `TLDAdversary.csv` files. Run:
  `python run_52class_adversary.py`
  Requires the trained `phishphresh/multiclass_brand/models/` artifacts and the replication package at `${REPO_ROOT}/Replication_Package/Replication_Package/Datasets/Adversary_Dataset/`.
- `train_binary.py`: Binary URLPhishNet baseline trained on the cached phishphresh chunks, then evaluated on the Sabir adversarial URL files. Run:
  `python train_binary.py`
  Requires the phishphresh chunk cache from `train_multiclass_brand.py` and the Sabir replication package.
- `train_binary_adv.py`: Adversarially trained binary URLPhishNet using phishphresh plus Sabir adversarial URLs injected into training. Run:
  `python train_binary_adv.py`
  Requires the same inputs as `train_binary.py`.
- `train_binary_paper_data.py`: Binary URLPhishNet trained on Sabir's own `Leg_Training.csv` and `Phish_Training.csv`, then compared directly against the paper's reported table. Run:
  `python train_binary_paper_data.py`
  Requires `${REPO_ROOT}/Replication_Package/Replication_Package/Datasets/Training_Dataset/...`.
- `train_binary_paper_data_adv.py`: Adversarial-training version of the paper-data experiment using Sabir's normal and adversarial datasets. Run:
  `python train_binary_paper_data_adv.py`
  Requires the same replication-package layout as `train_binary_paper_data.py`.
- `train_binary_feat_only.py`: Binary ablation using only the 67 structured URL features and no CharCNN branch. Run:
  `python train_binary_feat_only.py`
  Requires the phishphresh chunk cache and Sabir adversarial CSVs for evaluation.
- `train_52class_feat_only.py`: 52-class ablation using only the 67 structured URL features and the class map learned by `train_multiclass_brand.py`. Run:
  `python train_52class_feat_only.py`
  Requires the phishphresh chunk cache and `phishphresh/multiclass_brand/models/class_map.json`.
- `train_binary_gan_adv.py`: Feature-space GAN attack experiment based on AlEroud and Karabatis. It trains a standalone feature MLP phishing detector, trains the GAN, evaluates evasion, and writes a reusable adversarial dataset. Run:
  `python train_binary_gan_adv.py`
  Requires the phishphresh chunk cache.
- `train_binary_gan_hardened.py`: Hardens the feature-only detector by retraining it on real data plus the adversarial vectors produced by `train_binary_gan_adv.py`. Run:
  `python train_binary_gan_hardened.py`
  Requires `binary_phishing_adv_gan/logs/adversarial_dataset.csv` to exist.
- `train_binary_gan_exact_paper.py`: Repeats the GAN attack using the exact paper loss functions and then checks those adversarial vectors against the Mahalanobis OOD model from Exp 10. Run:
  `python train_binary_gan_exact_paper.py`
  Requires the phishphresh chunk cache and `binary_mahalanobis_ood/models/` artifacts from `train_mahalanobis_ood.py`.
- `train_mahalanobis_ood.py`: FeatureMLP-only Mahalanobis OOD defense. It trains a feature-only detector, fits class-conditional Gaussians on the penultimate layer, and evaluates on GAN vectors from Exp 8. Run:
  `python train_mahalanobis_ood.py`
  Requires the phishphresh chunk cache and the GAN adversarial dataset from `train_binary_gan_adv.py`.
- `train_charcnn_mahalanobis_ood.py`: Dual-stream Mahalanobis OOD defense for the full binary CharCNN + FeatureMLP model. Run:
  `python train_charcnn_mahalanobis_ood.py`
  Requires the phishphresh chunk cache and the GAN adversarial dataset from `train_binary_gan_adv.py`.
- `train_replication_gan_test.py`: Cross-dataset validation of the Mahalanobis OOD defense using Sabir replication-package URLs and a replication-package-trained GAN. Run:
  `python train_replication_gan_test.py`
  Requires the replication package and the Exp 10 `binary_mahalanobis_ood/models/` artifacts.
- `generate_predictions.py`: Reconstructs the 52-class test split and saves per-row predictions, including URL, true brand, predicted brand, and phishing correctness. Run:
  `python generate_predictions.py`
  Requires the `phishphresh/multiclass_brand/models/` artifacts and chunk cache.
- `analyze_feature_importance.py`: Permutation-importance analysis for the Exp 13 dual-stream model. It ranks the 67 structured features by macro-F1 drop. Run:
  `python analyze_feature_importance.py`
  Requires `binary_charcnn_mahalanobis_ood/models/model_best.pth` and `binary_charcnn_mahalanobis_ood/models/scaler.pkl` from `train_charcnn_mahalanobis_ood.py`.

### Root-level tracked generated artifacts

- `phishing_model.pth`: Saved model checkpoint produced by `content.py`. Significance: it is the trained multimodal HTML/URL/text model. Not runnable; regenerate with `python content.py`.
- `test_results.csv`: Per-sample test output produced by `content.py`. Significance: it records true labels, predicted labels, confidence, and correctness for that pipeline. Not runnable; regenerate with `python content.py`.
- `checkpoint_train_50000.npz`, `checkpoint_train_53000.npz`, `checkpoint_train_87861.npz`, `checkpoint_train_137861.npz`, `checkpoint_train_187861.npz`, `checkpoint_train_237861.npz`, `checkpoint_train_287861.npz`, `checkpoint_train_337861.npz`, `checkpoint_train_387861.npz`, `checkpoint_train_437861.npz`, `checkpoint_train_487861.npz`, `checkpoint_train_498255.npz`: Resumable train-split checkpoints written by `content.py`. Each file stores extracted HTML, URL, TF-IDF, and label arrays for part of the streamed dataset. Not runnable; delete them only if you want `content.py` to rebuild the train cache from scratch.
- `checkpoint_test_50000.npz`, `checkpoint_test_100000.npz`, `checkpoint_test_113853.npz`, `checkpoint_test_163853.npz`, `checkpoint_test_168060.npz`: Resumable test-split checkpoints written by `content.py`. Same usage pattern as the train checkpoints. Not runnable.

### `phishphresh/multiclass_brand/` artifacts

- `phishphresh/multiclass_brand/models/model_best.pth`, `phishphresh/multiclass_brand/models/scaler.pkl`: Best 52-class URLPhishNet checkpoint and its fitted scaler from `train_multiclass_brand.py`. These are the primary research artifacts used by `run_52class_adversary.py` and `generate_predictions.py`. Not runnable; regenerate with `python train_multiclass_brand.py`.
- `phishphresh/multiclass_brand/models/class_map.json`, `phishphresh/multiclass_brand/models/class_map.csv`: Machine-readable and tabular mappings from brand strings to class IDs for the 52-class model. Used by `train_52class_feat_only.py` and analysis scripts. Not runnable; regenerate with `python train_multiclass_brand.py`.
- `phishphresh/multiclass_brand/logs/final_report.txt`: Final summary report from the 52-class training run. Not runnable; inspect directly or regenerate with `python train_multiclass_brand.py`.
- `phishphresh/multiclass_brand/logs/adversarial_results.txt`: Summary adversarial evaluation report for the 52-class model. Produced by `run_52class_adversary.py`. Not runnable; regenerate with `python run_52class_adversary.py`.
- `phishphresh/multiclass_brand/logs/adversarial_52class_Domain_Adversary.csv`, `phishphresh/multiclass_brand/logs/adversarial_52class_Path_Adversary.csv`, `phishphresh/multiclass_brand/logs/adversarial_52class_TLD_Adversary.csv`: Per-URL predictions from 52-class adversarial evaluation against Sabir's three attack sets. Not runnable; regenerate with `python run_52class_adversary.py`.

### `multiclass_feat_only/` artifacts

- `multiclass_feat_only/models/model_best.pth`, `multiclass_feat_only/models/scaler.pkl`: Best checkpoint and scaler for the 52-class feature-only ablation from `train_52class_feat_only.py`. Not runnable; regenerate with `python train_52class_feat_only.py`.
- `multiclass_feat_only/logs/training_log.csv`: Per-epoch training metrics for the 52-class feature-only ablation. Not runnable; regenerate with `python train_52class_feat_only.py`.
- `multiclass_feat_only/logs/results.txt`: Final evaluation summary for the 52-class feature-only ablation. Not runnable; regenerate with `python train_52class_feat_only.py`.

### `binary_phishing/` artifacts

- `binary_phishing/models/model_best.pth`, `binary_phishing/models/scaler.pkl`: Best checkpoint and scaler for the binary URLPhishNet baseline from `train_binary.py`. Not runnable; regenerate with `python train_binary.py`.
- `binary_phishing/logs/run.log`, `binary_phishing/logs/stdout.log`, `binary_phishing/logs/training_log.csv`: Runtime log capture and per-epoch metrics for the binary baseline. Not runnable; regenerate with `python train_binary.py`.
- `binary_phishing/logs/adversarial_results.txt`: Aggregate adversarial evaluation summary for the binary baseline. Not runnable; regenerate with `python train_binary.py`.
- `binary_phishing/logs/adversarial_binary_Domain_Adversary.csv`, `binary_phishing/logs/adversarial_binary_Path_Adversary.csv`, `binary_phishing/logs/adversarial_binary_TLD_Adversary.csv`: Per-URL adversarial predictions for the binary baseline. Not runnable; regenerate with `python train_binary.py`.

### `binary_phishing_adv/` artifacts

- `binary_phishing_adv/models/model_best.pth`, `binary_phishing_adv/models/scaler.pkl`: Best checkpoint and scaler for the adversarially trained binary URLPhishNet from `train_binary_adv.py`. Not runnable; regenerate with `python train_binary_adv.py`.
- `binary_phishing_adv/logs/training_log.csv`: Per-epoch training metrics for the adversarially trained binary model. Not runnable; regenerate with `python train_binary_adv.py`.
- `binary_phishing_adv/logs/adversarial_results.txt`: Aggregate adversarial evaluation summary for the adversarially trained binary model. Not runnable; regenerate with `python train_binary_adv.py`.
- `binary_phishing_adv/logs/adversarial_binary_Domain_Adversary.csv`, `binary_phishing_adv/logs/adversarial_binary_Path_Adversary.csv`, `binary_phishing_adv/logs/adversarial_binary_TLD_Adversary.csv`: Per-URL adversarial predictions for the adversarially trained binary model. Not runnable; regenerate with `python train_binary_adv.py`.

### `binary_paper_data/` artifacts

- `binary_paper_data/models/model_best.pth`, `binary_paper_data/models/scaler.pkl`: Best checkpoint and scaler for the binary model trained on Sabir's original training data in `train_binary_paper_data.py`. Not runnable; regenerate with `python train_binary_paper_data.py`.
- `binary_paper_data/logs/run.log`, `binary_paper_data/logs/training_log.csv`: Runtime capture and per-epoch metrics for the paper-data binary model. Not runnable; regenerate with `python train_binary_paper_data.py`.
- `binary_paper_data/logs/comparison_vs_table7.txt`: Direct comparison report between this model and Sabir Table 7. Not runnable; regenerate with `python train_binary_paper_data.py`.
- `binary_paper_data/logs/adversarial_binary_Domain_Adversary.csv`, `binary_paper_data/logs/adversarial_binary_Path_Adversary.csv`, `binary_paper_data/logs/adversarial_binary_TLD_Adversary.csv`: Per-URL adversarial predictions on Sabir's three attack sets. Not runnable; regenerate with `python train_binary_paper_data.py`.

### `binary_paper_data_adv/` artifacts

- `binary_paper_data_adv/models/model_best.pth`, `binary_paper_data_adv/models/scaler.pkl`: Best checkpoint and scaler for the paper-data adversarial-training run from `train_binary_paper_data_adv.py`. Not runnable; regenerate with `python train_binary_paper_data_adv.py`.
- `binary_paper_data_adv/logs/run.log`, `binary_paper_data_adv/logs/training_log.csv`: Runtime capture and per-epoch metrics for the paper-data adversarial-training model. Not runnable; regenerate with `python train_binary_paper_data_adv.py`.
- `binary_paper_data_adv/logs/comparison_vs_table7_adv.txt`: Direct comparison report against the paper's adversarial-training rows. Not runnable; regenerate with `python train_binary_paper_data_adv.py`.
- `binary_paper_data_adv/logs/adversarial_binary_Domain_Adversary.csv`, `binary_paper_data_adv/logs/adversarial_binary_Path_Adversary.csv`, `binary_paper_data_adv/logs/adversarial_binary_TLD_Adversary.csv`: Per-URL adversarial predictions for the paper-data adversarially trained model. Not runnable; regenerate with `python train_binary_paper_data_adv.py`.

### `binary_feat_only/` artifacts

- `binary_feat_only/models/model_best.pth`, `binary_feat_only/models/scaler.pkl`: Best checkpoint and scaler for the binary feature-only ablation from `train_binary_feat_only.py`. Not runnable; regenerate with `python train_binary_feat_only.py`.
- `binary_feat_only/logs/training_log.csv`: Per-epoch metrics for the feature-only ablation. Not runnable; regenerate with `python train_binary_feat_only.py`.
- `binary_feat_only/logs/adversarial_results.txt`: Aggregate adversarial evaluation summary for the feature-only binary ablation. Not runnable; regenerate with `python train_binary_feat_only.py`.
- `binary_feat_only/logs/adversarial_binary_Domain_Adversary.csv`, `binary_feat_only/logs/adversarial_binary_Path_Adversary.csv`, `binary_feat_only/logs/adversarial_binary_TLD_Adversary.csv`: Per-URL adversarial predictions for the feature-only binary ablation. Not runnable; regenerate with `python train_binary_feat_only.py`.

### `binary_phishing_adv_gan/` artifacts

- `binary_phishing_adv_gan/models/feature_mlp_pd.pth`, `binary_phishing_adv_gan/models/pd_scaler.pkl`: Standalone feature-only phishing detector and scaler used as the GAN target in `train_binary_gan_adv.py`. Not runnable; regenerate with `python train_binary_gan_adv.py`.
- `binary_phishing_adv_gan/models/gan_generator.pth`: Trained GAN generator from `train_binary_gan_adv.py`. Not runnable; regenerate with `python train_binary_gan_adv.py`.
- `binary_phishing_adv_gan/logs/run.log`: Runtime log for the GAN attack experiment. Not runnable; regenerate with `python train_binary_gan_adv.py`.
- `binary_phishing_adv_gan/logs/gan_training_loss.csv`: GAN training curve for the attack experiment. Not runnable; regenerate with `python train_binary_gan_adv.py`.
- `binary_phishing_adv_gan/logs/pd_evasion_report.txt`: Summary evasion report showing how well the GAN fools the target detector. Not runnable; regenerate with `python train_binary_gan_adv.py`.

### `binary_gan_hardened/` artifacts

- `binary_gan_hardened/models/hardened_mlp.pth`, `binary_gan_hardened/models/scaler.pkl`: Hardened feature-only detector and scaler trained on real plus GAN adversarial vectors by `train_binary_gan_hardened.py`. Not runnable; regenerate with `python train_binary_gan_hardened.py`.
- `binary_gan_hardened/logs/run.log`, `binary_gan_hardened/logs/stdout_exact_paper.log`, `binary_gan_hardened/logs/training_metrics.csv`: Runtime capture and epoch-level metrics for the hardened detector. Not runnable; regenerate with `python train_binary_gan_hardened.py`.

### `binary_gan_exact_paper/` artifacts

- `binary_gan_exact_paper/models/feature_mlp_pd.pth`, `binary_gan_exact_paper/models/pd_scaler.pkl`: Target feature-only detector and scaler trained inside the exact-paper-loss GAN experiment. Not runnable; regenerate with `python train_binary_gan_exact_paper.py`.
- `binary_gan_exact_paper/models/gan_generator.pth`: GAN generator trained with the exact AlEroud loss equations. Not runnable; regenerate with `python train_binary_gan_exact_paper.py`.
- `binary_gan_exact_paper/logs/run.log`, `binary_gan_exact_paper/logs/stdout_exact_paper.log`: Runtime logs for the exact-paper-loss GAN experiment. Not runnable; regenerate with `python train_binary_gan_exact_paper.py`.
- `binary_gan_exact_paper/logs/gan_training_reward.csv`: Paper-style reward trace for the exact-loss GAN training run. Not runnable; regenerate with `python train_binary_gan_exact_paper.py`.
- `binary_gan_exact_paper/logs/pd_evasion_report.txt`: Evasion summary for the exact-paper-loss GAN. Not runnable; regenerate with `python train_binary_gan_exact_paper.py`.

### `binary_mahalanobis_ood/` artifacts

- `binary_mahalanobis_ood/models/feature_mlp.pth`, `binary_mahalanobis_ood/models/scaler.pkl`, `binary_mahalanobis_ood/models/mahalanobis_params.pkl`: Trained feature-only detector, scaler, and fitted Mahalanobis means/precision/threshold from `train_mahalanobis_ood.py`. These are reused by `train_binary_gan_exact_paper.py` and `train_replication_gan_test.py`. Not runnable; regenerate with `python train_mahalanobis_ood.py`.
- `binary_mahalanobis_ood/logs/run.log`, `binary_mahalanobis_ood/logs/stdout_exact_paper.log`, `binary_mahalanobis_ood/logs/training_metrics.csv`: Runtime capture and epoch-level training metrics for the Mahalanobis OOD experiment. Not runnable; regenerate with `python train_mahalanobis_ood.py`.
- `binary_mahalanobis_ood/logs/ood_report.txt`: Summary report for real-data accuracy and GAN OOD detection performance. Not runnable; regenerate with `python train_mahalanobis_ood.py`.

### `binary_replication_gan_test/` artifacts

- `binary_replication_gan_test/models/gan_generator_reppack.pth`: GAN generator trained on the Sabir replication-package phishing features for the cross-dataset validation study in `train_replication_gan_test.py`. Not runnable; regenerate with `python train_replication_gan_test.py`.
- `binary_replication_gan_test/logs/run.log`: Runtime log for the cross-dataset validation experiment. Not runnable; regenerate with `python train_replication_gan_test.py`.
- `binary_replication_gan_test/logs/gan_training_loss.csv`: GAN training curve for the replication-package GAN. Not runnable; regenerate with `python train_replication_gan_test.py`.
- `binary_replication_gan_test/logs/eval_A_real_urls.txt`, `binary_replication_gan_test/logs/eval_B_gan_vectors.txt`, `binary_replication_gan_test/logs/eval_C_sabir_adversarial.txt`: Three separate evaluation reports for real Sabir URLs, replication-package GAN vectors, and Sabir adversarial URLs. Not runnable; regenerate with `python train_replication_gan_test.py`.

### `feature_importance/` artifacts

- `feature_importance/logs/run.log`: Runtime log for the permutation-importance run. Not runnable; regenerate with `python analyze_feature_importance.py`.
- `feature_importance/logs/permutation_importance.csv`: Ranked per-feature importance values for the 67 structured URL features. Not runnable; regenerate with `python analyze_feature_importance.py`.
- `feature_importance/logs/feature_importance_report.txt`: Human-readable summary of the permutation-importance results. Not runnable; regenerate with `python analyze_feature_importance.py`.

## Practical notes before you run anything

- The hybrid pipeline and the research pipeline are separate. Training one does not automatically prepare the other.
- `train_multiclass_brand.py` is the main entry point for generating the phishphresh `.npz` cache that many later experiments consume.
- Several research scripts assume `${REPO_ROOT}/Replication_Package/Replication_Package/Datasets/` exists, but that directory is not committed here.
- `requirements.txt` is enough for the core training code, not for every optional browser, PDF, or vision-based feature in the repo.
- Most model, scaler, and report files in subdirectories are outputs, not hand-authored source files. They should be regenerated by rerunning the script that owns them rather than edited directly.

## Quick command index

```bash
# Hybrid detector
DATASET_PATH="${REPO_ROOT}/backend/dataset/combined_dataset.csv" \
SHORTLIST_DIR="${REPO_ROOT}/backend/dataset/PS-02_Shortlisting_set" \
python main.py

# 52-class URLPhishNet
HF_TOKEN=hf_xxx python train_multiclass_brand.py

# Binary baseline
python train_binary.py

# Binary adversarial training
python train_binary_adv.py

# Feature-only GAN attack
python train_binary_gan_adv.py

# Feature-space Mahalanobis OOD defense
python train_mahalanobis_ood.py

# Dual-stream Mahalanobis OOD defense
python train_charcnn_mahalanobis_ood.py

# Feature importance on Exp 13
python analyze_feature_importance.py

# DOM experiment
python content.py
```
