# Phishing Detection Repository Guide

This repository is a collection of phishing-detection pipelines, research utilities, and experiment drivers rather than a single packaged application. The tracked codebase currently spans four related workstreams:

- a hybrid domain and CSE submission pipeline,
- a DOM-only structural phishing pipeline,
- a visual screenshot similarity and classification pipeline,
- and an older multimodal HTML plus URL baseline.

## Scope of This README

This README intentionally documents only the files that are currently tracked in git and therefore expected to be pushed with the repository. Untracked notebooks, local datasets, generated plots, temporary models, and ignored output folders are intentionally excluded.

## Repository Structure at a Glance

- `main.py`, `train_main.py`, `test_main.py`, `Dockerfile`, and `docker-compose.yml` form the hybrid submission-oriented pipeline.
- `dom_tree_builder.py`, `dom_only_robust_detector.py`, `dom_tree_node_detector.py`, `dom_target_multiclass_detector.py`, `create_subsample_dataset.py`, `mlp_subsample_train.py`, and `two_stage_mlp_train.py` form the DOM-heavy research pipeline.
- `screenshot.py`, `visual_similarity.py`, `visual_similarity_detector.py`, `visual_supervised_detector.py`, `visual_dual_train_test.py`, `compare_pgportal.py`, and `example_rotated.html` form the visual pipeline.
- `content.py` is an older standalone multimodal training script that produces the tracked `phishing_model.pth` and `test_results.csv` artifacts.

## Path Conventions Used Below

All commands assume you are running from the repository root.

Replace placeholder paths such as these with your own local data:

- `/abs/path/to/combined_dataset.csv`
- `/abs/path/to/shortlisting_dir`
- `/abs/path/to/reference.html`
- `/abs/path/to/candidate.html`
- `./dom_unsup_features`
- `./dom_unsup_subsample`
- `./dom_unsup_subsample_combined`
- `./visual_consolidated_screenshots/manifest.csv`
- `/abs/path/to/phishing_screenshots`
- `/abs/path/to/benign_screenshots`

## Environment Setup

### 1. Base Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Additional packages used by some tracked scripts

`requirements.txt` covers the lightweight baseline dependencies, but several tracked research scripts import additional packages that are not listed there. Install them if you plan to use the corresponding files:

```bash
pip install transformers opencv-python pillow playwright lightgbm xgboost psutil reportlab python-whois
```

### 3. Extra runtime prerequisites

- Install Chromium for Playwright-based scripts:

```bash
playwright install chromium
```

- Some DOM and screenshot scripts stream the `phreshphish/phreshphish` dataset from Hugging Face. If the dataset requires authentication in your environment, provide a token with the script's `--hf-token` argument.
- GPU acceleration is optional. PyTorch, LightGBM, and XGBoost scripts can still run on CPU.

## Recommended Entry Points

### 1. Hybrid submission pipeline

Run the full train and test flow:

```bash
DATASET_PATH=/abs/path/to/combined_dataset.csv \
SHORTLIST_DIR=/abs/path/to/shortlisting_dir \
python main.py
```

Run train-only:

```bash
DATASET_PATH=/abs/path/to/combined_dataset.csv python train_main.py
```

Run test-only:

```bash
SHORTLIST_DIR=/abs/path/to/shortlisting_dir python test_main.py
```

Containerized run:

```bash
docker build -t phishing-detection .
docker compose up --build
```

Note: `docker-compose.yml` currently hardcodes `/app/backend/dataset/...` environment paths. Update those values or override them before relying on the compose workflow.

### 2. DOM-only quick comparison

For direct HTML-to-HTML structural comparison:

```bash
python dom_only_robust_detector.py compare \
  --reference-html /abs/path/to/reference.html \
  --candidate-html /abs/path/to/candidate.html \
  --reference-domain https://www.examplebank.com \
  --candidate-domain https://secure-examplebank-login.com
```

### 3. DOM feature-store and classification workflow

Extract DOM feature chunks from the dataset:

```bash
python dom_tree_node_detector.py unsup-extract \
  --split train \
  --feature-dir ./dom_unsup_features \
  --batch-size 512
```

Build top-target labels and train the multiclass brand model:

```bash
python dom_target_multiclass_detector.py build-y11 \
  --split train \
  --chunk-files ./dom_unsup_features/train_domfeat_chunk_*.npz \
  --mapping-path ./dom_unsup_features/top10_target_map.json

python dom_target_multiclass_detector.py mc-train \
  --feature-root ./dom_unsup_features \
  --mapping-path ./dom_unsup_features/top10_target_map.json \
  --model-path ./models/dom_brand.pkl \
  --algorithm lightgbm
```

Create a balanced subsample and train a neural model:

```bash
python create_subsample_dataset.py \
  --source-dir ./dom_unsup_features \
  --output-dir ./dom_unsup_subsample_combined \
  --mapping-path ./dom_unsup_features/top10_target_map.json

python mlp_subsample_train.py \
  --feature-root ./dom_unsup_subsample_combined \
  --output-path ./models/mlp_subsample_model.pt \
  --model-type mlp
```

### 4. Visual workflow

Collect sample screenshots from the dataset:

```bash
python screenshot.py --mode one-per-class --split train --out-dir ./sample_preview
```

Train the supervised visual detector:

```bash
python visual_supervised_detector.py \
  --mode train \
  --phishing-dir /abs/path/to/phishing_screenshots \
  --benign-dir /abs/path/to/benign_screenshots \
  --model-path ./models/visual_sup_detector.pkl
```

Train the dual-task visual benchmark:

```bash
python visual_dual_train_test.py \
  --manifest-path ./visual_consolidated_screenshots/manifest.csv \
  --model-type extra_trees \
  --output-model-path ./models/visual_dual_detector.pkl \
  --report-json ./visual_dual_report.json
```

### 5. Older multimodal baseline

Run the standalone HTML plus URL plus text training script:

```bash
python content.py
```

This script trains from streamed data and writes `phishing_model.pth`, `test_results.csv`, and `content.log`.

## Tracked File Catalog

### Repository, Configuration, and Deployment Files

| File | Significance | What it does | How to use |
| --- | --- | --- | --- |
| `.gitignore` | Repository hygiene control | Ignores generated models, logs, JSON and NPZ outputs, screenshot folders, dataset folders, and other bulky artifacts so experiment outputs stay out of version control. | Not executable. Review it before adding new output folders or committing generated artifacts. |
| `README.md` | Primary navigation document | Explains the tracked repository structure, major workflows, and the correct entrypoint for each pushed file. | Read this first. Update it whenever tracked entrypoints or expected data layouts change. |
| `requirements.txt` | Base dependency manifest | Installs the core libraries used by the lighter baseline and hybrid scripts such as `torch`, `pandas`, `scikit-learn`, `datasets`, `tldextract`, and `crawl4ai`. | `pip install -r requirements.txt` |
| `Dockerfile` | Hybrid pipeline container recipe | Builds a Python 3.10 image, installs `requirements.txt`, copies the repository into `/app`, and defaults to `python train_main.py`. | `docker build -t phishing-detection .` |
| `docker-compose.yml` | Hybrid train and test orchestration | Defines `train` and `test` services around the Dockerfile and mounts the repository into the container. | `docker compose up --build`, or run services individually with `docker compose run --build --rm train` and `docker compose run --rm test` after fixing dataset paths. |

### Documentation and Tracked Artifacts

| File | Significance | What it does | How to use |
| --- | --- | --- | --- |
| `technical_document.md` | Formal hybrid-method write-up | Describes the team background, problem framing, hybrid architecture, features, and deployment concept for the older CSE-oriented pipeline. | Not executable. Use it as the detailed design note for `main.py` and related hybrid files. |
| `docs/PhishPhresh.md` | Multimodal baseline report | Summarizes the older `content.py` model, its architecture, and the tracked performance figures for the `phreshphish` dataset. | Not executable. Use it as the companion report for `content.py`, `phishing_model.pth`, and `test_results.csv`. |
| `BRAND_WISE_METRICS.md` | Results reference | Stores per-brand metrics derived from confusion matrices for visual and DOM brand-classification experiments. | Not executable. Read it when comparing model families or preparing result summaries. |
| `phishing_model.pth` | Tracked model artifact | Saved PyTorch checkpoint produced by the older `content.py` training pipeline. | Not directly runnable. Load it from Python with `torch.load("phishing_model.pth", map_location="cpu")` or regenerate it by running `python content.py`. |
| `test_results.csv` | Tracked evaluation artifact | Contains row-level predictions and confidence scores from the older `content.py` evaluation flow. | Inspect it with a spreadsheet or from Python, for example `python -c "import pandas as pd; print(pd.read_csv('test_results.csv').head())"` |

### Hybrid Submission Pipeline

#### `main.py`

- Significance: Primary hybrid orchestration script and the closest thing in the tracked repository to an end-to-end submission pipeline.
- What it does: Trains a 51-feature PyTorch classifier for `Phishing` vs `Suspected`, applies rule-based Critical Sector Entity mapping, optionally captures screenshots and WHOIS evidence, and writes submission outputs under `PS-02_<ApplicationID>_Submission/` and `submission/`.
- Run: `DATASET_PATH=/abs/path/to/combined_dataset.csv SHORTLIST_DIR=/abs/path/to/shortlisting_dir python main.py`

#### `train_main.py`

- Significance: Minimal train-only entrypoint for the hybrid pipeline.
- What it does: Instantiates `HybridPhishingDetector` from `main.py` and calls `train_model`.
- Run: `DATASET_PATH=/abs/path/to/combined_dataset.csv python train_main.py`

#### `test_main.py`

- Significance: Minimal test-only entrypoint for the hybrid pipeline.
- What it does: Loads `submission/hybrid_model.pkl` if present, scans `.xlsx` or `.csv` shortlist files, applies the hybrid detection flow, and writes submission outputs.
- Run: `SHORTLIST_DIR=/abs/path/to/shortlisting_dir python test_main.py`

#### `content.py`

- Significance: Older standalone multimodal baseline preserved as a full training script.
- What it does: Streams the `phreshphish/phreshphish` dataset, extracts DOM, URL, and hashed text features with multiprocessing, trains a fused PyTorch classifier, and writes `phishing_model.pth`, `test_results.csv`, and `content.log`.
- Run: `python content.py`

### DOM Pipeline

#### `dom_tree_builder.py`

- Significance: Foundational DOM parsing utility reused by multiple tracked scripts.
- What it does: Crawls or parses HTML into an explicit DOM tree, stores node metadata, exposes parent and child traversal helpers, and provides structural comparison utilities.
- Run: `python dom_tree_builder.py` to execute the built-in live-site demo against `https://sbi.bank.in/`; in practice this file is more often imported by other scripts.

#### `dom_only_robust_detector.py`

- Significance: Smallest standalone DOM semantic comparison CLI in the repository.
- What it does: Converts HTML into semantic DOM profiles and supports `compare`, `detect`, and `profile` subcommands for structural phishing analysis.
- Run: `python dom_only_robust_detector.py compare --reference-html /abs/path/to/reference.html --candidate-html /abs/path/to/candidate.html --reference-domain https://www.examplebank.com --candidate-domain https://secure-examplebank-login.com`

#### `dom_tree_node_detector.py`

- Significance: Main DOM-only experimentation CLI for binary phishing detection.
- What it does: Extends the standalone DOM comparator with feature-store extraction, one-class unsupervised training, supervised binary training, evaluation, and single-HTML prediction.
- Run: `python dom_tree_node_detector.py sup-train --feature-root ./dom_unsup_features --model-path ./models/dom_sup.pkl --algorithm random_forest`

#### `dom_target_multiclass_detector.py`

- Significance: Main DOM brand-classification CLI for top-target multiclass experiments.
- What it does: Builds top-target mappings, creates `y11` sidecar labels for feature chunks, trains multiclass models such as Random Forest, Extra Trees, XGBoost, and LightGBM, evaluates them, and predicts a target brand from one HTML file.
- Run: `python dom_target_multiclass_detector.py mc-train --feature-root ./dom_unsup_features --mapping-path ./dom_unsup_features/top10_target_map.json --model-path ./models/dom_brand.pkl --algorithm lightgbm`

#### `dummy.py`

- Significance: Preserved older snapshot of the DOM binary pipeline.
- What it does: Implements an earlier version of the DOM semantic profiler plus unsupervised and supervised training commands similar to `dom_tree_node_detector.py`, but with a narrower CLI and more rigid assumptions.
- Run: `python dummy.py sup-train --train-chunks ./dom_unsup_features/train_domfeat_chunk_00000.npz --model-path ./models/dom_dummy.pkl`

#### `create_subsample_dataset.py`

- Significance: Dataset-construction utility for later DOM neural experiments.
- What it does: Merges existing train and test feature chunks, remaps the known targets, caps the unknown class, enforces class and binary balance rules, and writes a new balanced feature store plus metadata.
- Run: `python create_subsample_dataset.py --source-dir ./dom_unsup_features --output-dir ./dom_unsup_subsample_combined --mapping-path ./dom_unsup_features/top10_target_map.json`

#### `mlp_subsample_train.py`

- Significance: Main neural baseline for training on the balanced DOM subsample.
- What it does: Loads chunked DOM features, optionally collapses to top known classes plus unknown, trains either an `mlp` or `cnn1d` head, and supports weighted sampling, balanced batches, synthetic oversampling, and class-threshold tuning.
- Run: `python mlp_subsample_train.py --feature-root ./dom_unsup_subsample_combined --output-path ./models/mlp_subsample_model.pt --model-type mlp`

#### `two_stage_mlp_train.py`

- Significance: More advanced DOM neural experiment than the single-stage MLP baseline.
- What it does: Trains a two-stage pipeline where stage 1 separates known vs unknown and stage 2 resolves the brand among known classes, with targeted boundary-focused synthetic oversampling and threshold search.
- Run: `python two_stage_mlp_train.py --feature-root ./dom_unsup_subsample --subsample-test-feature-root ./dom_unsup_subsample --test-feature-root ./dom_unsup_features --output-path ./models/two_stage_mlp_model.pt`

### Visual Pipeline

#### `screenshot.py`

- Significance: Screenshot acquisition utility for the visual pipeline.
- What it does: Either pulls one benign and one phishing HTML sample from the Hugging Face dataset and renders screenshots, or scans live URLs, keeps only reachable pages, and writes screenshot logs as CSV and JSONL.
- Run: `python screenshot.py --mode one-per-class --split train --out-dir ./sample_preview`

#### `visual_similarity.py`

- Significance: Main pairwise visual comparison demo and reusable implementation.
- What it does: Combines DINOv2 semantic similarity, region similarity, pixel-based comparison, SSIM, and optional rotation correction to compare screenshots or URLs.
- Run: `python visual_similarity.py`

#### `visual_similarity_detector.py`

- Significance: Tracked duplicate of `visual_similarity.py` kept under an alternate filename.
- What it does: Provides the same screenshot capture, DINOv2 comparison, and rotation-invariant visual detection logic as `visual_similarity.py`.
- Run: `python visual_similarity_detector.py`

#### `visual_supervised_detector.py`

- Significance: Supervised screenshot-based phishing classifier.
- What it does: Recursively discovers `clean_js_on.jpg` screenshots under phishing and benign roots, extracts DINOv2 embeddings, trains one of several classifiers, and supports benchmark, predict, and compare modes.
- Run: `python visual_supervised_detector.py --mode train --phishing-dir /abs/path/to/phishing_screenshots --benign-dir /abs/path/to/benign_screenshots --model-path ./models/visual_sup_detector.pkl`

#### `visual_dual_train_test.py`

- Significance: Main visual multiclass plus binary benchmarking script.
- What it does: Reads a consolidated screenshot manifest, extracts DINOv2 features, trains either classical models or neural heads, and reports both phishing detection and brand classification quality.
- Run: `python visual_dual_train_test.py --manifest-path ./visual_consolidated_screenshots/manifest.csv --model-type extra_trees --output-model-path ./models/visual_dual_detector.pkl --report-json ./visual_dual_report.json`

#### `compare_pgportal.py`

- Significance: Focused visual demo script for one manual comparison case.
- What it does: Uses `visual_similarity_detector.py` to compare a reference PG portal screenshot against a rendered HTML version and prints DINOv2, pixel, and combined similarity outputs plus attention maps.
- Run: `python compare_pgportal.py`

#### `example_rotated.html`

- Significance: Demo asset for rotation-invariant visual comparison.
- What it does: Provides the rotated local HTML page used by the direct-run visual similarity demo to test whether rotation correction improves similarity scoring.
- Run or use: Open it directly in a browser, or keep it in place when running `python visual_similarity.py` or `python visual_similarity_detector.py`.

## Important Notes and Caveats

- `visual_similarity.py` and `visual_similarity_detector.py` are effectively duplicate implementations. In direct-run mode they use live screenshot capture and also rely on `example_rotated.html`.
- The direct visual demo currently references `example_rotated.html` through the absolute path `/home/vk/Phishing_Detection/example_rotated.html`. If you move the repository, update that path inside the script before running the demo.
- `compare_pgportal.py` expects local inputs named `./screenshots/pgportal.png` and `./example-pgportal.html`, but those inputs are not tracked in the repository. The script will not run successfully on a clean clone until you provide them.
- Most DOM training scripts expect feature stores such as `./dom_unsup_features` or `./dom_unsup_subsample*`, but those data directories are intentionally ignored by git and must be created locally.
- The compose workflow is only for the older hybrid pipeline. The DOM and visual research scripts are meant to be run directly in Python.

## Suggested Reading Order

If you are new to the repository, use this order:

1. Read `README.md` for the high-level map.
2. Use `main.py` if you want the hybrid submission-style flow.
3. Use `dom_only_robust_detector.py` or `dom_tree_node_detector.py` if you want DOM-only work.
4. Use `visual_supervised_detector.py` or `visual_dual_train_test.py` if you want screenshot-based work.
5. Read `technical_document.md`, `docs/PhishPhresh.md`, and `BRAND_WISE_METRICS.md` when you need deeper background or reported results.
