# DOM- and Visual-Based Phishing Detection: Methodology and Results Report

## 1. Scope, Evidence Base, and Reading Guide

This report documents only the **DOM-based** and **visual-based** phishing detection work that is actually implemented and/or reported in this repository. It is written strictly from repository evidence and does **not** infer results that are not present in code or committed artifacts.

### 1.1 Primary evidence reviewed

The report is grounded mainly in these files:

- DOM comparison and DOM feature extraction:
  - `dom_tree_builder.py`
  - `dom_only_robust_detector.py`
  - `dom_tree_node_detector.py`
  - `dom_target_multiclass_detector.py`
- Visual comparison and visual feature extraction:
  - `visual_similarity.py`
  - `visual_similarity_detector.py`
  - `compare_pgportal.py`
  - `visual_supervised_detector.py`
  - `visual_dual_train_test.py`
  - `screenshot.py`
- Dataset preparation / class-mapping / class-imbalance studies:
  - `create_subsample_dataset.py`
  - `create_mapping.py`
  - `train_without_majority.py`
  - `train_no_majority_full.py`
  - `train_no_majority_simple.py`
  - `train_no_majority_stream.py`
  - `train_subsample_no_majority.py`
  - `run_no_majority_training.py`
  - `mlp_subsample_train.py`
  - `two_stage_mlp_train.py`
- Result artifacts:
  - `dom_model_compare_sup_eval.json`
  - `dom_model_compare_iforest_eval.json`
  - `dom_model_compare_ocsvm_eval.json`
  - `dom_model_compare_ocsvm_train.json`
  - `dom_brand_stage1_results.jsonl`
  - `all_model_results.jsonl`
  - `dom_stage1_mapping.json`
  - `dom_unsup_subsample_brand_stage1/metadata.json`
  - `visual_benchmark_report.json`
  - `visual_benchmark_report_full.json`
  - `visual_dual_report.json`
  - `visual_dual_report_all.json`
  - `visual_dual_report_all_brand_gated.json`
  - `visual_dual_cnn1d_e100.json`
  - `visual_dual_mlp_e100.json`
  - `BRAND_WISE_METRICS.md`
  - `results_comparison_summary.txt`
  - `training_visualizations/README.md`
  - `training_visualizations/lightgbm_builtin/README.md`

### 1.2 Important repository observations

- `visual_similarity.py` and `visual_similarity_detector.py` are identical implementations of the visual pairwise-comparison pipeline.
- `dom_only_robust_detector.py` is the standalone DOM-comparison script; the same comparison logic family is also integrated inside `dom_tree_node_detector.py`, where it is used as the basis for binary DOM classifiers.
- `dummy.py` is another DOM detector variant/snapshot, but it is not the main file used for the current reported results.
- `README.md`, `technical_document.md`, `main.py`, `content.py`, `train_main.py`, `test_main.py`, `Dockerfile`, and `docker-compose.yml` belong to an older or separate **hybrid lexical/domain** pipeline. Because the request here is specifically for **DOM-based** and **visual-based** methodology/results, those files are acknowledged but not used as the basis for claims below.
- For the pairwise website-comparison stage, the repository contains **working code and generated screenshots/attention maps**, but it does **not** contain a committed JSON/CSV benchmark with final pairwise scores. That distinction is kept explicit in the results section.

### 1.3 How this report is organized

The report follows the sequence requested:

1. **Prediction by comparing two websites**.
2. **Prediction of benign vs phishing**.
3. **Prediction of target classes / brands**.

For each method, the report states:

- what is used,
- why it is used,
- how it is used,
- the advantage of using it,
- and the results that are actually available in the repository.

## 2. Methodology

## 2.1 Task 1: Predicting by Comparing Two Websites

This task is implemented in **two parallel ways** in the repository:

- a **DOM structural comparison** path, and
- a **visual screenshot comparison** path.

The central idea is the same in both cases: instead of classifying a single page in isolation, the system compares a suspicious page to one or more reference pages and measures how closely it mimics a legitimate target.

## 2.1.1 DOM structural comparison

### What is used

The DOM comparison path is implemented through:

- `DOMTreeBuilder` in `dom_tree_builder.py`
- `DOMOnlyRobustDetector` in `dom_only_robust_detector.py`
- the embedded `DOMOnlyRobustDetector` in `dom_tree_node_detector.py`

### Why it is used

Phishing pages often change superficial HTML details while preserving the **interaction structure** that matters operationally:

- forms,
- password fields,
- outbound actions,
- page depth/layout,
- repeated container patterns,
- and specific login/update/verification text.

A DOM-comparison method is used because it can detect **structural mimicry** even when the attacker rewrites raw markup, renames classes/IDs, or changes cosmetic content.

### How it is used

#### A. Building the DOM tree

`DOMTreeBuilder` creates a tree with explicit node IDs, parent-child links, depth, attributes, and direct text content.

The builder supports two input modes:

- `build_tree(...)`: crawl a live URL, optionally with JavaScript rendering.
- `build_from_html_string(...)`: build directly from an HTML string.

This matters because the rest of the DOM pipeline can work on:

- live websites,
- locally stored HTML,
- or HTML already present in datasets.

#### B. Converting raw DOM into a semantic profile

`DOMOnlyRobustDetector.build_profile(...)` does not compare raw HTML strings directly. It converts each page into a **semantic DOM profile** containing:

- `role_counts`: counts of semantic roles such as `form_external`, `anchor_internal`, `input_password`, `heading`, `container`, etc.
- `path_ngrams`: role-path n-grams through the DOM tree. Default n-gram size is `3`.
- `depth_hist`: a histogram of node depths.
- `behavior_edges`: event/interaction-like patterns derived from DOM behavior.
- `text_keywords`: counts of suspicious keywords.
- `attribute_features`: stable tag-attribute signatures.
- `metrics`: summary scalars such as:
  - `total_nodes`
  - `interactive_ratio`
  - `edge_count`
  - `keyword_density`

The profile builder also applies several normalizations that are important to robustness:

- hidden elements are skipped if they use `hidden`, `aria-hidden`, or CSS hiding patterns such as `display:none`, `visibility:hidden`, `opacity:0`, `height:0`, or `width:0`.
- noisy attributes such as `id`, `class`, `style`, `nonce`, `integrity`, `crossorigin`, and `data-*` / `aria-*` are suppressed from the stable attribute signature.
- generic wrapper containers with at most one child are down-weighted by not letting them explode the path space.

#### C. Semantic role extraction

Tags are mapped into coarser semantic roles.

Examples from the code:

- `input[type=password] -> input_password`
- `input[type=email] -> input_email`
- `form` -> `form_<target_class>` where target class is one of `internal`, `external`, `relative`, `empty`, etc.
- `a` -> `anchor_<target_class>`
- `iframe` -> `iframe_<target_class>`
- `div`, `span`, `section`, `article`, `main`, `header`, `footer`, `nav` -> `container`
- headers -> `heading`
- text-like tags -> `text`

This role abstraction is a major design choice: it reduces sensitivity to surface markup differences while keeping phishing-relevant interaction meaning.

#### D. Behavior extraction

The detector explicitly extracts behavioral cues from the DOM:

- form action target class,
- form method,
- whether a form subtree contains a password field,
- anchor destinations,
- iframe destinations,
- script patterns that imply:
  - redirect behavior,
  - popup behavior,
  - forced submit,
  - network calls,
  - and suspicious decoding/evaluation patterns.

Examples of regex-triggered script behaviors in the code include:

- `window.location`, `location.href`, `location.replace`
- `window.open`, `prompt`, `alert`
- `.submit(`
- `fetch`, `XMLHttpRequest`, `axios`
- `atob`, `fromCharCode`, `eval`

#### E. Text-keyword extraction

Suspicious text keywords are counted directly from visible text. The keyword set in code is:

- `login`
- `sign in`
- `verify`
- `password`
- `otp`
- `secure`
- `update`
- `bank`
- `account`

#### F. Node-to-node similarity

When two pages are compared, node similarity is computed from three components:

- `0.4 * tag_match`
- `0.3 * role_match`
- `0.3 * attribute_similarity`

The attribute similarity itself is split into:

- `60%` key-structure overlap, and
- `40%` exact value agreement on shared keys.

#### G. Tree traversal comparison

The core comparison is not just bag-of-features matching. It does a **depth-wise one-to-one assignment** using a Hungarian maximization procedure.

For each depth:

- each reference node is matched against candidate nodes,
- the detector adds a parent-consistency bonus if parent roles also align,
- and records the best one-to-one alignment.

A matched node counts as a successful match when its similarity is at least `0.7`.

#### H. Context blending

To stabilize traversal matching, the detector also computes Jaccard-style similarities over:

- path n-grams,
- attribute features,
- behavior edges,
- depth histogram,
- text keywords.

These are blended as:

- `0.35 * path_ngram_context`
- `0.35 * attribute_context`
- `0.15 * behavior_context`
- `0.10 * depth_context`
- `0.05 * text_context`

Then the final tree similarity is:

- `0.90 * raw_tree_traversal_similarity`
- `+ 0.10 * global_context`

#### I. Final verdict and risk logic

The DOM pairwise comparator returns:

- `high_semantic_match` if semantic similarity `>= 0.85`
- `moderate_semantic_match` if `>= 0.70`
- `low_semantic_match` otherwise

It also assigns a separate risk label:

- `high_risk_behavior` if there is an external/empty-target password form,
- `possible_brand_mimic` if similarity is very high,
- `needs_manual_review` for moderate matches,
- `low_structural_match` otherwise.

### Advantage of this method

The DOM comparison approach gives four practical advantages:

1. It is more robust than raw HTML string matching.
2. It captures phishing-relevant behavior, not only appearance.
3. It is explainable because scores come from explicit structural components.
4. It can be reused both for direct website-to-website comparison and for feature generation in downstream classifiers.

## 2.1.2 Visual screenshot comparison

### What is used

The visual comparison path is implemented through:

- `VisualPhishingDetector` in `visual_similarity.py` / `visual_similarity_detector.py`
- `ScreenshotCapture`
- `RotationHandler`
- `PixelwiseComparator`
- `DINOv2Comparator`
- the demonstration script `compare_pgportal.py`

### Why it is used

Many phishing sites are primarily **visual impersonations**. A screenshot-based method is therefore useful because it captures what the end user would actually see:

- logos,
- page layout,
- spacing,
- headers,
- forms,
- and other brand-specific visual patterns.

The implementation intentionally combines **deep semantic similarity** and **classical pixel-level similarity** because those two views solve different failure modes.

### How it is used

#### A. Screenshot acquisition

`ScreenshotCapture.capture_screenshot(...)` uses Playwright with these important defaults:

- Chromium browser
- default viewport `1920 x 1080`
- `wait_until='networkidle'`
- additional wait for `domcontentloaded`
- extra `2` seconds before capture

This is meant to allow dynamic pages to settle before similarity is measured.

#### B. Rotation handling

Before pixel comparison, `RotationHandler` can estimate relative rotation using ORB features and affine estimation.

If a rotation larger than `2.0` degrees is detected, the candidate image is rotated back before comparison.

This is not a cosmetic detail. It explicitly makes the pixel-based branch less brittle to viewpoint or transform changes.

#### C. Pixel-wise similarity

`PixelwiseComparator` combines three classical metrics:

- Mean Squared Error (MSE)
- Structural Similarity Index (SSIM)
- color histogram correlation

The pipeline computes:

- normalized MSE as `1 - min(mse / 255^2, 1)`
- clipped histogram similarity in `[0, 1]`
- final pixel similarity as the average of:
  - normalized MSE
  - SSIM
  - histogram similarity

So pixel similarity is:

`pixel_similarity = (normalized_mse + ssim + histogram_similarity) / 3`

#### D. DINOv2 semantic similarity

`DINOv2Comparator` uses the Hugging Face model `facebook/dinov2-base`.

It extracts:

- the CLS token as a global representation,
- patch tokens as spatial representations.

Two semantic signals are then computed:

- **global cosine similarity** between CLS tokens,
- **region similarity** using the top-`k` most salient patches (default `k = 50`).

Region similarity itself is formed from:

- cosine similarity between the mean of important patches, and
- average best-match patch similarity.

This design is meaningful because phishing mimicry often preserves local high-value regions such as:

- header/logo area,
- login panel,
- CTA buttons,
- or multi-column layout.

#### E. Final combined score

The default combined detector uses:

- `70%` DINOv2 region similarity
- `30%` pixel similarity

So the final score is:

`combined_score = 0.7 * region_similarity + 0.3 * pixel_similarity`

#### F. Verdict logic

The code maps the combined score to verdicts as follows:

- `MATCHES_LEGITIMATE` if `combined_score >= 0.85`
- `POTENTIAL_PHISHING` if `0.70 <= combined_score < 0.85`
- `LIKELY_SAFE` otherwise

The same threshold pattern is also used for DINO-only and pixel-only comparisons.

#### G. Database-style matching

`compare_with_database(...)` compares a suspicious screenshot against multiple legitimate references and selects the best-scoring match.

This is important because phishing detection is usually not a one-reference problem. Real deployment needs a one-to-many reference comparison workflow.

### Advantage of this method

The visual comparison approach provides:

1. strong sensitivity to real end-user-visible impersonation,
2. robustness to small textual or color changes via DINOv2,
3. robustness to viewpoint shifts via rotation correction,
4. and interpretable outputs through separate semantic, pixel, SSIM, and final scores.

## 2.1.3 Screenshot rendering and live collection support

### What is used

`screenshot.py` provides two supporting capabilities:

- rendering HTML into PNG screenshots,
- collecting live screenshots from streamed dataset URLs.

### Why it is used

The visual pipeline needs consistent image inputs. In a phishing dataset, some samples may exist only as HTML; others may require live fetching.

### How it is used

- `render_html_to_png(...)` renders HTML in Playwright and can block external requests for deterministic rendering.
- `collect_live_samples(...)` streams dataset records, tests reachability, and stores screenshots plus CSV/JSONL logs.

### Advantage of this method

This allows the repository to support:

- offline HTML-to-image conversion,
- live collection experiments,
- and reproducible screenshot generation for comparison/classification pipelines.

## 2.2 Task 2: Predicting Benign vs Phishing

The repository contains both **direct binary classifiers** and **derived binary decisions**.

A very important methodological distinction is this:

- some binary models are trained directly on the original `benign vs phishing` label,
- other binary outputs are derived from a multiclass brand model using `known brand => phishing` and `unknown => benign`.

Those are related but **not identical tasks**, so their numbers should not be compared naively.

## 2.2.1 DOM supervised binary classification

### What is used

`HTMLSupervisedDOMClassifier` in `dom_tree_node_detector.py`.

Supported algorithms in code:

- Random Forest
- Extra Trees
- Gradient Boosting
- Logistic Regression

Committed evaluation artifacts are present for:

- Random Forest
- Extra Trees

### Why it is used

Once the semantic DOM profile exists, a direct binary classifier is the most operational way to answer the triage question:

- Is this page benign?
- Or is it phishing?

### How it is used

The pipeline is:

1. Build a DOM semantic profile.
2. Flatten it into a feature dictionary.
3. Hash it into a fixed `4096`-dimensional vector using `FeatureHasher(..., alternate_sign=False)`.
4. Standardize the feature vector with `StandardScaler`.
5. Train a supervised classifier.
6. Predict phishing probability and threshold it at `0.5`.

The flattened feature groups are:

- `metric::...`
- `role::...`
- `path::...`
- `depth::...`
- `behavior::...`
- `text::...`
- `attr::...`

The important model settings used in code are:

- Random Forest:
  - `n_estimators=200`
  - `max_depth=20`
  - `min_samples_split=10`
  - `class_weight='balanced'`
- Extra Trees:
  - same main settings as Random Forest
- Logistic Regression:
  - `max_iter=1000`
  - `solver='liblinear'`
  - `class_weight='balanced'`
- Gradient Boosting:
  - `learning_rate=0.05`
  - tree depth `3`

The class has support for:

- full-data training,
- streaming buffered training,
- validation tracking,
- optional warm-start growth,
- optional batch-ensemble inference.

### Advantage of this method

This method converts detailed DOM semantics into a scalable fixed-size representation that can be fed to standard classifiers. It gives a much more compact deployment path than storing or comparing full DOM trees at inference time.

## 2.2.2 DOM unsupervised / anomaly-based binary classification

### What is used

`HTMLUnsupervisedDOMClassifier` in `dom_tree_node_detector.py`.

Supported algorithms:

- Isolation Forest
- One-Class SVM

### Why it is used

This branch is designed for the scenario where **benign structure is easier to model than phishing diversity**.

Instead of learning a benign-vs-phishing boundary directly, it learns normal DOM structure from benign pages and flags structural anomalies as phishing-like.

### How it is used

The feature construction is the same as the supervised DOM binary path:

- build semantic DOM profile,
- flatten to dictionary,
- hash to 4096 dimensions,
- standardize.

The major difference is training:

- only benign samples are used to fit the anomaly model.
- the default contamination parameter is `0.15`.
- Isolation Forest uses `300` estimators.
- One-Class SVM uses `RBF` kernel with `nu=contamination` and `gamma='scale'`.

The implementation includes memory-protected streaming logic:

- benign samples are buffered up to `100,000`,
- then subsampled to `50,000` for Isolation Forest or `20,000` for One-Class SVM,
- the scaler is fitted or updated incrementally,
- and the anomaly model is re-fitted after each buffer.

### Advantage of this method

The advantage is conceptual simplicity under weak supervision: it tries to define “normal benign DOM” and treat departures as suspicious. It is useful as a baseline or fallback strategy when labeled phishing data is limited.

## 2.2.3 Visual supervised binary classification

### What is used

`VisualSupervisedDetector` in `visual_supervised_detector.py`.

Algorithms supported in code:

- Random Forest
- Extra Trees
- Logistic Regression
- Gradient Boosting

### Why it is used

If a screenshot is available, direct visual classification gives a binary answer without depending on DOM availability, source-code quality, or site obfuscation.

### How it is used

The pipeline is:

1. discover images named `clean_js_on.jpg`,
2. extract DINOv2 embeddings,
3. scale them,
4. train a binary classifier,
5. report accuracy, ROC-AUC, confusion matrix, and per-class metrics.

The DINOv2 feature vector is not just a CLS token. It concatenates:

- CLS token (`768` dims)
- patch mean (`768` dims)
- patch std (`768` dims)
- patch max (`768` dims)

This produces a `3072`-dimensional visual descriptor.

The committed classifiers use these settings:

- Random Forest:
  - `n_estimators=200`
  - `max_depth=20`
  - `class_weight='balanced'`
- Extra Trees:
  - `n_estimators=300`
  - `max_depth=24`
  - `class_weight='balanced'`
- Logistic Regression:
  - `max_iter=1000`
  - `solver='liblinear'`
  - `class_weight='balanced'`
- Gradient Boosting:
  - `n_estimators=250`
  - `learning_rate=0.05`
  - `max_depth=3`

### Advantage of this method

This method gives a strong screenshot-only phishing detector using a pretrained foundation vision backbone, which is valuable when HTML is unavailable, malformed, or intentionally hostile.

## 2.2.4 Visual reference comparison with binary context

### What is used

`compare_to_reference(...)` in `visual_supervised_detector.py`.

### Why it is used

This is a bridge between the pairwise-comparison task and the binary-classification task. It asks:

- how visually similar is the suspect page to a reference brand page?
- and does the trained classifier also think it is phishing?

### How it is used

The function:

- computes cosine similarity between the two DINOv2 feature vectors,
- computes phishing probability from the trained binary classifier,
- returns `possible_phishing` when:
  - normalized visual similarity `> 0.7`, and
  - phishing score `> 0.5`.

### Advantage of this method

This is a useful operational fusion utility because it does not rely on either visual similarity or binary probability alone.

## 2.3 Task 3: Predicting Target Classes / Brands

## 2.3.1 DOM multiclass target prediction

### What is used

`HTMLTargetMulticlassDOMClassifier` in `dom_target_multiclass_detector.py`.

The current committed runtime mapping in `dom_stage1_mapping.json` defines **9 known target classes plus `unknown_agg`**, for a total of **10 classes**:

- `facebook`
- `meta`
- `usps`
- `at&t`
- `robinhood`
- `whatsapp`
- `booking`
- `instagram`
- `naver`
- `unknown_agg`

### Why it is used

Binary phishing detection is not enough for brand-targeted phishing analysis. A security workflow also needs to answer:

- which brand is being impersonated?
- or is this page outside the modeled brand set?

### How it is used

#### A. Label construction

The multiclass labels are created by aligning each feature chunk with a sidecar `y11` label file.

Important behavior in the mapping code:

- known phishing targets are assigned specific IDs,
- benign pages and phishing pages outside the chosen target set fall into `unknown` / `unknown_agg`.

#### B. Feature representation

The classifier reuses the same hashed DOM semantic feature representation as the DOM binary branch:

- 4096-dimensional hashed vector
- standardized with `StandardScaler`

#### C. Model families

Supported in code:

- Random Forest
- Extra Trees
- Gradient Boosting
- Logistic Regression
- XGBoost (optional)
- LightGBM (optional)

Committed result artifacts are present for:

- Random Forest
- Extra Trees
- Logistic Regression
- LightGBM

Key settings in code:

- Random Forest / Extra Trees:
  - `n_estimators=300`
  - `max_depth=24`
  - `min_samples_split=10`
  - `class_weight='balanced_subsample'`
- Logistic Regression:
  - `max_iter=1200`
  - `class_weight='balanced'`
  - `solver='lbfgs'`
- LightGBM:
  - `n_estimators=300`
  - `learning_rate=0.05`
  - `num_leaves=31`
  - `max_depth=6`
  - GPU preferred if available, CPU fallback if not

#### D. Class balancing inside training

The multiclass trainer explicitly balances classes before fitting.

The current stage-1 training log and metadata show this behavior clearly:

- raw train size: `66,499`
- balanced train size used for fitting: `48,755`
- validation fraction: `0.1`
- known-class cap ratio applied in committed runs: `max_class_ratio = 2.5`
- resulting cap: `7,132` for the majority known classes in the logged run
- default unknown cap rule in code: `median_known_count * unknown_max_multiplier`, with multiplier default `2.0`

This balancing is not cosmetic. It is a direct attempt to prevent the dominant target classes from overwhelming the minority classes.

#### E. Derived binary output

The multiclass classifier also derives a binary phishing decision by treating:

- `predicted_class != unknown_agg` as phishing,
- `predicted_class == unknown_agg` as benign.

This derived binary metric is useful, but it must be read as a **known-target-vs-unknown** decision, not as a generic screenshot/DOM phishing detector in the same sense as the direct binary models.

### Advantage of this method

This gives target-level phishing attribution, which is the most informative outcome for brand-monitoring and threat intelligence workflows.

## 2.3.2 DOM class-imbalance and “no-majority” studies

### What is used

The repository contains multiple scripts that test the effect of excluding dominant target classes:

- `train_without_majority.py`
- `train_no_majority_full.py`
- `train_no_majority_simple.py`
- `train_no_majority_stream.py`
- `train_subsample_no_majority.py`
- `run_no_majority_training.py`

The excluded classes are consistently:

- `facebook`
- `meta`
- `usps`

### Why it is used

These three classes dominate the stage-1 dataset. The experiments test whether poor macro-F1 is caused by the model itself or by skewed class distribution.

### How it is used

The scripts explore several variants:

- **full-data remap**: exclude major classes from training and remap them to `unknown_agg` in evaluation.
- **stream-sampled training**: cap per-class samples while streaming chunks to avoid out-of-memory issues.
- **subsampled 7-class evaluation**: train and test on the remaining seven classes only.

This last variant is the cleanest controlled comparison and is the basis of `results_comparison_summary.txt`.

### Advantage of this method

It directly measures whether the system becomes fairer and more useful for smaller target classes once the largest classes are removed.

## 2.3.3 Experimental neural DOM classifiers

### What is used

Two experimental neural DOM branches are present:

- `mlp_subsample_train.py`
- `two_stage_mlp_train.py`

### Why they are used

These scripts explore whether a neural model can improve class separation beyond tree-based models, especially for minority brands and unknown rejection.

### How they are used

#### `mlp_subsample_train.py`

This script implements a neural multiclass training pipeline over hashed DOM features using:

- MLP and 1D-CNN model definitions,
- class-balanced batch sampling,
- weighted random sampling,
- optional focal loss,
- synthetic minority interpolation,
- optional unknown-threshold calibration,
- optional per-class confidence thresholds.

#### `two_stage_mlp_train.py`

This script implements an explicitly decomposed two-stage pipeline:

- **Stage 1**: known-vs-unknown detection
- **Stage 2**: classification among known classes only

It also includes:

- boundary-focused synthetic augmentation for minority classes,
- threshold search for unknown rejection,
- per-class threshold calibration for stage 2,
- early stopping and learning-rate scheduling.

### Advantage of these methods

These scripts are valuable as research directions because they tackle two hard problems directly:

- minority-class scarcity,
- and the need to reject “unknown” pages instead of forcing every page into a known brand.

### Important note on results

The repository contains model weights for these branches (for example `mlp_subsample_model.pt`, `cnn1d_subsample_model.pt`, `mlp_top5_plus_unknown.pt`), but it does **not** contain a committed evaluation report for them comparable to the JSON artifacts used elsewhere. Therefore they are documented methodologically, but this report does not invent numerical conclusions for them.

## 2.3.4 Visual brand classification and dual-task modeling

### What is used

`visual_dual_train_test.py` is the core script for visual brand prediction.

It uses:

- DINOv2 features from `visual_supervised_detector.py`
- classical models:
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - Logistic Regression
- neural heads:
  - MLP
  - CNN1D

### Why it is used

A screenshot-based detector is much more informative when it can answer not just “phishing or benign”, but “which brand is being mimicked”.

### How it is used

#### A. Label design

The visual brand classifier builds the top-`k` phishing brands from the manifest and appends `unknown`.

The committed top-10 brand set is:

- `microsoft`
- `paypal`
- `facebook`
- `genericemail`
- `att`
- `amazon`
- `steam`
- `usps`
- `dhl`
- `metamask`
- plus `unknown`

#### B. Input source

The script reads `visual_consolidated_screenshots/manifest.csv`, which contains:

- `new_path`
- `source_path`
- `is_phishing`
- `brand_primary`
- `brand_all`

#### C. Feature extraction

The visual dual pipeline uses the same `3072`-dimensional DINOv2 embedding construction as the visual binary pipeline.

#### D. Model heads

Classical heads:

- Random Forest: `250` trees, `max_depth=24`, balanced classes
- Extra Trees: `350` trees, `max_depth=24`, balanced classes
- Logistic Regression: `max_iter=1500`, balanced classes
- Gradient Boosting: supported in code but intentionally omitted from the default `all` bundle

Neural heads:

- **MLP**:
  - default hidden sizes `1024, 512, 256`
  - dropout `0.2`
  - Adam optimizer
- **CNN1D**:
  - 1D convolutions over the embedding vector treated as a 1D signal
  - channels `32 -> 64 -> 128`
  - adaptive average pooling to length `64`
  - dense layer of size `512`

For neural heads, the code explicitly compensates for imbalance:

- binary case: positive-class weighting in `BCEWithLogitsLoss`
- multiclass case: inverse-frequency weights in `CrossEntropyLoss`

#### E. Two binary interpretations exist in artifacts

The visual dual branch is especially important because the repository contains **two binary result interpretations**:

1. **Direct binary evaluation**
   - stored in `visual_dual_report.json`, `visual_dual_report_all.json`, `visual_dual_cnn1d_e100.json`, and `visual_dual_mlp_e100.json`
   - this treats the manifest’s phishing flag directly as the binary target
   - test split: `600 phishing / 442 benign`

2. **Brand-gated binary evaluation**
   - stored in `visual_dual_report_all_brand_gated.json`
   - this treats:
     - `predicted_brand != unknown` as phishing
     - `predicted_brand == unknown` as benign
   - test split: `354 phishing / 688 benign`
   - this is a stricter “known top-brand phishing vs unknown” decision rule

### Advantage of this method

This is the most flexible visual pipeline in the repository because it supports:

- exact brand prediction,
- binary phishing prediction,
- classical and deep heads on the same feature backbone,
- and alternative binary label semantics depending on the deployment objective.

## 2.4 Result visualization and traceability

The repository also contains dedicated result-visualization scripts:

- `plot_training_visualizations.py`
- `plot_lightgbm_builtin_visuals.py`

These generate:

- DOM evaluation bars,
- DOM fit-curve summaries,
- visual epoch curves for neural models,
- visual model comparison bars,
- and LightGBM-specific plots such as feature importance and split histograms.

This is methodologically important because it shows the project is not storing only final scalar metrics; it is also preserving training/evaluation diagnostics.

## 3. Results

## 3.1 Result interpretation notes

Before listing numbers, three cautions are necessary:

1. **Pairwise comparison stage**
   - The repository contains code and generated image artifacts for pairwise comparison, but no committed machine-readable benchmark of final scores.
2. **Direct binary vs derived binary**
   - Direct binary classifiers are not numerically identical tasks to brand-gated known-vs-unknown binary outputs.
3. **Older vs newer DOM multiclass logs**
   - `all_model_results.jsonl` is an earlier experimental multiclass result log.
   - `dom_brand_stage1_results.jsonl` is the current stage-1 multiclass benchmark log.
   - The latter should be treated as the main DOM multiclass result source.

## 3.2 Task 1 Results: Comparing Two Websites

## 3.2.1 What result evidence exists in the repository

### Visual comparison

The visual pairwise-comparison path has clear evidence of execution:

- `compare_pgportal.py` compares `screenshots/pgportal.png` against `example-pgportal.html` rendered into `screenshots/example-pgportal-captured.png`.
- The repository also contains attention-map outputs:
  - `screenshots/pgportal_attention.png`
  - `screenshots/example-pgportal_attention.png`
- `visual_similarity.py` also contains demo code for:
  - same-site comparison (`SBI` vs `SBI`),
  - different-site comparison (`SBI` vs `ICICI`),
  - and rotation-invariance testing using `example_rotated.html`.
- The corresponding screenshots exist in `screenshots/`, including:
  - `example_original.png`
  - `example_rotated.png`
  - `sbi_bank.png`
  - `icici_bank.png`
  - `example-pgportal-captured.png`

### DOM comparison

The DOM pairwise-comparison path is implemented and callable via `compare_html(...)` and `detect_against_references(...)`, but there is no committed benchmark file of DOM pairwise similarity scores.

## 3.2.2 What numerical result evidence does **not** exist

No committed JSON, CSV, or markdown table in the repository stores final pairwise-comparison scores such as:

- DOM similarity values across a benchmark set,
- visual combined scores across a benchmark set,
- pairwise comparison ROC curves,
- or threshold-based precision/recall summaries.

Therefore, for Task 1, the repository supports a **detailed methodology and executable demos**, but it does **not** provide a committed aggregate benchmark.

## 3.2.3 Supporting live-collection result

`screenshot.py` was also used in live collection experiments, and the committed result logs show how hard live phishing capture is:

- `live_collect_pilot/live_results_phishing.csv`:
  - scanned rows: `2,189`
  - reachable phishing screenshots captured: `9`
- `live_collect_full_phishing/live_results_phishing.csv`:
  - scanned rows: `298,326`
  - reachable phishing screenshots captured: `8`

This is a useful empirical result because it explains why the main visual modeling work relies on consolidated/local screenshot assets instead of live-only collection.

## 3.3 Task 2 Results: Predicting Benign vs Phishing

## 3.3.1 DOM supervised binary results

Source: `dom_model_compare_sup_eval.json`

Test set size from the committed report:

- benign: `48,851`
- phishing: `41,261`
- total: `90,112`

| Model | Accuracy | Balanced Accuracy | Benign F1 | Phishing F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8187 | 0.8134 | 0.8399 | 0.7912 | 0.8897 | 0.8630 |
| Extra Trees | 0.8155 | 0.8106 | 0.8363 | 0.7887 | 0.8797 | 0.8329 |

### Interpretation

- The supervised DOM binary path is a credible direct detector.
- Random Forest is slightly stronger than Extra Trees on all headline metrics in the committed evaluation.
- Performance is reasonably balanced across benign and phishing classes, which is important for a true binary detector.

## 3.3.2 DOM unsupervised binary results

Sources:

- `dom_model_compare_iforest_eval.json`
- `dom_model_compare_ocsvm_eval.json`
- `dom_model_compare_ocsvm_train.json`

### Training evidence

The One-Class SVM training log shows a large-scale benign-only fit:

- training chunks: `7,831`
- total streamed samples: `501,184`
- benign samples seen: `278,222`

### Test results

| Model | Accuracy | Balanced Accuracy | Benign F1 | Phishing F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.4736 | 0.4385 | 0.6378 | 0.0371 | 0.1861 | 0.3090 |
| One-Class SVM | 0.4953 | 0.4628 | 0.6456 | 0.1236 | 0.2445 | 0.3418 |

### Interpretation

- The unsupervised DOM anomaly approach underperforms badly as a phishing detector in the committed results.
- Both models preserve benign recall much more than phishing recall.
- One-Class SVM is better than Isolation Forest here, but still far below the supervised DOM binary models.

This is a strong empirical finding: in this repository, **benign-only anomaly modeling is not competitive with supervised DOM classification**.

## 3.3.3 Visual supervised binary benchmark results

Sources:

- `visual_benchmark_report.json`
- `visual_benchmark_report_full.json`

The repository contains two committed benchmark runs.

### Smaller benchmark run

Train/test sizes:

- train: `2,195`
- test: `549`
- benign support: `249`
- phishing support: `300`

| Model | Accuracy | ROC-AUC | Benign F1 | Phishing F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.8907 | 0.9469 | 0.8859 | 0.8951 |
| Extra Trees | 0.8852 | 0.9529 | 0.8814 | 0.8889 |
| Logistic Regression | 0.8743 | 0.9367 | 0.8589 | 0.8867 |
| Gradient Boosting | 0.8798 | 0.9420 | 0.8685 | 0.8893 |

### Full benchmark run

Train/test sizes:

- train: `4,166`
- test: `1,042`
- benign support: `442`
- phishing support: `600`

| Model | Accuracy | ROC-AUC | Benign F1 | Phishing F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.8695 | 0.9545 | 0.8553 | 0.8811 |
| Extra Trees | 0.8695 | 0.9569 | 0.8580 | 0.8792 |
| Logistic Regression | 0.8637 | 0.9400 | 0.8429 | 0.8797 |
| Gradient Boosting | 0.8647 | 0.9433 | 0.8327 | 0.8864 |

### Interpretation

- Visual supervised binary classification is strong and consistent.
- On the full benchmark, Random Forest and Extra Trees tie on accuracy (`0.8695`), while Extra Trees has the strongest ROC-AUC (`0.9569`).
- The visual direct binary branch is therefore one of the best-performing pure binary pipelines in the repository.

## 3.3.4 Visual dual-task direct binary results

Source: `visual_dual_report_all.json`

This is the direct phishing-vs-benign binary interpretation using the same DINOv2 feature backbone but evaluated jointly with brand prediction.

Test split:

- benign: `442`
- phishing: `600`
- total: `1,042`

| Model | Brand Accuracy | Brand Macro-F1 | Binary Accuracy | Binary F1 | Binary ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN1D | 0.7994 | 0.6697 | 0.8512 | 0.8681 | 0.9315 |
| Extra Trees | 0.8551 | 0.7473 | 0.8906 | 0.9000 | 0.9655 |
| Logistic Regression | 0.8608 | 0.7785 | 0.8695 | 0.8861 | 0.9426 |
| MLP | 0.8560 | 0.7683 | 0.8781 | 0.8930 | 0.9559 |
| Random Forest | 0.8493 | 0.7385 | 0.8820 | 0.8930 | 0.9582 |

### Interpretation

- In the direct visual dual setup, **Extra Trees** gives the best binary result (`0.8906` accuracy, `0.9000` F1, `0.9655` ROC-AUC).
- **Logistic Regression** gives the best brand accuracy (`0.8608`) and best brand macro-F1 (`0.7785`).
- This means the best brand predictor is not the same as the best direct binary detector.

## 3.3.5 Visual dual-task brand-gated binary results

Source: `visual_dual_report_all_brand_gated.json`

This is the stricter binary interpretation where only modeled known-brand predictions count as phishing.

Test split:

- benign / unknown: `688`
- phishing / known-brand: `354`
- total: `1,042`

| Model | Brand Accuracy | Brand Macro-F1 | Binary Accuracy | Binary F1 |
|---|---:|---:|---:|---:|
| CNN1D | 0.8234 | 0.7323 | 0.8560 | 0.8206 |
| Extra Trees | 0.8551 | 0.7473 | 0.8858 | 0.8465 |
| Logistic Regression | 0.8608 | 0.7785 | 0.8916 | 0.8600 |
| MLP | 0.7764 | 0.6971 | 0.8081 | 0.7768 |
| Random Forest | 0.8493 | 0.7385 | 0.8800 | 0.8375 |

### Interpretation

- In the brand-gated setup, **Logistic Regression** becomes the strongest overall model on both brand accuracy and derived binary accuracy/F1.
- This is an important repository finding: once the binary definition becomes “known brand vs unknown,” the most suitable model shifts.

## 3.4 Task 3 Results: Predicting Target Classes / Brands

## 3.4.1 Earlier DOM multiclass experimental results

Source: `all_model_results.jsonl`

This file stores an earlier generation of DOM multiclass experiments with a different data regime. The committed evaluation summaries are:

| Model | Multiclass Accuracy | Macro-F1 | Weighted F1 | Derived Binary Accuracy | Derived Binary F1 | OVR Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.3496 | 0.3894 | 0.3609 | 0.9488 | 0.9418 | 0.8308 |
| Extra Trees | 0.1402 | 0.2192 | 0.1349 | 0.8966 | 0.8903 | 0.7905 |
| LightGBM | 0.4946 | 0.4881 | 0.5107 | 0.9482 | 0.9414 | 0.8226 |

### Interpretation

This older result log shows a very important pattern:

- exact brand attribution was much harder in the earlier setup,
- but the derived known-vs-unknown binary decision was already comparatively easy.

In other words, the system could often detect that a page belonged to the “known phishing target” side even when it still struggled to assign the correct exact class.

## 3.4.2 Current DOM stage-1 multiclass benchmark

Primary sources:

- `dom_brand_stage1_results.jsonl`
- `dom_unsup_subsample_brand_stage1/metadata.json`
- `BRAND_WISE_METRICS.md`

Current committed stage-1 dataset facts:

- known classes: `9`
- total classes including unknown: `10`
- train rows: `66,499`
- test rows: `16,626`
- test class distribution:
  - `facebook`: `2,800`
  - `meta`: `3,827`
  - `usps`: `3,159`
  - `at&t`: `1,345`
  - `robinhood`: `1,031`
  - `whatsapp`: `906`
  - `booking`: `727`
  - `instagram`: `718`
  - `naver`: `713`
  - `unknown_agg`: `1,400`

### Overall model results

The raw JSONL evaluation log gives these stage-1 multiclass results:

| Model | Multiclass Accuracy | Macro-F1 | Weighted F1 | Derived Binary Accuracy | Derived Binary F1 | OVR Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6981 | 0.6840 | 0.7188 | 0.8646 | 0.9222 | 0.9230 |
| Extra Trees | 0.4583 | 0.4959 | 0.4955 | 0.8869 | 0.9369 | 0.8901 |
| Logistic Regression | 0.6571 | 0.6231 | 0.6591 | 0.8939 | 0.9424 | 0.8243 |
| LightGBM | 0.7103 | 0.6955 | 0.7251 | 0.8727 | 0.9274 | 0.9227 |

### Interpretation

- **LightGBM** is the strongest committed DOM brand classifier overall.
- **Random Forest** is a close second on exact brand classification.
- **Logistic Regression** is weaker on exact brand assignment than LightGBM/RandomForest, but its **derived binary** F1 is the strongest.
- **Extra Trees** is the clearest example of the difference between tasks: it is weak on exact brand prediction, yet strong on the easier known-vs-unknown derived binary metric.

### Note on a minor metric discrepancy

`BRAND_WISE_METRICS.md` lists DOM LightGBM brand accuracy as `0.7118`, while the raw machine-generated `dom_brand_stage1_results.jsonl` evaluation entry records `0.7103`. This report treats the JSONL evaluation log as the authoritative source because it is the direct run output.

## 3.4.3 DOM brand-level observations from the committed brand-wise breakdown

Primary source: `BRAND_WISE_METRICS.md`

For the strongest DOM model family (LightGBM), per-class accuracy is:

- `facebook`: `0.6446`
- `meta`: `0.7021`
- `usps`: `0.7708`
- `at&t`: `0.7063`
- `robinhood`: `0.8070`
- `whatsapp`: `0.6115`
- `booking`: `0.7565`
- `instagram`: `0.7145`
- `naver`: `0.7391`
- `unknown_agg`: `0.7007`

### Interpretation

The main DOM class-level pattern is:

- strongest classes: `robinhood`, `usps`, `booking`, `naver`
- mid-range classes: `meta`, `at&t`, `instagram`, `unknown_agg`
- weakest among known targets: `whatsapp` and `facebook`

This agrees with `results_comparison_summary.txt`, which explicitly identifies `whatsapp` as a consistently difficult class.

## 3.4.4 DOM no-majority (7-class) controlled results

Primary sources:

- `dom_brand_stage1_results.jsonl`
- `results_comparison_summary.txt`

The controlled no-majority experiment excludes:

- `facebook`
- `meta`
- `usps`

Remaining classes:

- `at&t`
- `robinhood`
- `whatsapp`
- `booking`
- `instagram`
- `naver`
- `unknown_agg`

Committed 7-class subsample results:

| Model | Test Accuracy | Macro-F1 | Weighted F1 |
|---|---:|---:|---:|
| Random Forest | 0.7528 | 0.7702 | 0.7605 |
| Extra Trees | 0.7494 | 0.7667 | 0.7564 |
| Logistic Regression | 0.6746 | 0.6843 | 0.6713 |
| LightGBM | 0.7504 | 0.7678 | 0.7573 |

The repository’s own comparison summary reports the improvement from the full 10-class setup to the 7-class setup as:

- Random Forest: `69.81% -> 75.28%` accuracy (`+5.47 pts`)
- Extra Trees: `45.83% -> 74.94%` accuracy (`+29.11 pts`)
- Logistic Regression: `65.71% -> 67.46%` accuracy (`+1.75 pts`)
- LightGBM: `71.03% -> 75.04%` accuracy (`+4.01 pts`)

### Interpretation

This is one of the clearest methodological findings in the repository:

- removing the three dominant classes improves **all** algorithms,
- the biggest gain is for **Extra Trees**, showing that it was highly sensitive to class imbalance,
- and the best 7-class performer becomes **Random Forest**, with LightGBM extremely close.

### Additional no-majority experiment to note

`train_no_majority_stream.py` produced a committed result entry with extremely poor performance:

- accuracy: `0.0352`
- macro-F1: `0.0211`
- binary accuracy: `0.0589`

This should be treated as a **failed resource-saving experiment**, not as the representative no-majority result. The controlled subsample 7-class study is the meaningful no-majority benchmark.

## 3.4.5 Visual brand-classification results

Primary sources:

- `visual_dual_report_all.json`
- `visual_dual_report_all_brand_gated.json`
- `BRAND_WISE_METRICS.md`

Current committed visual brand-classification dataset:

- total images: `5,208`
- train: `4,166`
- test: `1,042`
- top brands + unknown:
  - `microsoft`
  - `paypal`
  - `facebook`
  - `genericemail`
  - `att`
  - `amazon`
  - `steam`
  - `usps`
  - `dhl`
  - `metamask`
  - `unknown`

### Direct visual brand results

| Model | Brand Accuracy | Brand Macro-F1 | Weighted F1 |
|---|---:|---:|---:|
| CNN1D | 0.7994 | 0.6697 | 0.8258 |
| Extra Trees | 0.8551 | 0.7473 | 0.8819 |
| Logistic Regression | 0.8608 | 0.7785 | 0.8870 |
| MLP | 0.8560 | 0.7683 | 0.8844 |
| Random Forest | 0.8493 | 0.7385 | 0.8753 |

### Interpretation

- **Logistic Regression** is the best committed direct visual brand classifier by both brand accuracy and brand macro-F1.
- **MLP** is very close behind Logistic Regression.
- **CNN1D** is the weakest of the five committed direct visual brand models in its default run.

## 3.4.6 Extended-epoch neural visual runs

Primary sources:

- `visual_dual_cnn1d_e100.json`
- `visual_dual_mlp_e100.json`

These files capture longer neural training runs (`100` epochs).

| Model | Brand Accuracy | Brand Macro-F1 | Binary Accuracy | Binary F1 | Binary ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN1D (100 epochs) | 0.8369 | 0.7398 | 0.8541 | 0.8725 | 0.9374 |
| MLP (100 epochs) | 0.8580 | 0.7712 | 0.8791 | 0.8932 | 0.9544 |

### Interpretation

- The longer CNN1D run improves substantially over the default CNN1D result, especially on brand accuracy and macro-F1.
- The longer MLP run produces a small but real improvement over the default MLP brand metrics.
- Even after the extended run, Logistic Regression remains the strongest committed direct visual brand model overall, but the gap is narrow.

## 3.4.7 Visual brand-wise observations

Primary source: `BRAND_WISE_METRICS.md`

### Strong visual classes

Across the committed visual models, the most consistently strong classes are:

- `microsoft`
- `paypal`
- `genericemail`
- `usps`

Examples from the per-brand tables:

- `microsoft`: up to `0.9512` accuracy under Logistic Regression and MLP
- `paypal`: `0.9565` accuracy under several models
- `genericemail`: up to `0.9655` under Logistic Regression and MLP
- `usps`: `0.9333` in several runs

### Weak visual classes

The most difficult committed visual classes are:

- `amazon`
- `metamask`
- and, in some models, `steam` / `dhl` in one-vs-rest precision terms

Examples:

- `amazon` is only `0.20-0.24` accuracy in multiple models
- `metamask` ranges roughly from `0.4615` to `0.6154`

### Important nuance

Some classes such as `att` show perfect class accuracy/recall in the brand table but low precision in one-vs-rest terms because the model over-predicts them. This means per-class “correct out of support” and one-vs-rest precision should be interpreted together.

## 3.5 Supporting visualization outputs

The repository also preserves visual diagnostics for the reported methods.

### DOM visualizations

From `training_visualizations/README.md` and `training_visualizations/lightgbm_builtin/README.md`, the committed outputs include:

- `training_visualizations/dom/dom_eval_metrics.png`
- `training_visualizations/dom/dom_fit_round_curves.png`
- `training_visualizations/dom/dom_train_val_metrics.png`
- `training_visualizations/lightgbm_builtin/feature_importance_gain.png`
- `training_visualizations/lightgbm_builtin/feature_importance_split.png`
- `training_visualizations/lightgbm_builtin/learning_curve_dom_curve_fallback.png`
- `training_visualizations/lightgbm_builtin/split_value_histogram_feature_1293.png`
- LightGBM tree visualizations in both Matplotlib and Graphviz form

An important detail recorded in the LightGBM visualization README is that:

- native LightGBM eval history was not available,
- so the learning curve figure was reconstructed from the DOM JSONL logs.

### Visual model visualizations

Committed visual plots include:

- `training_visualizations/visual/cnn1d_epoch_training.png`
- `training_visualizations/visual/mlp_epoch_training.png`
- `training_visualizations/visual/neural_val_macro_f1_comparison.png`
- `training_visualizations/visual/visual_model_eval_overview.png`

These support the narrative that the repository did not stop at single scalar results; it also tracked training trajectories for neural visual models.

## 4. Overall Conclusions Grounded in the Repository

1. **Website-to-website comparison is fully implemented in both DOM and visual form, but the repository does not contain a committed aggregate numeric benchmark for that stage.**
   - What exists is executable methodology, generated screenshots, and attention visualizations.

2. **For direct binary phishing detection, supervised methods clearly outperform anomaly-based DOM methods.**
   - DOM supervised Random Forest / Extra Trees are materially better than DOM Isolation Forest / One-Class SVM.
   - Visual supervised binary classifiers are strong, with full-benchmark ROC-AUC up to `0.9569`.

3. **For exact class prediction, the strongest current DOM model is LightGBM, while the strongest current direct visual brand model is Logistic Regression.**
   - DOM stage-1 best committed multiclass accuracy: `0.7103` (LightGBM)
   - Visual direct best committed brand accuracy: `0.8608` (Logistic Regression)

4. **Class imbalance is one of the dominant issues in the DOM multiclass pipeline, and the repository contains direct evidence that rebalancing/excluding majority classes helps substantially.**
   - The 7-class no-majority study improves all algorithms.
   - Extra Trees benefits the most from removing the three dominant classes.

5. **The repository distinguishes between two different binary questions in the multiclass setting:**
   - generic phishing vs benign,
   - and known-brand phishing vs unknown.
   
   That distinction is essential to interpreting the reported numbers correctly.

6. **The visual pipeline is more mature than a single-method screenshot classifier.**
   - It includes pairwise similarity, direct binary classification, direct brand classification, brand-gated binary evaluation, attention visualization, and screenshot collection utilities.

## 5. File-to-Role Summary

For methodology reporting purposes, the repository’s DOM/visual files can be understood in these groups:

### Core DOM implementation

- `dom_tree_builder.py`: builds explicit DOM trees from live URLs or HTML strings
- `dom_only_robust_detector.py`: standalone DOM structural comparison
- `dom_tree_node_detector.py`: DOM comparison + supervised/unsupervised binary classifiers
- `dom_target_multiclass_detector.py`: multiclass DOM target classifier

### Core visual implementation

- `visual_similarity.py` / `visual_similarity_detector.py`: screenshot pairwise similarity
- `visual_supervised_detector.py`: direct visual binary classifier
- `visual_dual_train_test.py`: visual brand classifier with binary views
- `compare_pgportal.py`: concrete pairwise-comparison demo
- `screenshot.py`: HTML rendering and live screenshot collection

### Data balancing / label preparation / experiments

- `create_subsample_dataset.py`
- `create_mapping.py`
- `train_without_majority.py`
- `train_no_majority_full.py`
- `train_no_majority_simple.py`
- `train_no_majority_stream.py`
- `train_subsample_no_majority.py`
- `run_no_majority_training.py`
- `mlp_subsample_train.py`
- `two_stage_mlp_train.py`

### Main result artifacts

- DOM direct binary: `dom_model_compare_sup_eval.json`
- DOM anomaly binary: `dom_model_compare_iforest_eval.json`, `dom_model_compare_ocsvm_eval.json`
- DOM current stage-1 multiclass: `dom_brand_stage1_results.jsonl`
- DOM earlier multiclass experiments: `all_model_results.jsonl`
- Visual direct binary: `visual_benchmark_report.json`, `visual_benchmark_report_full.json`
- Visual dual direct: `visual_dual_report.json`, `visual_dual_report_all.json`
- Visual dual brand-gated: `visual_dual_report_all_brand_gated.json`
- Visual extended neural runs: `visual_dual_cnn1d_e100.json`, `visual_dual_mlp_e100.json`
- Brand breakdowns and summaries: `BRAND_WISE_METRICS.md`, `results_comparison_summary.txt`

This completes the requested DOM-based and visual-based methodology and result report using only what is present in the repository.


# URL-Based Phishing Detection Report

## Scope

This report is based only on what is implemented and what is already recorded in this repository. It focuses on the URL-based pipeline because the request was corrected toward URL-based methods. DOM-oriented, visual-oriented, and `dom_unsup_*` subsample pipelines are present in the tree but are not part of this report's methodology/results scope.

The main URL-based experiment lineage in this repository is built around these files:

- Core feature source and legacy baseline definitions: `main.py`
- Main URLPhishNet experiments: `train_multiclass_brand.py`, `train_binary.py`, `train_binary_adv.py`, `train_binary_paper_data.py`, `train_binary_paper_data_adv.py`
- Ablations: `train_binary_feat_only.py`, `train_52class_feat_only.py`
- Adversarial evaluation utility: `run_52class_adversary.py`
- GAN attack/defense experiments: `train_binary_gan_adv.py`, `train_binary_gan_hardened.py`, `train_binary_gan_exact_paper.py`, `train_mahalanobis_ood.py`, `train_replication_gan_test.py`, `train_charcnn_mahalanobis_ood.py`
- Interpretation and prediction utilities: `analyze_feature_importance.py`, `generate_predictions.py`

The result artifacts used in this report come from the checked-in logs and reports under:

- `phishphresh/multiclass_brand/logs/`
- `binary_phishing/logs/`
- `binary_phishing_adv/logs/`
- `binary_paper_data/logs/`
- `binary_paper_data_adv/logs/`
- `binary_feat_only/logs/`
- `multiclass_feat_only/logs/`
- `binary_phishing_adv_gan/logs/`
- `binary_gan_hardened/logs/`
- `binary_mahalanobis_ood/logs/`
- `binary_replication_gan_test/logs/`
- `binary_gan_exact_paper/logs/`
- `feature_importance/logs/`

## Methodology

### 1. Shared URL Representation Used Across the Main Experiments

#### 1.1 Base 51 engineered features from `main.py`

The majority of the repository's URL-based experiments reuse `main.py.FeatureExtractor.extract_features()`. This function returns 51 deterministic URL/domain features. They are not learned features; they are hand-crafted measurements computed directly from the URL string.

| Feature group | What is used | Why it is used | How it is used | Advantage |
|---|---|---|---|---|
| Basic structure (15) | `domain_length`, `dot_count`, `dash_count`, `underscore_count`, `digit_count`, `uppercase_count`, `digit_ratio`, `special_char_ratio`, `domain_parts`, `longest_part`, `shortest_part`, `avg_part_length`, `vowel_count`, `consonant_count`, `has_consecutive_chars` | Phishing URLs often differ from benign URLs in length, separator usage, tokenization, and character composition. | The extractor lowercases the string, splits on `.` and counts simple structural signals. | Very cheap to compute, stable, and directly interpretable. |
| Entropy/complexity (10) | `char_entropy`, `part_length_variance`, `part_length_std`, `bigram_entropy`, `unique_char_ratio`, `transition_entropy`, `trigram_count`, `unique_trigram_ratio`, `vowel_consonant_ratio`, `consonant_clusters` | Malicious URLs often rely on randomness, unusual character transitions, or compressed token patterns. | The extractor computes Shannon-style entropy and n-gram statistics over the lowercased string. | Captures obfuscation and unnatural lexical composition without needing external lookups. |
| TLD signals (5) | `tld_length`, `is_common_tld`, `is_country_tld`, `is_suspicious_tld`, `is_indian_tld` | TLD choice is a useful prior because benign infrastructure and low-cost phishing campaigns often occupy different TLD regions. | `tldextract` is used to isolate the suffix, then fixed rule checks are applied. | Low-cost, domain-knowledge-rich features that often separate easy cases. |
| Legitimate-domain / sector / brand pattern signals (10) | `legitimate_domain_similarity`, `matches_legitimate_domain`, `legitimate_keyword_count`, `contains_legitimate_keyword`, `brand_keyword_count`, `has_brand_variation`, `finance_keywords`, `gov_keywords`, `telecom_keywords`, `tech_keywords` | The original project was built around detecting domains targeting named organizations and sectors. | The extractor compares the URL/domain against hardcoded legitimate domains and keyword lists defined in `main.py`. | Injects domain knowledge that a purely character-level model would have to rediscover from data. |
| Lexical pattern signals (11) | `dictionary_word_count`, `char_repetition_score`, `has_year_pattern`, `starts_with_number`, `ends_with_number`, `max_numeric_sequence`, `has_subdomain`, `subdomain_count`, `suspicious_pattern_count`, `has_homograph`, `complexity_score` | These capture common phishing tricks such as suspicious tokens, repeated characters, long digit runs, subdomain abuse, and visual substitutions. | The extractor applies regexes and counting logic directly to the normalized string. | Makes the model sensitive to common phishing heuristics while keeping inference deterministic. |

#### 1.2 The extra 16 URL-level structural features in `URLFeatureExtractorV2`

The dual-stream and most binary experiments do not stop at the 51 base features. They extend them with 16 additional URL-level measurements inside `URLFeatureExtractorV2`, giving a total of 67 structured features.

These 16 added values are:

1. Normalized total URL length
2. Normalized path length
3. Normalized query length
4. Normalized fragment length
5. Path depth
6. Number of query parameters
7. HTTPS flag
8. Non-standard port flag
9. `@` present in netloc
10. IP-literal host flag
11. Percent-hex escape ratio
12. Redirect-parameter flag
13. Double-slash-in-path flag
14. Shannon entropy of full URL
15. Shannon entropy of query
16. Shannon entropy of path

**Why these are used:** the 51 base features in `main.py` are largely domain-centric, while Sabir-style attacks and real phishing URLs also manipulate the path, query string, fragment, encoding style, and URL routing behavior. The extra 16 features explicitly widen coverage from the hostname to the full URL surface.

**How they are used:** `urllib.parse.urlparse()` is used to split the URL, the path/query/fragment are measured separately, URL-encoded hex sequences are counted with regex, redirect-like parameters are detected with regex, and Shannon entropy is computed for the full URL, path, and query independently.

**Advantage:** these features are complementary to the base 51 features. They improve sensitivity to path/query manipulations that are invisible if the model only inspects the domain portion.

#### 1.3 Character-level representation

The full URLPhishNet models also encode the raw URL string character by character through `encode_url()`.

- Sequence length is fixed at 256 characters.
- Index `0` is padding.
- Printable ASCII 32-127 is remapped to indices `1-96`.
- Non-ASCII or unsupported characters are mapped to index `97`.

**Why this is used:** hand-crafted features are aggregated summaries. They can miss fine-grained character substitutions such as `paypal` to `paypa1`, or suspicious path strings whose aggregate statistics remain similar.

**How it is used:** the encoded 256-length character tensor is fed into a CharCNN branch.

**Advantage:** the character branch preserves local order and exact substrings, which is important for typosquatting, token insertion, and path-level deception.

#### 1.4 Shared data handling and training conventions

Across the main URLPhishNet experiments, the following implementation conventions are reused:

- `StandardScaler` is fit on training features and then applied to test and adversarial evaluation features.
- The default split is 80/20 with `RANDOM_STATE=42`.
- Class imbalance is handled with `WeightedRandomSampler` in binary experiments and with a more elaborate weighting scheme in the 52-class model.
- Gradient clipping at norm `1.0` is used in the neural training loops.
- Saved artifacts normally include a trained model (`.pth`), a fitted scaler (`.pkl`), per-epoch CSV logs, and a text summary report.

#### 1.5 Important metric interpretation note

Many adversarial evaluation files contain only phishing URLs. In those settings:

- Accuracy is effectively the phishing detection rate.
- FNR is the phishing miss rate and is the most meaningful error measure.
- FPR is structurally `0` because there are no benign samples in the adversarial set.
- MCC often collapses to `0` even when accuracy changes, because the confusion matrix is single-class on the ground-truth side.

For that reason, this report emphasizes detection rate and FNR on adversarial-only datasets.

### 2. Shared Neural Architectures

#### 2.1 Full dual-stream URLPhishNet

The full model appears in the multiclass and binary URLPhishNet scripts.

**What is used**

- `FeatureMLP`: 67 -> 256 projection -> 2 residual blocks -> 128 output
- `CharCNN`: embedding (64-dim) -> parallel 1D convolutions with kernel sizes 3, 5, and 7 -> deeper convolution -> global max pooling + global average pooling
- Fusion classifier: concatenated 128-d feature branch and 768-d character branch = 896-d fused vector, then `896 -> 512 -> 256 -> output`

**Why it is used**

- The feature branch carries explicit domain knowledge.
- The character branch carries exact sequential evidence.
- The fusion head allows the final classifier to combine high-level domain heuristics with raw lexical patterns.

**How it is used**

- Structured features are scaled and passed to the MLP.
- Character IDs are embedded and convolved in parallel at three receptive fields.
- The pooled outputs are concatenated, then classified.

**Advantage**

- The architecture is intentionally complementary: the feature branch improves stability and interpretability, while the character branch restores information that handcrafted aggregates discard.

#### 2.2 Feature-only ablation network

The ablation models in `train_binary_feat_only.py` and `train_52class_feat_only.py` remove the CharCNN branch and keep only the structured feature MLP.

**Why this is used:** it isolates the contribution of the character branch.

**Advantage:** it provides an internal control experiment, so improvements can be attributed to architecture design rather than only data or optimization differences.

### 3. Dataset-Specific Preparation

#### 3.1 PhishPhresh / phreshphish cached chunks

`train_multiclass_brand.py` is the main dataset-building entrypoint for the URLPhishNet branch.

**What is used:** the script streams the dataset, extracts 67 features and `char_ids`, and saves compressed `.npz` chunks of 10,000 rows each under `phishphresh/multiclass_brand/data/`.

**Why it is used:** the dataset is large enough that repeat extraction from raw source would be slow and memory-heavy.

**How it is used:** once a split is fully extracted, a `_train_complete.flag` or `_test_complete.flag` file is written so later runs can skip reprocessing.

**Advantage:** resumable preprocessing, consistent cached features for all downstream experiments, and exact reproducibility of the train/test split across scripts.

#### 3.2 Sabir replication package data

The paper-comparison scripts load:

- `Leg_Training.csv`
- `Phish_Training.csv`
- `DomainAdversary.csv`
- `PathAdversary.csv`
- `TLDAdversary.csv`

**Why this is used:** it gives a direct comparison against Sabir et al. Table 7.

**How it is used:** the paper-data scripts extract the same 67 structured features and the same 256-character tensors used elsewhere.

**Operational details in code:** the legitimate training set is capped to 300,000 URLs for speed, path-adversary training is capped to 100,000 URLs, and path-adversary evaluation is capped to 50,000 URLs.

**Advantage:** the architecture can be compared on the same attack family without changing the feature or model stack.

### 4. Method-by-Method Methodology

#### 4.1 Binary benign/phishing prediction on the cached PhishPhresh data: `train_binary.py`

**What is used:** the full dual-stream URLPhishNet, using the 67 structured features plus the 256-character sequence.

**Why it is used:** collapsing the task to benign vs phishing measures how strong the fused URL representation is when the label space is simple and operationally relevant.

**How it is used:** all cached train and test chunks are loaded, any non-empty target is mapped to phishing, features are scaled, class imbalance is handled with inverse-frequency sampling and weighted cross-entropy, and the model is trained for 40 epochs with AdamW plus OneCycleLR.

**Advantage:** this is the cleanest operational baseline in the repository. It shows the raw binary detection strength of the full architecture before any adversarial hardening.

#### 4.2 Binary adversarially trained URLPhishNet on PhishPhresh: `train_binary_adv.py`

**What is used:** the same binary URLPhishNet as above, but its training set is augmented with Sabir adversarial URLs, all labeled as phishing.

**Why it is used:** the repository explicitly asks whether adversarial robustness can be improved by injecting the known attack distribution into training while keeping the original test split unchanged.

**How it is used:** the original PhishPhresh train split is preserved, adversarial URLs are feature-extracted and character-encoded, and then appended to the training data only. Domain and TLD adversaries are used fully, while path adversaries are capped at 100,000 for training.

**Advantage:** it directly measures the trade-off between standard accuracy and robustness against known lexical attacks.

#### 4.3 Multiclass brand prediction: `train_multiclass_brand.py`

**What is used:** the same dual-stream architecture, but trained for a 52-class problem.

**Why it is used:** beyond simply saying a URL is phishing, this method asks which brand is being impersonated. That is a harder but operationally richer task.

**How it is used:**

- The script keeps benign as class `0`.
- The top 50 most frequent phishing brands with at least 30 samples each get their own class.
- All remaining phishing brands are merged into a single `other_phishing` class.
- The model is trained with Focal Loss, effective-number class weights, a `1.5x` benign boost, and feature-branch mixup (`alpha=0.2`).

**Advantage:** it handles long-tail class imbalance explicitly and supports brand attribution instead of only binary flagging.

#### 4.4 Adversarial evaluation of the 52-class model: `run_52class_adversary.py`

**What is used:** the best saved 52-class URLPhishNet, the saved scaler, and the saved class map.

**Why it is used:** the 52-class model cannot be compared to Sabir's paper directly at the label level, so the script converts its output into a binary phishing/not-benign decision during attack evaluation.

**How it is used:** for each crafted URL, if the predicted class is anything other than benign, the prediction counts as phishing detection.

**Advantage:** it allows the multiclass model to be tested against the same adversarial URL corpora used by the binary models.

#### 4.5 Direct paper-comparison model on Sabir training data: `train_binary_paper_data.py`

**What is used:** the same binary URLPhishNet architecture, but trained on the Sabir replication package's normal training URLs.

**Why it is used:** this is the repository's direct apples-to-apples comparison against the paper's Table 7 normal and adversarial columns.

**How it is used:** `Leg_Training.csv` and `Phish_Training.csv` are combined, split 80/20 with stratification, scaled, and then used to train the binary URLPhishNet. The trained model is then evaluated on the three adversarial URL sets.

**Advantage:** it isolates architecture differences from dataset differences.

#### 4.6 Paper-style adversarial training on Sabir data: `train_binary_paper_data_adv.py`

**What is used:** the same paper-data binary URLPhishNet, but with the Sabir adversarial URL sets injected into the training data.

**Why it is used:** this is the repository's reproduction of the paper's adversarial-training setting, using the repository's architecture instead of the paper's traditional models.

**How it is used:** held-out normal test data remains untouched; adversarial URLs are only added to training.

**Advantage:** it measures whether the same adversarial-training protocol is enough to harden the stronger dual-stream model.

#### 4.7 Binary feature-only ablation: `train_binary_feat_only.py`

**What is used:** a residual MLP over the 67 structured features only, with no CharCNN branch.

**Why it is used:** the repository wants a direct answer to whether the character branch matters for benign/phishing prediction.

**How it is used:** the same cached PhishPhresh split is used, the same StandardScaler and weighted sampling logic are used, and the model is evaluated on the same three Sabir adversarial URL sets.

**Advantage:** this is the cleanest internal control for measuring the value added by character-level modeling.

#### 4.8 52-class feature-only ablation: `train_52class_feat_only.py`

**What is used:** a multiclass MLP on the same 67 features, with no character stream.

**Why it is used:** brand identification should be particularly sensitive to explicit character patterns in brand names and typosquatting variants.

**How it is used:** the class map from the main 52-class experiment is reused, so the label space is identical.

**Advantage:** it quantifies how much brand attribution depends on actual characters rather than only aggregate feature values.

#### 4.9 GAN-based feature-space attack on a standalone phishing detector: `train_binary_gan_adv.py`

**What is used:**

- A standalone feature-only phishing detector (`FeatureMLPPD`) with architecture `67 -> [256, 128, 64] -> 2`
- A generator with 3 hidden layers of 120 neurons each
- A discriminator with 2 hidden layers of 120 neurons each
- AlEroud-style 2-bit encoding, which converts each of the 67 features into two binary indicators, producing a 134-dimensional binary input space

**Why it is used:** the repository wants to replicate the AlEroud and Karabatis claim that feature-based phishing detectors can be fooled by adversarial feature-space perturbations.

**How it is used:**

- The feature-only PD is first trained on scaled continuous features.
- Phishing training samples are converted to 134-bit vectors using per-feature quartile and median thresholds.
- The generator receives a phishing binary vector plus 64-dimensional random noise and produces another 134-dimensional vector.
- The discriminator is trained to approximate the PD's behavior so gradients can flow to the generator.
- The adversarial binary outputs are thresholded, mapped back to continuous values using region midpoints, rescaled, and sent through the PD.

**Advantage:** it creates adversarial examples entirely in feature space and also produces a reusable adversarial training set for downstream defense experiments.

#### 4.10 Adversarially hardened feature-only detector: `train_binary_gan_hardened.py`

**What is used:** the same `67 -> [256, 128, 64] -> 2` feature-only detector as the GAN PD, but retrained on real training data plus the adversarial vectors saved by Exp 8.

**Why it is used:** this asks whether explicit adversarial retraining can close the very large evasion gap opened by the GAN.

**How it is used:** `binary_phishing_adv_gan/logs/adversarial_dataset.csv` is appended to the real training split, scaled jointly, and used to train a hardened detector.

**Advantage:** a direct defense experiment using attack-generated data already produced inside the same repository.

#### 4.11 Mahalanobis OOD defense on the feature-only detector: `train_mahalanobis_ood.py`

**What is used:** a feature-only detector with an exposed 64-dimensional penultimate layer, plus a Mahalanobis distance model fit on those penultimate representations.

**Why it is used:** adversarially training on known attacks is attack-specific. Mahalanobis OOD detection offers a different defense: reject inputs whose internal representations are too far from the training distribution.

**How it is used:**

- The feature-only classifier is trained first.
- Penultimate features are extracted for all training samples.
- Class means and a pooled covariance matrix are fit in that 64-dimensional space.
- The inverse covariance is used as a precision matrix.
- The OOD threshold is set to the 99th percentile of the training Mahalanobis scores.

**Advantage:** it does not require GAN examples during training and instead uses representation geometry as the defense signal.

#### 4.12 Cross-dataset validation of the OOD defense: `train_replication_gan_test.py`

**What is used:** the saved Exp 10 feature-only classifier, scaler, and Mahalanobis parameters, with no retraining.

**Why it is used:** the repository explicitly tests whether the OOD defense only works on the PhishPhresh distribution or whether it generalizes to a different dataset and attack generation regime.

**How it is used:**

- Part A: evaluate Exp 10 on real replication-package URLs
- Part B: train a replication-package GAN and evaluate its adversarial vectors with Exp 10
- Part C: evaluate the Sabir crafted URL sets with Exp 10

**Advantage:** it separates dataset transfer from attack transfer, which is important when claiming robustness.

#### 4.13 Exact-paper GAN loss reimplementation: `train_binary_gan_exact_paper.py`

**What is used:** the same PD/G/D architecture as Exp 8, but the GAN loss is rewritten to follow the exact equations described in the AlEroud paper.

**Why it is used:** this script is a control experiment intended to test whether the OOD result depends on the exact GAN loss interpretation.

**How it is used:** the discriminator is treated as estimating `P(legitimate | f)` and the logged rewards are the paper-style negative reward curves rather than only standard positive losses.

**Advantage:** it decouples "the GAN attack exists" from "the exact loss-sign convention used in the paper."

#### 4.14 Dual-stream Mahalanobis OOD script: `train_charcnn_mahalanobis_ood.py`

**What is used:** the full dual-stream binary URLPhishNet, but modified so the 256-dimensional fusion-layer penultimate representation can be extracted and used for Mahalanobis OOD scoring.

**Why it is used:** Exp 10's OOD defense is attached to the feature-only detector. This script attempts to move the same defense to the stronger dual-stream model.

**How it is used:** the script trains a binary dual-stream model, fits class-conditional Gaussians in the 256-dimensional fusion space, calibrates a 99th-percentile threshold, and defines a fallback for feature-space GAN vectors by sending all-zero character tensors to the CharCNN branch.

**Advantage:** in principle, it aims to combine near-URLPhishNet normal accuracy with OOD-style rejection.

#### 4.15 Permutation feature-importance analysis: `analyze_feature_importance.py`

**What is used:** the dual-stream model expected from Exp 13, wrapped as an `sklearn` estimator so `sklearn.inspection.permutation_importance` can be applied.

**Why it is used:** this experiment asks which of the 67 structured features still matter after the character branch and feature branch are fused together.

**How it is used:** one structured feature column is shuffled at a time across a capped 20,000-sample test subset while the character tensor is held fixed; the metric of interest is macro F1 drop.

**Advantage:** it measures contribution, not just correlation. A feature is important here only if disturbing it harms the trained fused model.

#### 4.16 Row-level prediction export utility: `generate_predictions.py`

**What is used:** the saved best 52-class URLPhishNet model, scaler, and class map.

**Why it is used:** the main training scripts report aggregate metrics, but investigations often need per-row predictions and brand labels.

**How it is used:** the script reconstructs raw URLs from the saved `char_ids`, rebuilds the same 80/20 split, runs inference, and saves a CSV containing actual brand, predicted brand, actual phishing flag, predicted phishing flag, and correctness columns.

**Advantage:** it makes the model auditable at instance level without changing the training pipeline.

#### 4.17 Other URL-centric files that exist but are not part of the main experiment chain

- `train_phreshphish_url.py` trains the older `main.py.PhishingNet` on the phish dataset using only the 51 hand-crafted features and a streaming chunking strategy. It is methodologically relevant as an earlier baseline and as evidence for the chunking approach, but there is no matching checked-in result directory in this checkout.
- `README.md`, `technical_document.md`, `train_main.py`, and `test_main.py` describe or support a separate hybrid challenge pipeline. `main.py` is still important here because its `FeatureExtractor` is reused, but the end-to-end challenge workflow is not the same as the URLPhishNet experiment line.

## Results

### 1. Summary of Core Classification Methods

| Method | Task | Main recorded result | Adversarial result |
|---|---|---:|---:|
| `train_binary.py` | Benign vs phishing on cached PhishPhresh | 98.68% accuracy, 0.9867 macro F1, 1.11% FPR, 1.57% FNR | Domain 48.23%, Path 23.78%, TLD 43.02% detection |
| `train_binary_adv.py` | Same, but adversarially trained | 98.58% accuracy, 0.9856 macro F1, 1.08% FPR, 1.85% FNR | Domain 100.00%, Path 99.92%, TLD 100.00% detection |
| `train_multiclass_brand.py` + `run_52class_adversary.py` | 52-class benign/brand/other-phishing | 77.64% accuracy, 0.5368 macro F1 | Domain 27.06%, Path 36.32%, TLD 27.12% detection |
| `train_binary_paper_data.py` | Binary direct comparison on Sabir data | 99.18% accuracy, MCC 0.978, FNR 1.42% | Domain 27.48%, Path 100.00%, TLD 50.46%, average 59.31% |
| `train_binary_paper_data_adv.py` | Paper-style adversarial training on Sabir data | 99.25% accuracy, MCC 0.980, FNR 1.62% | Domain 100.00%, Path 100.00%, TLD 100.00% |
| `train_binary_feat_only.py` | Binary ablation, 67 features only | 96.22% accuracy, 0.9618 macro F1, 3.31% FPR, 4.36% FNR | Domain 92.69%, Path 1.77%, TLD 70.37% detection |
| `train_52class_feat_only.py` | 52-class ablation, 67 features only | 38.20% accuracy, 0.3280 macro F1, 58.21% FPR, 0.63% FNR | No separate adversarial report saved |

### 2. Detailed Results by Method

#### 2.1 Binary URLPhishNet on PhishPhresh

Source: `binary_phishing/logs/adversarial_results.txt`, `binary_phishing/logs/training_log.csv`

- Best epoch: 39
- Normal test accuracy: 98.68%
- Macro F1: 0.9867
- MCC: 0.9734
- FPR: 1.11%
- FNR: 1.57%

Adversarial evaluation:

- Domain adversary: 48.23% detection, 51.77% FNR
- Path adversary: 23.78% detection, 76.22% FNR
- TLD adversary: 43.02% detection, 56.98% FNR

Interpretation grounded in the artifacts: the full dual-stream model is very strong on normal data but is still vulnerable to the Sabir crafted URL attacks, especially the path attack.

#### 2.2 Adversarially trained binary URLPhishNet on PhishPhresh

Source: `binary_phishing_adv/logs/adversarial_results.txt`, `binary_phishing_adv/logs/training_log.csv`

- Best epoch: 13
- Normal test accuracy: 98.58%
- Macro F1: 0.9856
- MCC: 0.9712
- FPR: 1.08%
- FNR: 1.85%

Adversarial evaluation:

- Domain adversary: 100.00% detection, 0.00% FNR
- Path adversary: 99.92% detection, 0.08% FNR
- TLD adversary: 100.00% detection, 0.00% FNR

The saved report explicitly frames this as a robustness trade-off: a very small drop in normal accuracy buys a very large gain in robustness to the injected attack families.

#### 2.3 52-class URLPhishNet

Source: `phishphresh/multiclass_brand/logs/final_report.txt`, `phishphresh/multiclass_brand/logs/adversarial_results.txt`

- Best epoch: 39
- Test accuracy: 77.64%
- Macro F1: 0.5368
- Number of classes: 52

The saved classification report shows that performance is uneven across brands, which is expected in a long-tail multiclass problem. Benign and `other` are strong, many high-frequency brands are solid, and several low-support or easily confusable brands are weak.

Adversarial evaluation:

- Domain adversary: 27.06% detection, 72.94% FNR
- Path adversary: 36.32% detection, 63.68% FNR
- TLD adversary: 27.12% detection, 72.88% FNR

This means the multiclass model is useful for brand attribution on normal data, but on these adversarial URL sets its binary phishing/not-benign behavior degrades heavily.

#### 2.4 Direct paper comparison on Sabir data

Source: `binary_paper_data/logs/comparison_vs_table7.txt`, `binary_paper_data/logs/training_log.csv`

- Best epoch: 20
- Normal accuracy: 99.18%
- MCC: 0.978
- FNR: 1.42%

The saved comparison file places this above the paper's listed best normal-accuracy models, including:

- Basic Lexical+Ext + XGB: 98.58%
- EXPOSE: 98.46%
- Char n-gram + LGBM: 98.31%
- URLNET: 98.27%

Adversarial evaluation recorded in the saved report:

- Domain adversary: 27.48% detection, 72.52% FNR
- Path adversary: 100.00% detection, 0.00% FNR
- TLD adversary: 50.46% detection, 49.54% FNR
- Average adversarial accuracy in the saved file: 59.31%

This should be read exactly as stored: the repository records a very strong normal-data result and a mixed adversarial result, with path-adversary detection recorded as perfect on the evaluated subset.

#### 2.5 Paper-style adversarial training on Sabir data

Source: `binary_paper_data_adv/logs/comparison_vs_table7_adv.txt`, `binary_paper_data_adv/logs/training_log.csv`

- Best epoch: 19
- Normal accuracy: 99.25%
- MCC: 0.980
- FNR: 1.62%

Saved adversarial evaluation:

- Domain adversary: 100.00% detection
- Path adversary: 100.00% detection
- TLD adversary: 100.00% detection

The report presents this as outperforming the paper's adversarially trained traditional models on the evaluated subsets.

#### 2.6 Binary feature-only ablation

Source: `binary_feat_only/logs/adversarial_results.txt`

- Best epoch: 30
- Normal accuracy: 96.22%
- Macro F1: 0.9618
- MCC: 0.9236
- FPR: 3.31%
- FNR: 4.36%

Adversarial evaluation:

- Domain adversary: 92.69% detection
- Path adversary: 1.77% detection
- TLD adversary: 70.37% detection

This is one of the clearest architecture findings in the repository. Relative to the full binary URLPhishNet, removing the CharCNN branch lowers normal accuracy and makes the path adversary much more damaging.

#### 2.7 52-class feature-only ablation

Source: `multiclass_feat_only/logs/results.txt`

- Best epoch: 28
- Accuracy: 38.20%
- Macro F1: 0.3280
- Weighted F1: 0.4615
- MCC: 0.3151
- FPR: 58.21%
- FNR: 0.63%

Compared with the full 52-class URLPhishNet, this large performance drop is strong repository-internal evidence that the character stream is central to multiclass brand attribution.

### 3. Summary of GAN and Defense Methods

| Method | Main recorded result |
|---|---|
| `train_binary_gan_adv.py` | Feature-only PD accuracy 95.52%; GAN adversarial detection 0.00%; evasion 100.00%; 238,729 adversarial vectors generated |
| `train_binary_gan_hardened.py` | Normal accuracy 94.34%; GAN evasion reduced from 100.00% to 0.00% |
| `train_mahalanobis_ood.py` | Normal accuracy 95.47%; 1.04% real URLs flagged as OOD; 100.00% of Exp 8 GAN vectors flagged as OOD |
| `train_replication_gan_test.py` | Cross-dataset real URL accuracy 57.32%; replication-package GAN vectors 100.00% OOD detected; Sabir path adversary 0.00% OOD flagged |
| `train_binary_gan_exact_paper.py` | Exact-paper-loss run records 0.0% PD evasion and 100.0% OOD detection in the checked-in report |
| `train_charcnn_mahalanobis_ood.py` | Script exists, but no matching result directory is committed in this checkout |
| `analyze_feature_importance.py` | Top saved importance features are path/query/path entropy and TLD-related structural features |

### 4. Detailed GAN and Defense Results

#### 4.1 GAN feature-space attack on the feature-only PD

Source: `binary_phishing_adv_gan/logs/pd_evasion_report.txt`, `binary_phishing_adv_gan/logs/gan_training_loss.csv`

- PD architecture: `67 -> [256, 128, 64] -> 2`
- PD best accuracy: 95.52%
- Original phishing detection rate on the test set: 94.40%
- Adversarial detection rate after GAN generation: 0.00%
- Evasion rate: 100.00%
- Adversarial vectors generated: 238,729

The GAN training CSV shows a fast collapse of generator loss toward zero:

- Epoch 1 generator loss: 0.207406
- Epoch 2 generator loss: 0.000028
- By epoch 3 onward the saved CSV records generator loss as `0.0`

Within the repository, this is the strongest evidence that a feature-only URL detector can be completely bypassed in the implemented feature-space threat model.

#### 4.2 Hardened feature-only detector trained on GAN outputs

Source: `binary_gan_hardened/logs/stdout_exact_paper.log`, `binary_gan_hardened/logs/training_metrics.csv`

Normal test results:

- Accuracy: 94.34%
- Macro F1: 0.9422
- MCC: 0.8880
- FPR: 1.23%
- FNR: 11.11%

GAN evaluation:

- Adversarial vectors tested: 238,729
- Correctly caught as phishing: 238,729 (100.00%)
- Still evading: 0 (0.00%)

This saved run shows the direct cost-benefit of adversarial retraining in this repository: lower normal accuracy than the unhardened PD, but complete removal of the recorded Exp 8 evasion.

#### 4.3 Mahalanobis OOD defense on the feature-only detector

Source: `binary_mahalanobis_ood/logs/ood_report.txt`, `binary_mahalanobis_ood/logs/training_metrics.csv`

Model and threshold:

- Architecture: `67 -> [256, 128, 64] -> 2`
- Penultimate dimension: 64
- OOD threshold: 446.5667 (99th percentile of training scores)

Real test-set behavior:

- Test samples: 133,263
- Standard accuracy: 95.47%
- Standard FNR: 6.01%
- OOD-flagged real URLs: 1,382 (1.04%)
- Of flagged real URLs, 71.4% are phishing

GAN behavior:

- Adversarial count: 238,729
- Classifier evasion without OOD: 100.0%
- OOD detected: 100.0%
- Adversarial-score median: 6,924,816.2213
- Training-score median: 38.7236

This is a very specific repository conclusion: the Mahalanobis detector is extremely strong against the implemented GAN feature-space attack, even though the classifier alone is completely fooled.

#### 4.4 Cross-dataset validation of the Mahalanobis defense

Source: `binary_replication_gan_test/logs/eval_A_real_urls.txt`, `eval_B_gan_vectors.txt`, `eval_C_sabir_adversarial.txt`

Part A, real replication-package URLs:

- Samples: 39,339
- Accuracy: 57.32%
- FNR: 34.27%
- FPR: 50.81%
- OOD flagged: 2,648 (6.73%)
- Of those OOD flags, 96.8% are phishing

Part B, replication-package GAN vectors:

- Adversarial vectors: 77,354
- Evasion without OOD: 100.0%
- OOD detected: 100.0%
- Adversarial-score median: 118,653.91
- Saved threshold in this file: 513.1637

Part C, Sabir adversarial URLs:

- Domain adversary: 92.39% detection, 7.61% FNR, 0.00% OOD flagged
- Path adversary: 0.56% detection, 99.44% FNR, 0.00% OOD flagged
- TLD adversary: 67.88% detection, 32.12% FNR, 3.44% OOD flagged

The repository evidence here is nuanced. The OOD defense generalizes very strongly to GAN-generated feature-space attacks across datasets, but it does not act as a universal detector for Sabir's string-level URL mutations, especially the path adversary.

#### 4.5 Exact-paper GAN-loss experiment

Source: `binary_gan_exact_paper/logs/pd_evasion_report.txt`, `binary_gan_exact_paper/logs/gan_training_reward.csv`

Saved GAN training summary:

- Final generator paper-style reward at epoch 100: `-0.0000`
- Final discriminator paper-style reward at epoch 100: `-0.0642`

Saved outcome:

- Adversarial vectors: 238,729
- Evasion rate against the PD: 0.0%
- OOD detected by the Exp 10 defense: 238,729 / 238,729 (100.0%)
- Adversarial-score median: 61,962.05
- OOD threshold: 446.5667
- Score/threshold ratio: 139x above

This file should be reported exactly as stored. It differs materially from Exp 8 because the checked-in Exp 12 artifact records zero evasion, not high evasion.

#### 4.6 Dual-stream Mahalanobis OOD script

Source: `train_charcnn_mahalanobis_ood.py`

The repository contains a complete methodology script for attaching Mahalanobis OOD detection to the full dual-stream binary model, including:

- training on the same cached PhishPhresh split,
- fitting Gaussian class statistics in a 256-dimensional fusion representation,
- and handling feature-space GAN vectors by passing zero character tensors to the CharCNN branch.

However, there is no committed `binary_charcnn_mahalanobis_ood/` result directory in this checkout, so no numeric result from this method should be claimed here.

#### 4.7 Permutation feature importance on the dual-stream model

Source: `feature_importance/logs/feature_importance_report.txt`

Top recorded features by macro-F1 drop when shuffled:

1. `path_depth` - 0.0114
2. `shannon_entropy_query` - 0.0080
3. `shannon_entropy_path` - 0.0045
4. `is_common_tld` - 0.0032
5. `query_length_normalized` - 0.0023

Per-group average F1-drop:

- Structural / URL-level: 0.0019
- TLD signals: 0.0010
- Basic domain structure: 0.0002
- Entropy and complexity: 0.0001
- CSE pattern / brand: 0.0001
- Lexical patterns: approximately 0

The saved artifact therefore says that, in the fused model, path/query structure and TLD-related signals are the strongest structured-feature contributors among the 67 engineered inputs.

## Artifact Integrity Notes

The repository contains a few important evidence-boundary issues that should be stated explicitly in a professional report:

1. `analyze_feature_importance.py` expects model artifacts under `binary_charcnn_mahalanobis_ood/`, but that output directory is not committed in this checkout. The saved feature-importance report exists, but the underlying Exp 13 artifacts are not auditable here.
2. `binary_gan_exact_paper/logs/pd_evasion_report.txt` records `0.0%` evasion, which differs from Exp 8's `100.0%` evasion and should not be merged with it.
3. `binary_replication_gan_test/logs/eval_B_gan_vectors.txt` records an OOD threshold of `513.1637`, while `binary_mahalanobis_ood/logs/ood_report.txt` records `446.5667`. Since both artifacts are committed, the safest approach is to report each file's saved value as-is.
4. The feature names hardcoded in `analyze_feature_importance.py` are script-defined labels for the 67 feature positions. They should be treated as the labels used in that saved experiment, not as a verbatim dump of the `main.py` feature-key names.

## Final Method-Level Conclusions Grounded in the Repository

1. The strongest normal-data URL classifier in the repository is the dual-stream URLPhishNet, not the feature-only baseline.
2. The CharCNN branch is materially important. It improves binary accuracy and is even more important for multiclass brand identification.
3. The Sabir-style string-space attacks remain effective against non-adversarially trained models, especially on the path adversary.
4. Adversarial training with the Sabir attack sets is highly effective in the repository's recorded runs, both on the PhishPhresh split and on the Sabir training data split.
5. A standalone feature-only detector is completely vulnerable to the implemented AlEroud-style GAN attack in Exp 8.
6. Two different defenses are implemented against that GAN threat: explicit adversarial retraining and Mahalanobis OOD rejection. Both are highly effective against the repository's saved Exp 8 GAN vectors.
7. The OOD defense generalizes strongly to GAN-generated feature-space attacks across datasets, but it does not serve as a universal detector for all crafted URL mutations, especially Sabir's path attack.



\section{Results}

\subsection{Evaluation on the PhreshPhish Dataset}

We evaluated three components of the pipeline on the PhreshPhish dataset: the URL-only neural classifier (Layer~2), a GAN adversarial robustness evaluation with Mahalanobis OOD defense on Layer~2, and the DOM/HTML+URL fusion classifier (Layer~3). These experiments validate both the effectiveness of each individual modality and the robustness of the domain classifier against feature-space adversarial attacks.

\subsubsection{URL-Only Neural Network (Layer 2)}

The URL-only neural classifier was trained on 533,052 samples and evaluated on a held-out test set of 133,263 samples from the PhreshPhish dataset. The network architecture maps a 51-dimensional domain feature vector through fully connected layers ($51 \to 128 \to 64 \to 32 \to 16 \to 2$) with Batch Normalization, ReLU activations, and Dropout(0.3). Training ran for 100 epochs using Adam optimization with class-imbalance-aware \texttt{WeightedRandomSampler}.

Table~\ref{tab:url-nn-phreshphish} reports the final evaluation metrics at the best checkpoint (epoch~95).

\begin{table}[t]
\centering
\caption{URL-only neural network performance on PhreshPhish (test support: 133,263).}
\label{tab:url-nn-phreshphish}
\begin{tabular}{lcccc}
\toprule
Class & Precision & Recall & F1 & Support \\
\midrule
Benign/Suspected & 0.94 & 0.94 & 0.94 & 73,598 \\
Phishing         & 0.93 & 0.93 & 0.93 & 59,665 \\
\midrule
\textbf{Overall} & \textbf{0.94} & \textbf{0.94} & \textbf{0.94} & 133,263 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Confusion matrix — URL-only neural network on PhreshPhish.}
\label{tab:url-nn-cm}
\begin{tabular}{lcc}
\toprule
 & Predicted Benign & Predicted Phishing \\
\midrule
Actual Benign   & 69,406 & 4,192 \\
Actual Phishing & 4,227  & 55,438 \\
\bottomrule
\end{tabular}
\end{table}

The model achieved a test accuracy of \textbf{93.68\%} and a ROC-AUC of \textbf{0.9846} at the best checkpoint. Both classes attained balanced precision, recall, and F1 scores of 0.93--0.94, with 4,192 benign false positives and 4,227 phishing false negatives out of 133,263 test samples, indicating well-calibrated detection with no strong class bias.

\subsubsection{GAN Adversarial Attack and Mahalanobis OOD Detection}

We replicated the GAN-based feature-space attack of AlEroud and Karabatis~\cite{aleroud2020} on PhreshPhish. The full dual-stream classifier (CharCNN + feature MLP, $98.68\%$ standard accuracy) was used as the attack target. A standalone feature MLP was first trained as the discriminator proxy, then the GAN generator was trained for 100 epochs to produce adversarial feature vectors.

Table~\ref{tab:gan-evasion} shows how each configuration handles GAN-generated adversarial samples. Without any defense, the GAN evades the feature-based classifier entirely. Adding the Mahalanobis OOD layer as a post-classification screen recovers full detection with no GAN samples seen during training.

\begin{table}[t]
\centering
\caption{GAN evasion and Mahalanobis OOD detection results on 238,729 adversarial vectors (PhreshPhish). OOD FPR is the fraction of real URLs incorrectly flagged as adversarial.}
\label{tab:gan-evasion}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrr}
\toprule
\textbf{Configuration} & \textbf{Normal Acc.} & \textbf{GAN Evasion} & \textbf{OOD FPR} \\
\midrule
Feature MLP, no defense         & 95.44\% & \textbf{100.0\%} & ---    \\
Feature MLP + adv.\ training    & 94.34\% & 0.0\%            & 1.23\% \\
Feature MLP + Mahalanobis OOD   & 95.45\% & 0.0\%            & 1.04\% \\
CharCNN + MLP + Mahalanobis OOD & \textbf{98.68\%} & 0.0\%   & 3.60\% \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:ood-scores} shows the Mahalanobis score distributions for real URLs versus GAN-generated vectors. The separation is extreme: GAN vectors all cluster at a single score of 2,150 because the binary encoding collapses all adversarial outputs to the same few values, while real URLs score around 164 at the median. The OOD threshold (99th percentile of training scores) sits at 1,222.

\begin{table}[t]
\centering
\caption{Mahalanobis score distributions for real URLs and GAN adversarial vectors. CharCNN\,+\,MLP model; flagging threshold\,=\,1{,}222 (99th percentile of training scores).}
\label{tab:ood-scores}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrr}
\toprule
\textbf{Input Type} & \textbf{Median Score} & \textbf{Flagged as OOD} \\
\midrule
Real training URLs      & 163.7     & ---                    \\
Real test URLs          & 164.0     & 3.60\%                 \\
GAN adversarial vectors & \textbf{2{,}150.1} & \textbf{100.0\%} \\
\bottomrule
\end{tabular}
\end{table}

The result confirms that the OOD layer is a practical, training-free GAN defense. The key reason it works is structural: the GAN encodes features into a 2-bit binary representation before generating adversarial outputs, which forces every output to land at one of just three fixed midpoint values per feature. Real URLs never follow this pattern, so the anomaly stands out clearly in the classifier's internal representation space. The OOD FPR of 3.60\% on real URLs is a conservative estimate given the 99th-percentile threshold; tuning to the 97th or 98th percentile would reduce this further with only marginal impact on GAN detection.

\subsubsection{DOM Tree / HTML+URL Fusion Classifier (Layer 3)}

The HTML+URL fusion classifier was trained on the PhreshPhish dataset using a multi-branch neural network combining 22 HTML structural features, 19 URL lexical-statistical features, and 1,500 TF-IDF text features (total: 1,541 features). Training used 498,255 samples over 20 epochs; the test set comprised 168,060 samples (91,260 benign, 76,800 phishing).

Table~\ref{tab:dom-phreshphish} summarizes per-class performance on the test split.

\begin{table}[t]
\centering
\caption{DOM/HTML+URL fusion classifier performance on PhreshPhish (test support: 168,060).}
\label{tab:dom-phreshphish}
\begin{tabular}{lcccc}
\toprule
Class & Precision & Recall & F1 & Support \\
\midrule
Benign   & 0.93 & 0.99 & 0.96 & 91,260 \\
Phishing & 0.99 & 0.92 & 0.95 & 76,800 \\
\midrule
\textbf{Overall} & \textbf{0.96} & \textbf{0.96} & \textbf{0.96} & 168,060 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Confusion matrix — DOM/HTML+URL fusion classifier on PhreshPhish.}
\label{tab:dom-phreshphish-cm}
\begin{tabular}{lcc}
\toprule
 & Predicted Benign & Predicted Phishing \\
\midrule
Actual Benign   & 90,399 & 861   \\
Actual Phishing & 6,336  & 70,464 \\
\bottomrule
\end{tabular}
\end{table}

The fusion model achieved \textbf{95.72\%} best training accuracy and \textbf{96\%} test accuracy with a macro-averaged ROC-AUC of \textbf{0.9938} (epoch 20). Notably, phishing precision reached 0.99 with only 861 false positives on the benign class, while phishing recall of 0.92 (6,336 false negatives) reflects a conservative threshold that avoids over-flagging. The combination of HTML structural, URL lexical, and TF-IDF textual signals yields a substantially stronger and more balanced detector than either URL-only or DOM-only baselines.

\subsection{DOM Model Comparison}

\subsubsection{Evaluation Protocol}
All three models are evaluated on the same held-out DOM feature store generated from the test split (total support: 90,112 samples; 48,851 benign and 41,261 phishing), ensuring direct comparability. We report class-wise precision/recall/F1 together with accuracy and ROC-AUC, followed by confusion-matrix and aggregate error-profile metrics.

\begin{table*}[t]
\centering
\caption{DOM-level comparison prior to visual benchmarking (test support: 90,112).}
\label{tab:dom-pre-visual-main}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lcccccccc}
\toprule
\multirow{2}{*}{Model} & \multirow{2}{*}{Accuracy} & \multirow{2}{*}{ROC-AUC} & \multicolumn{3}{c}{Benign Class} & \multicolumn{3}{c}{Phishing Class} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9}
 &  &  & Precision & Recall & F1 & Precision & Recall & F1 \\
\midrule
Random Forest    & \textbf{0.8187} & \textbf{0.8897} & 0.8060 & \textbf{0.8767} & \textbf{0.8399} & \textbf{0.8371} & 0.7501 & \textbf{0.7912} \\
Extra Trees      & 0.8155 & 0.8797 & \textbf{0.8058} & 0.8693 & 0.8363 & 0.8293 & \textbf{0.7519} & 0.7887 \\
Isolation Forest & 0.4736 & 0.1861 & 0.5086 & 0.8548 & 0.6378 & 0.1141 & 0.0221 & 0.0371 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}


\begin{table*}[t]
\centering
\caption{DOM-level detailed error profile before visual stage.}
\label{tab:dom-pre-visual-errors}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lcccccccc}
\toprule
Model & TN & FP & FN & TP & Specificity & Sensitivity & Macro-F1 & Weighted-F1 \\
\midrule
Random Forest    & 42,828 & 6,023 & 10,310 & 30,951 & \textbf{0.8767} & 0.7501 & \textbf{0.8155} & \textbf{0.8176} \\
Extra Trees      & 42,465 & 6,386 & 10,236 & 31,025 & 0.8693 & \textbf{0.7519} & 0.8125 & 0.8145 \\
Isolation Forest & 41,760 & 7,091 & 40,348 & 913    & 0.8548 & 0.0221 & 0.3374 & 0.3627 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}


\subsection{Visual Benchmarking}
The final dataset size was 5,208 screenshots, comprising 3,000 phishing and 2,208 benign samples.

We used a stratified 80:20 train-test split with fixed random seed (\texttt{random\_state=42}), yielding 4,166 training and 1,042 test samples. All models were evaluated on the same split for fair comparison.

Feature extraction used DINOv2 embeddings with pooled patch statistics, and classification was benchmarked using Random Forest, Extra Trees, Logistic Regression, and Gradient Boosting.

\subsection{Model Comparison}
Table~\ref{tab:visual-benchmark-main} summarizes primary model metrics. Extra Trees delivered the highest ROC-AUC, while Random Forest and Extra Trees tied for highest accuracy. Gradient Boosting produced the highest phishing recall, while sacrificing benign precision due to higher false positives.

\begin{table*}[t]
\centering
\caption{Supervised visual benchmark on 5,208 screenshots (test set: 1,042).}
\label{tab:visual-benchmark-main}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lcccccccc}
\toprule
\multirow{2}{*}{Model} & \multirow{2}{*}{Accuracy} & \multirow{2}{*}{ROC-AUC} & \multicolumn{3}{c}{Benign Class} & \multicolumn{3}{c}{Phishing Class} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9}
 &  &  & Precision & Recall & F1 & Precision & Recall & F1 \\
\midrule
Random Forest       & \textbf{0.8695} & 0.9545 & 0.8072 & 0.9095 & 0.8553 & 0.9265 & 0.8400 & 0.8811 \\
Extra Trees         & \textbf{0.8695} & \textbf{0.9569} & 0.7965 & \textbf{0.9299} & \textbf{0.8580} & \textbf{0.9411} & 0.8250 & 0.8792 \\
Logistic Regression & 0.8637 & 0.9400 & 0.8247 & 0.8620 & 0.8429 & 0.8948 & 0.8650 & 0.8797 \\
Gradient Boosting   & 0.8647 & 0.9433 & \textbf{0.8753} & 0.7941 & 0.8327 & 0.8580 & \textbf{0.9167} & \textbf{0.8864} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}


\begin{table*}[t]
\centering
\caption{Detailed error profile and aggregate metrics on the test split.}
\label{tab:visual-benchmark-detail}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lcccccccc}
\toprule
Model & TN & FP & FN & TP & Specificity & Sensitivity & Macro-F1 & Weighted-F1 \\
\midrule
Random Forest       & 402 & 40 & 96  & 504 & 0.9095 & 0.8400 & 0.8682 & 0.8702 \\
Extra Trees         & 411 & 31 & 105 & 495 & \textbf{0.9299} & 0.8250 & \textbf{0.8686} & \textbf{0.8702} \\
Logistic Regression & 381 & 61 & 81  & 519 & 0.8620 & 0.8650 & 0.8613 & 0.8641 \\
Gradient Boosting   & 351 & 91 & 50  & 550 & 0.7941 & \textbf{0.9167} & 0.8596 & 0.8636 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}