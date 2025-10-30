Hybrid Phishing Detector - Docker Quickstart

Files in this folder:
- main.py: hybrid detector (combined train + test orchestrator)
- train_main.py: train-only entrypoint
- test_main.py: test-only entrypoint
- Dockerfile: container build recipe
- docker-compose.yml: one-command runner
- requirements.txt: Python dependencies

Prerequisites:
- Docker Desktop installed and running

How to run:
1) Build and run with compose (recommended):
   docker compose up --build

   What this does:
   - Runs training: `python train_main.py` with DATASET_PATH
   - Then runs testing: `python test_main.py` with SHORTLIST_DIR
   - Mounts the current project at /app so local datasets are visible

   Compose environment variables:
   - DATASET_PATH=/app/backend/dataset/combined_dataset.csv
   - SHORTLIST_DIR=/app/backend/dataset/PS-02_Shortlisting_set

Run services individually with compose:
   # Train only
   docker compose run --build --rm train

   # Test only (requires trained model at submission/hybrid_model.pkl)
   docker compose run --rm test

2) Alternatively, plain Docker:
   docker build -t hybrid-detector .

   # Train only
   docker run --rm -v "%cd%":/app -e DATASET_PATH=/app/backend/dataset/combined_dataset.csv hybrid-detector python train_main.py

   # Test only (after model exists)
   docker run --rm -v "%cd%":/app -e SHORTLIST_DIR=/app/backend/dataset/PS-02_Shortlisting_set hybrid-detector python test_main.py

Outputs:
- Trained model saved to `submission/hybrid_model.pkl` and `submission/hybrid_weights.pth`
- Submission files created under `PS-02_AIGR-123456_Submission` in the mounted project directory.

