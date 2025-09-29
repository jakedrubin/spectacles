# Spectacles project — preprocessing & recommendation helpers

This repository contains notebooks and helper scripts to build a face-to-frame recommendation pipeline.

Recommended GitHub layout:

- Keep the notebooks (`*.ipynb`) in the repo — they are the primary analysis artifacts. Commits should not include `data/`, `models/`, or large outputs — these are ignored by `.gitignore`.

New helper scripts:

- `src/preprocess.py` — merge datasets, one-hot encode frame features, save `data/processed_data.csv`, `data/X_columns.json`, `data/frame_catalog.csv`, and `data/scaler.joblib`.
- `src/recommend.py` — load a saved regression model (joblib) and recommend top-K frames for a new face JSON input.

How to use

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run preprocessing (requires the CSVs in repository root):

```bash
python src/preprocess.py
```

3. Train and save your regressor at `models/regressor.joblib` (not provided here). Then call recommendation:

```bash
python src/train.py     # trains and saves models/regressor.joblib
python src/recommend.py --face-json examples/new_face.json --model models/regressor.joblib --top-k 5
```

Secrets
- `old_specs.ipynb` previously contained a hard-coded Clarifai API key; that notebook has been updated to read such keys from the environment. Never commit API keys — store them as environment variables or in a protected secrets manager.

Next steps
- Optionally add a training script that trains a regressor and saves `models/regressor.joblib`.
- Add unit tests for landmark parsing and column alignment.

Files added by this cleanup: `src/preprocess.py`, `src/recommend.py`, `src/train.py`, `src/tryon.py`, `src/s.py`, `src/scraper.py`, `requirements.txt`, `.gitignore`.
