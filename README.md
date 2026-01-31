# Spectacles — Face-to-Frame Recommendation System

A face-to-frame recommendation pipeline that analyzes facial features and recommends eyeglass frames that complement your face shape.

## Features

- **Face Analysis**: Extract facial metrics from photos using MediaPipe Face Mesh
- **Frame Recommendation**: ML-powered recommendations based on facial features
- **Web Demo**: Interactive Gradio interface for easy testing

## Project Structure

```
├── demo.py                 # Gradio web interface
├── src/
│   ├── face_analysis.py    # MediaPipe facial feature extraction
│   ├── preprocess.py       # Dataset preprocessing and encoding
│   ├── recommend.py        # Frame recommendation engine
│   ├── train.py            # Model training script
│   ├── tryon.py            # Virtual try-on utilities
│   └── scraper.py          # Data scraping utilities
├── notebooks/              # Analysis notebooks
├── data/                   # Processed datasets (gitignored)
├── models/                 # Trained models (gitignored)
└── examples/               # Example inputs
```

## Setup

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate spectacles-demo
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

### Dependencies

- **Core**: numpy, pandas, scikit-learn, joblib
- **Face Analysis**: mediapipe, opencv-python, pillow
- **Web Demo**: gradio
- **Scraping** (optional): requests, beautifulsoup4, lxml, selenium, playwright

## Quick Start

### 1. Preprocess Data

```bash
python src/preprocess.py
```

This creates:
- `data/processed_data.csv` — merged and encoded dataset
- `data/X_columns.json` — feature column names
- `data/frame_catalog.csv` — frame metadata
- `data/scaler.joblib` — fitted scaler

### 2. Train the Model

```bash
python src/train.py
```

Saves the trained model to `models/regressor.joblib`.

### 3. Run the Web Demo

```bash
python demo.py
```

Opens a Gradio interface at http://127.0.0.1:7860 where you can:
- Upload a face photo
- View extracted facial features
- Get top 5 frame recommendations

### 4. Command-Line Recommendation

```bash
python src/recommend.py --face-json examples/new_face.json --model models/regressor.joblib --top-k 5
```

### 5. Analyze a Face Image

```bash
python src/face_analysis.py path/to/face.jpg
```

## Secrets

API keys should never be committed. Store them as environment variables or in a secrets manager.

## Notes

- Notebooks (`*.ipynb`) are kept in the repo as primary analysis artifacts
- `data/`, `models/`, and large outputs are gitignored
