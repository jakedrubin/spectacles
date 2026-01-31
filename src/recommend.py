"""recommend.py
Load a trained regressor and recommend top-k frames for a given face.
This expects artifacts produced by train.py (data/X_columns.json, scaler.joblib)
and a saved regression model at models/regressor.joblib.
"""
import json
from pathlib import Path
import pandas as pd
import joblib
import argparse


def load_artifacts(root: Path):
    data_dir = root / 'data'
    xcols = json.loads((data_dir / 'X_columns.json').read_text(encoding='utf-8'))
    scaler_path = data_dir / 'scaler.joblib'
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    frame_catalog = pd.read_csv(data_dir / 'frame_catalog.csv') if (data_dir / 'frame_catalog.csv').exists() else None
    return xcols, scaler, frame_catalog


def create_interaction_features(df):
    """
    Create interaction features between face and frame properties.
    Must match exactly what train.py does.
    """
    df = df.copy()
    
    # Face features (ratios)
    face_features = ['FacialSymmetry', 'GoldenRatioDeviation', 'EyeSpacingRatio', 
                     'JawlineWidthRatio', 'BrowToEyeDistance', 'LipToNoseDistance']
    
    # Frame numeric features (dimensions)
    frame_features = ['Width_mm', 'LensHeight_mm', 'LensWidth_mm', 
                      'NoseBridgeWidth_mm', 'TempleLength_mm']
    
    # Create interaction terms: face_ratio * frame_dimension
    for face_feat in face_features:
        if face_feat in df.columns:
            for frame_feat in frame_features:
                if frame_feat in df.columns:
                    interaction_name = f"{face_feat}_x_{frame_feat}"
                    df[interaction_name] = df[face_feat] * df[frame_feat]
    
    # Create face-specific ratios with frame dimensions
    if 'EyeSpacingRatio' in df.columns and 'Width_mm' in df.columns:
        df['EyeToFrameWidth'] = df['EyeSpacingRatio'] * 100 / df['Width_mm'].clip(lower=1)
    
    if 'JawlineWidthRatio' in df.columns and 'Width_mm' in df.columns:
        df['JawToFrameWidth'] = df['JawlineWidthRatio'] * 100 / df['Width_mm'].clip(lower=1)
    
    return df


def recommend(new_face: dict, model_path: Path, top_k: int = 5):
    root = Path(__file__).parent.parent
    xcols, scaler, frame_catalog = load_artifacts(root)

    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    # Build combined DF: duplicate new_face for each frame
    if frame_catalog is None:
        raise FileNotFoundError("frame_catalog.csv not found in data/ — run preprocess.py first")

    face_df = pd.DataFrame([new_face] * len(frame_catalog)).reset_index(drop=True)
    combined = pd.concat([face_df.reset_index(drop=True), frame_catalog.reset_index(drop=True)], axis=1)

    # Create interaction features (same as training)
    combined = create_interaction_features(combined)

    # One-hot encode frame categorical features consistently with X_columns
    combined_encoded = pd.get_dummies(combined, columns=[c for c in ['Brand', 'Material', 'RimStyle', 'BridgeType', 'Color'] if c in combined.columns], drop_first=False)

    # Add missing columns and ensure column ordering
    for col in xcols:
        if col not in combined_encoded.columns:
            combined_encoded[col] = 0
    combined_encoded = combined_encoded[xcols]

    # Scale numeric columns if scaler available
    if scaler:
        numeric_cols = combined_encoded.select_dtypes(include=['number']).columns
        combined_encoded[numeric_cols] = scaler.transform(combined_encoded[numeric_cols])

    preds = model.predict(combined_encoded)
    
    # DEBUG: Show prediction variance
    print(f"[DEBUG] Prediction range: min={preds.min():.4f}, max={preds.max():.4f}, std={preds.std():.4f}")
    print(f"[DEBUG] Top {top_k} frame IDs: ", end="")
    
    combined['PredictedBeautyScore'] = preds
    top = combined.sort_values(by='PredictedBeautyScore', ascending=False).head(top_k)
    
    print(top['FrameID'].tolist())
    
    return top[['FrameID'] + [c for c in ['Brand', 'Shape', 'Color'] if c in top.columns] + ['PredictedBeautyScore']]


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/regressor.joblib', help='Path to saved regressor')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--face-json', type=str, required=True, help='Path to json file with new face features')
    args = parser.parse_args()

    new_face = json.loads(Path(args.face_json).read_text(encoding='utf-8'))
    top = recommend(new_face, Path(args.model), top_k=args.top_k)
    print(top.to_string(index=False))


if __name__ == '__main__':
    cli()

