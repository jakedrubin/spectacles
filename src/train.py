"""train.py
Train a RandomForestRegressor with face-frame interaction features.
This creates interaction terms so the model learns which frames work with which faces.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def create_interaction_features(df):
    """
    Create interaction features between face and frame properties.
    This helps the model learn face-frame compatibility rather than just ranking frames.
    """
    df = df.copy()
    
    # Face features (ratios)
    face_features = ['FacialSymmetry', 'GoldenRatioDeviation', 'EyeSpacingRatio', 
                     'JawlineWidthRatio', 'BrowToEyeDistance', 'LipToNoseDistance']
    
    # Frame numeric features (dimensions)
    frame_features = ['Width_mm', 'LensHeight_mm', 'LensWidth_mm', 
                      'NoseBridgeWidth_mm', 'TempleLength_mm']
    
    # Create interaction terms: face_ratio * frame_dimension
    # This captures "wide face + narrow frame = bad fit" type relationships
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


def main(processed_csv='data/processed_data.csv', model_out='models/regressor.joblib'):
    p = Path(processed_csv)
    if not p.exists():
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv}. Run preprocess.py first.")
    df = pd.read_csv(p)
    if 'AdjustedBeautyScore' not in df.columns:
        raise ValueError('Target AdjustedBeautyScore not found in processed CSV')

    # Create interaction features
    print("Creating face-frame interaction features...")
    df = create_interaction_features(df)

    # Drop identifiers and target
    X = df.drop(columns=[c for c in ['FaceID', 'FrameID', 'AdjustedBeautyScore'] if c in df.columns])
    y = df['AdjustedBeautyScore']

    # Save updated column list
    xcols_path = Path('data/X_columns.json')
    with open(xcols_path, 'w', encoding='utf-8') as f:
        json.dump(list(X.columns), f, indent=2)
    print(f"Saved {len(X.columns)} feature columns to {xcols_path}")

    # Scale numeric features and save scaler
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    scaler_path = Path('data/scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15, min_samples_leaf=5)
    reg.fit(X_train, y_train)
    
    # Report feature importances for face-related features
    importances = dict(zip(X.columns, reg.feature_importances_))
    face_cols = [c for c in importances if any(f in c for f in ['Facial', 'Golden', 'Eye', 'Jaw', 'Brow', 'Lip'])]
    face_importance = sum(importances[c] for c in face_cols)
    print(f"Face-related feature importance: {face_importance:.1%}")
    
    # Report test score
    test_score = reg.score(X_test, y_test)
    print(f"Test R² score: {test_score:.4f}")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, model_out)
    print(f"Saved regressor to {model_out}")


if __name__ == '__main__':
    main()

