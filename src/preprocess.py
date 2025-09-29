"""preprocess.py
Merge frame catalog and face-frame dataset, produce a processed dataset for modeling,
and save model input column list and a scaler for later use.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def main():
    root = Path(__file__).parent.parent
    df_pairs_path = root / "Expanded_Face-to-Frame_Matching_Dataset.csv"
    df_frames_path = root / "Recommended_Eyeglass_Frames_By_Adjusted_Beauty_Score.csv"
    out_dir = root / "data"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {df_pairs_path} and {df_frames_path}")
    df1 = pd.read_csv(df_pairs_path)
    df2 = pd.read_csv(df_frames_path)

    # Drop duplicate FaceID rows in df2 keeping the first occurrence
    df2_unique = df2.drop_duplicates(subset='FaceID')

    # Columns to merge (same intent as the notebook)
    columns_to_merge = ['FaceID', 'FacialSymmetry', 'GoldenRatioDeviation', 'EyeSpacingRatio',
                        'JawlineWidthRatio', 'BrowToEyeDistance', 'LipToNoseDistance',
                        'Brand', 'Material', 'RimStyle', 'BridgeType', 'Color',
                        'Width_mm', 'LensHeight_mm', 'LensWidth_mm', 'NoseBridgeWidth_mm', 'TempleLength_mm']

    # Merge enriched columns into pair dataset
    df = df1.merge(df2_unique[columns_to_merge], on='FaceID', how='left')

    # Derive MaxBeauty_FrameID per FaceID (as in notebook)
    if 'BeautyScore' in df.columns:
        max_beauty_frame = df.loc[df.groupby('FaceID')['BeautyScore'].idxmax(), ['FaceID', 'FrameID']]
        max_beauty_frame = max_beauty_frame.rename(columns={'FrameID': 'MaxBeauty_FrameID'})
        df = df.merge(max_beauty_frame, on='FaceID', how='left')

    # Prepare modeling dataframe: keep face and frame features plus target if present
    target_col = 'AdjustedBeautyScore' if 'AdjustedBeautyScore' in df.columns else None

    face_features = [
        'FaceID', 'FacialSymmetry', 'GoldenRatioDeviation', 'EyeSpacingRatio',
        'JawlineWidthRatio', 'BrowToEyeDistance', 'LipToNoseDistance'
    ]

    frame_features = [
        'FrameID', 'Brand', 'Material', 'RimStyle', 'BridgeType', 'Color',
        'Width_mm', 'LensHeight_mm', 'LensWidth_mm', 'NoseBridgeWidth_mm', 'TempleLength_mm'
    ]

    cols_to_keep = [c for c in face_features + frame_features if c in df.columns]
    if target_col:
        cols_to_keep.append(target_col)

    df_model = df[cols_to_keep].copy()

    # One-hot encode categorical frame features
    categorical_cols = [c for c in ['Brand', 'Material', 'RimStyle', 'BridgeType', 'Color'] if c in df_model.columns]
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=False)

    # Define X and y for modeling (drop IDs)
    drop_cols = [c for c in ['FaceID', 'FrameID'] if c in df_encoded.columns]
    X = df_encoded.drop(columns=drop_cols + ([target_col] if target_col else []))
    y = df_encoded[target_col] if target_col else None

    # Fit and save a scaler for numeric features
    scaler = StandardScaler()
    X_numeric = X.select_dtypes(include=['number'])
    if not X_numeric.empty:
        scaler.fit(X_numeric)
        joblib.dump(scaler, out_dir / 'scaler.joblib')
        print(f"Saved scaler to {out_dir / 'scaler.joblib'}")

    # Save processed dataset
    processed_path = out_dir / 'processed_data.csv'
    df_encoded.to_csv(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path}")

    # Save X column list for later alignment
    xcols_path = out_dir / 'X_columns.json'
    with open(xcols_path, 'w', encoding='utf-8') as f:
        json.dump(list(X.columns), f, indent=2)
    print(f"Saved X columns to {xcols_path}")

    # Save a frame catalog (unique frames)
    frame_catalog_path = out_dir / 'frame_catalog.csv'
    if 'FrameID' in df_model.columns:
        frame_catalog = df_model[frame_features].drop_duplicates(subset='FrameID')
        frame_catalog.to_csv(frame_catalog_path, index=False)
        print(f"Saved frame catalog to {frame_catalog_path}")


if __name__ == '__main__':
    main()
