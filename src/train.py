"""train.py
Train a simple RandomForestRegressor on data/processed_data.csv and save model to models/regressor.joblib
"""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib


def main(processed_csv='data/processed_data.csv', model_out='models/regressor.joblib'):
    p = Path(processed_csv)
    if not p.exists():
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv}. Run preprocess.py first.")
    df = pd.read_csv(p)
    if 'AdjustedBeautyScore' not in df.columns:
        raise ValueError('Target AdjustedBeautyScore not found in processed CSV')

    # Drop identifiers and target
    X = df.drop(columns=[c for c in ['FaceID', 'FrameID', 'AdjustedBeautyScore'] if c in df.columns])
    y = df['AdjustedBeautyScore']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X_train, y_train)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, model_out)
    print(f"Saved regressor to {model_out}")


if __name__ == '__main__':
    main()
