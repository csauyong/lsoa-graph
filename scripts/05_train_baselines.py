"""
Stage 5 — Train tabular baselines (Ridge, Random Forest, XGBoost).

For each spatial CV fold:
  - Fit StandardScaler on training fold only
  - Train Ridge, RandomForest, XGBoost
  - Predict on held-out test fold
  - Append predictions to results

Output:
  outputs/results_baselines.csv
    columns: fold, model, lsoa_code, y_true, y_pred
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LSOA_FEATURES_PQ, FOLDS_PT, RESULTS_BASELINES_CSV,
    FEATURE_COLS, TARGET_COL, RANDOM_SEED, OUTPUTS_DIR,
)

np.random.seed(RANDOM_SEED)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading features...")
df = pd.read_parquet(LSOA_FEATURES_PQ)
print(f"  Shape: {df.shape}")

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)
lsoa_codes = df["LSOA21CD"].tolist()
print(f"  Features: {X.shape}  Target: {y.shape}")

print("Loading folds...")
folds = torch.load(FOLDS_PT, weights_only=False)
print(f"  {len(folds)} folds loaded")

# ── Model definitions ─────────────────────────────────────────────────────────
def make_models():
    return {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=0,
        ),
    }

# ── Training loop ─────────────────────────────────────────────────────────────
records = []

for fold_id, (train_idx, test_idx) in enumerate(folds):
    train_idx = train_idx.numpy()
    test_idx  = test_idx.numpy()

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"\nFold {fold_id}  train={len(train_idx):,}  test={len(test_idx):,}")

    for model_name, model in make_models().items():
        print(f"  Training {model_name}...", end=" ", flush=True)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        print("done")

        for i, idx in enumerate(test_idx):
            records.append({
                "fold":      fold_id,
                "model":     model_name,
                "lsoa_code": lsoa_codes[idx],
                "y_true":    float(y_test[i]),
                "y_pred":    float(preds[i]),
            })

# ── Save ──────────────────────────────────────────────────────────────────────
results = pd.DataFrame(records)
results.to_csv(RESULTS_BASELINES_CSV, index=False)
print(f"\nSaved: {RESULTS_BASELINES_CSV}")
print(f"  Rows: {len(results):,}")
print(f"  Models: {results['model'].unique().tolist()}")
print(f"  Folds:  {sorted(results['fold'].unique().tolist())}")
