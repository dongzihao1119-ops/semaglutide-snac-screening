"""Evaluate trained QSAR models with SHAP analysis.

Purpose: Load trained model, compute SHAP values, rank features by importance.

Input:
  - models/qsar_activity_xgboost.json (or lightgbm)
  - models/scaler.pkl
  - data/processed/pe_features.csv

Output:
  - results/qsar/feature_importance.csv
  - results/qsar/shap_values.csv

Example usage:
    python code/qsar/evaluate_qsar.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

FEATURES_PATH = DATA_PROCESSED / "pe_features.csv"
MODEL_PATH = MODELS_DIR / "qsar_activity_xgboost.json"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

PHYSCHEM_FEATURES = [
    "mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds",
    "aromatic_rings", "heavy_atoms", "fraction_csp3", "num_rings",
    "num_saturated_rings", "num_aliphatic_rings", "max_ring_size",
    "num_heteroatoms", "num_amide_bonds", "num_carboxyl",
    "num_hydroxyl", "num_aromatic_oh", "num_ether", "num_ester",
]
ECFP4_FEATURES = [f"ecfp4_{i:03d}" for i in range(128)]
CUSTOM_FEATURES = [
    "aliphatic_c_count", "aromatic_substitution_count",
    "has_ortho_oh_amide", "carboxyl_count", "amine_count",
    "hydrophobic_atom_count",
]


def main():
    qsar_results = RESULTS_DIR / "qsar"
    qsar_results.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Run train_qsar.py first.")
        return
    if not SCALER_PATH.exists():
        print(f"Scaler not found: {SCALER_PATH}")
        return

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(FEATURES_PATH)
    mask = df["teer_reduction_pct"].notna()
    df_valid = df[mask].copy()
    print(f"Loaded {len(df_valid)} compounds with TEER data")

    feature_cols = [c for c in PHYSCHEM_FEATURES + ECFP4_FEATURES + CUSTOM_FEATURES
                    if c in df_valid.columns]

    X = df_valid[feature_cols].fillna(0).values.astype(np.float64)
    X_scaled = scaler.transform(X)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(qsar_results / "feature_importance.csv", index=False)
    print(f"\nTop 10 features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s}  {row['mean_abs_shap']:.4f}")

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_csv(qsar_results / "shap_values.csv", index=False)
    print(f"\nResults saved to {qsar_results}/")


if __name__ == "__main__":
    main()
