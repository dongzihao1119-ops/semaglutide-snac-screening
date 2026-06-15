"""Train QSAR models for permeation enhancer activity prediction.

Purpose: Train XGBoost and LightGBM models with scaffold-based train/test
splitting to prevent data leakage. Saves best model, scaler, and predictions.

Input: data/processed/pe_features.csv
Output:
  - models/qsar_activity_xgboost.json or qsar_activity_lightgbm.txt
  - models/scaler.pkl
  - results/qsar/test_predictions.csv

Example usage:
    python code/qsar/train_qsar.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import (
    DATA_PROCESSED, MODELS_DIR, RESULTS_DIR,
    RANDOM_SEED, TEST_SIZE, TARGET_R2_ACTIVITY,
)

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import xgboost as xgb
import lightgbm as lgb

FEATURES_PATH = DATA_PROCESSED / "pe_features.csv"

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


def get_scaffold_groups(df):
    scaffolds = {}
    group_ids = np.zeros(len(df), dtype=int)
    next_id = 0
    for i, smiles in enumerate(df["clean_smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            group_ids[i] = next_id
            next_id += 1
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = next_id
            next_id += 1
        group_ids[i] = scaffolds[scaffold]
    print(f"  {len(scaffolds)} unique Murcko scaffolds for {len(df)} cpds")
    return group_ids


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "qsar").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} compounds")

    # --- Activity model (TEER reduction) ---
    print("\n" + "=" * 60)
    print("TRAINING: teer_reduction_pct")
    print("=" * 60)

    feature_cols = [c for c in PHYSCHEM_FEATURES + ECFP4_FEATURES + CUSTOM_FEATURES
                    if c in df.columns]
    print(f"  Using {len(feature_cols)} features")

    mask = df["teer_reduction_pct"].notna()
    df_valid = df[mask].copy()
    print(f"  {len(df_valid)} compounds with TEER data")

    X = df_valid[feature_cols].fillna(0).values.astype(np.float64)
    y = df_valid["teer_reduction_pct"].values.astype(np.float64)
    groups = get_scaffold_groups(df_valid)

    if len(y) < 5:
        print("TOO FEW SAMPLES for train/test split. Running CV on all data.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_SEED, verbosity=0,
        )
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)),
                                 scoring="r2")
        print(f"  XGBoost LOOCV R²: {scores.mean():.3f} ± {scores.std():.3f}")
        model.fit(X_scaled, y)
        model.save_model(str(MODELS_DIR / "qsar_activity_xgboost.json"))
        with open(MODELS_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("  Model saved (trained on all data — no test split possible)")
        return

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  Train: {len(X_train)} ({len(set(groups[train_idx]))} scaffolds)")
    print(f"  Test:  {len(X_test)} ({len(set(groups[test_idx]))} scaffolds)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, verbosity=0,
    )
    xgb_model.fit(X_train_s, y_train)
    xgb_preds = xgb_model.predict(X_test_s)
    xgb_r2 = r2_score(y_test, xgb_preds)
    print(f"\n  XGBoost  — R²={xgb_r2:.3f}, RMSE={np.sqrt(mean_squared_error(y_test, xgb_preds)):.1f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, verbose=-1,
    )
    lgb_model.fit(X_train_s, y_train)
    lgb_preds = lgb_model.predict(X_test_s)
    lgb_r2 = r2_score(y_test, lgb_preds)
    print(f"  LightGBM — R²={lgb_r2:.3f}, RMSE={np.sqrt(mean_squared_error(y_test, lgb_preds)):.1f}")

    # Save best
    if xgb_r2 >= lgb_r2:
        xgb_model.save_model(str(MODELS_DIR / "qsar_activity_xgboost.json"))
        print("\n  Saved: XGBoost")
    else:
        lgb_model.booster_.save_model(str(MODELS_DIR / "qsar_activity_lightgbm.txt"))
        print("\n  Saved: LightGBM")

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    best_r2 = max(xgb_r2, lgb_r2)
    print(f"  Target R²: {TARGET_R2_ACTIVITY}  |  Achieved: {best_r2:.3f}")
    print("  ✅ Target met!" if best_r2 >= TARGET_R2_ACTIVITY else
          "  ⚠️  Below target. Need more data or feature engineering.")

    preds_df = pd.DataFrame({
        "compound": df_valid["name"].iloc[test_idx].values,
        "y_true": y_test,
        "y_pred_xgb": xgb_preds,
        "y_pred_lgb": lgb_preds,
    })
    preds_df.to_csv(RESULTS_DIR / "qsar" / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
