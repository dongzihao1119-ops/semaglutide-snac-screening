"""Rigorous cross-validation for QSAR model.

Purpose: Validate XGBoost QSAR model with multiple CV strategies
to ensure the R²=0.757 result is not an artifact of small test set.

Strategies:
  1. Leave-one-out CV (LOO-CV) — gold standard for small datasets
  2. Repeated scaffold-based 5-fold CV (100 repeats)
  3. Random 5-fold CV (warns about scaffold leakage)
  4. Y-scrambling test (should give R² ≈ 0 if model is real)

Input: data/processed/pe_features.csv
Output: results/qsar/cv_summary.csv

Example: python code/qsar/cross_validate.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import (
    DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, RANDOM_SEED,
)

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    LeaveOneOut, cross_val_score, KFold, GroupKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import xgboost as xgb

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
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if s not in scaffolds:
            scaffolds[s] = next_id
            next_id += 1
        group_ids[i] = scaffolds[s]
    return group_ids


def main():
    qsar_results = RESULTS_DIR / "qsar"
    qsar_results.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(FEATURES_PATH)
    mask = df["teer_reduction_pct"].notna()
    df_valid = df[mask].copy()
    print(f"Valid compounds: {len(df_valid)}")

    feature_cols = [c for c in PHYSCHEM_FEATURES + ECFP4_FEATURES + CUSTOM_FEATURES
                    if c in df_valid.columns]
    X = df_valid[feature_cols].fillna(0).values.astype(np.float64)
    y = df_valid["teer_reduction_pct"].values.astype(np.float64)
    groups = get_scaffold_groups(df_valid)
    n_scaffolds = len(set(groups))
    print(f"Features: {len(feature_cols)}")
    print(f"Scaffolds: {n_scaffolds}")
    print(f"y range: {y.min():.0f}–{y.max():.0f}%, mean={y.mean():.0f}%")

    # Scale all data once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_kwargs = dict(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, verbosity=0,
    )

    results = {}

    # ===================================================================
    # 1. Leave-One-Out CV
    # ===================================================================
    print("\n" + "=" * 60)
    print("1. LEAVE-ONE-OUT CV")
    print("=" * 60)
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X_scaled):
        model = xgb.XGBRegressor(**model_kwargs)
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred_loo[test_idx] = model.predict(X_scaled[test_idx])
    loo_r2 = r2_score(y, y_pred_loo)
    loo_rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
    print(f"  LOO-CV R² = {loo_r2:.4f}")
    print(f"  LOO-CV RMSE = {loo_rmse:.1f}%")
    results["LOO-CV R²"] = loo_r2

    # ===================================================================
    # 2. Repeated scaffold-based 5-fold CV
    # ===================================================================
    print("\n" + "=" * 60)
    print("2. REPEATED SCAFFOLD-BASED 5-FOLD CV (100 repeats)")
    print("=" * 60)
    scaffold_cv_r2s = []
    for seed in range(RANDOM_SEED, RANDOM_SEED + 100):
        gkf = GroupKFold(n_splits=5)
        fold_r2s = []
        for train_idx, test_idx in gkf.split(X_scaled, y, groups):
            model = xgb.XGBRegressor(**{**model_kwargs, "random_state": seed})
            model.fit(X_scaled[train_idx], y[train_idx])
            preds = model.predict(X_scaled[test_idx])
            if len(np.unique(y[test_idx])) > 1:
                fold_r2s.append(r2_score(y[test_idx], preds))
        if fold_r2s:
            scaffold_cv_r2s.append(np.mean(fold_r2s))
    scaffold_cv_r2s = np.array(scaffold_cv_r2s)
    print(f"  Mean R² = {scaffold_cv_r2s.mean():.4f} ± {scaffold_cv_r2s.std():.4f}")
    print(f"  Median R² = {np.median(scaffold_cv_r2s):.4f}")
    print(f"  Min R² = {scaffold_cv_r2s.min():.4f}, Max R² = {scaffold_cv_r2s.max():.4f}")
    results["Scaffold-5Fold mean R²"] = scaffold_cv_r2s.mean()
    results["Scaffold-5Fold std R²"] = scaffold_cv_r2s.std()

    # ===================================================================
    # 3. Random 5-fold CV (WARNING: leaks scaffolds)
    # ===================================================================
    print("\n" + "=" * 60)
    print("3. RANDOM 5-FOLD CV (⚠️ scaffold leakage — upper bound only)")
    print("=" * 60)
    random_cv_r2s = []
    for seed in range(RANDOM_SEED, RANDOM_SEED + 100):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_r2s = []
        for train_idx, test_idx in kf.split(X_scaled):
            model = xgb.XGBRegressor(**{**model_kwargs, "random_state": seed})
            model.fit(X_scaled[train_idx], y[train_idx])
            preds = model.predict(X_scaled[test_idx])
            if len(np.unique(y[test_idx])) > 1:
                fold_r2s.append(r2_score(y[test_idx], preds))
        if fold_r2s:
            random_cv_r2s.append(np.mean(fold_r2s))
    random_cv_r2s = np.array(random_cv_r2s)
    print(f"  Mean R² = {random_cv_r2s.mean():.4f} ± {random_cv_r2s.std():.4f}")
    results["Random-5Fold mean R²"] = random_cv_r2s.mean()

    # ===================================================================
    # 4. Y-SCRAMBLING TEST
    # ===================================================================
    print("\n" + "=" * 60)
    print("4. Y-SCRAMBLING TEST (should give R² ≈ 0)")
    print("=" * 60)
    y_scrambled = y.copy()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(y_scrambled)
    y_pred_scram = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X_scaled):
        model = xgb.XGBRegressor(**model_kwargs)
        model.fit(X_scaled[train_idx], y_scrambled[train_idx])
        y_pred_scram[test_idx] = model.predict(X_scaled[test_idx])
    scram_r2 = r2_score(y_scrambled, y_pred_scram)
    print(f"  Y-scrambled LOO R² = {scram_r2:.4f}")
    print(f"  Expected: close to 0 or negative")
    results["Y-scrambled R²"] = scram_r2

    # ===================================================================
    # 5. Save predictions and CV results
    # ===================================================================
    preds_df = pd.DataFrame({
        "compound": df_valid["name"].values,
        "y_true": y,
        "y_pred_loo": y_pred_loo,
        "residual": y - y_pred_loo,
    })
    preds_df["abs_error"] = preds_df["residual"].abs()
    preds_df = preds_df.sort_values("abs_error")
    preds_df.to_csv(qsar_results / "cv_predictions.csv", index=False)

    print(f"\nBest predicted (lowest error):")
    for _, row in preds_df.head(5).iterrows():
        print(f"  {row['compound'][:50]:50s} true={row['y_true']:5.0f}% pred={row['y_pred_loo']:5.0f}% err={row['abs_error']:5.0f}%")

    print(f"\nWorst predicted (highest error):")
    for _, row in preds_df.tail(5).iterrows():
        print(f"  {row['compound'][:50]:50s} true={row['y_true']:5.0f}% pred={row['y_pred_loo']:5.0f}% err={row['abs_error']:5.0f}%")

    # Summary
    print(f"\n{'='*60}")
    print("CV SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save model trained on ALL data for future use
    print("\nTraining final model on all 52 compounds...")
    final_model = xgb.XGBRegressor(**model_kwargs)
    final_model.fit(X_scaled, y)
    final_model.save_model(str(MODELS_DIR / "qsar_activity_xgboost.json"))
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Final model saved.")


if __name__ == "__main__":
    main()
