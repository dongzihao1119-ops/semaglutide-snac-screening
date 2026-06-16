"""Classification model for PE mechanism prediction.

Purpose: Classify permeation enhancers by mechanism type
(transcellular vs paracellular) and by Maher 2016 class,
using molecular descriptors + ECFP4 fingerprints.

Input: data/processed/pe_features.csv (with mechanism_type, maher_class columns)
Output:
  - results/qsar/mechanism_classification_report.txt
  - results/qsar/mechanism_shap.csv

Example: python code/qsar/classify_mechanism.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import DATA_PROCESSED, RESULTS_DIR, RANDOM_SEED

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    LeaveOneOut, cross_val_predict, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef,
)
import xgboost as xgb
import shap

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


def main():
    qsar_results = RESULTS_DIR / "qsar"
    qsar_results.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)

    # ===================================================================
    # BINARY: paracellular vs transcellular (exclude mixed)
    # ===================================================================
    mask = df["mechanism_type"].isin(["paracellular", "transcellular"])
    df_bin = df[mask].copy()
    y_bin = (df_bin["mechanism_type"] == "paracellular").astype(int).values
    print(f"Binary classification: {len(df_bin)} compounds")
    print(f"  Paracellular: {y_bin.sum()}, Transcellular: {len(y_bin)-y_bin.sum()}")

    feature_cols = [c for c in PHYSCHEM_FEATURES + ECFP4_FEATURES + CUSTOM_FEATURES
                    if c in df_bin.columns]
    X_bin = df_bin[feature_cols].fillna(0).values.astype(np.float64)

    scaler = StandardScaler()
    X_bin_scaled = scaler.fit_transform(X_bin)

    # LOO-CV
    loo = LeaveOneOut()
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, reg_alpha=0.5,
        random_state=RANDOM_SEED, verbosity=0,
        eval_metric='logloss',
    )
    y_pred_loo = cross_val_predict(model, X_bin_scaled, y_bin, cv=loo)

    print(f"\n{'='*60}")
    print("BINARY: Paracellular vs Transcellular — LOO-CV")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy_score(y_bin, y_pred_loo):.3f}")
    print(f"  Balanced:  {balanced_accuracy_score(y_bin, y_pred_loo):.3f}")
    print(f"  MCC:       {matthews_corrcoef(y_bin, y_pred_loo):.3f}")
    print(f"\n{classification_report(y_bin, y_pred_loo, target_names=['Transcellular', 'Paracellular'])}")

    cm = confusion_matrix(y_bin, y_pred_loo)
    print(f"  Confusion Matrix:\n    TN={cm[0][0]}, FP={cm[0][1]}\n    FN={cm[1][0]}, TP={cm[1][1]}")

    # Train final model for SHAP
    model.fit(X_bin_scaled, y_bin)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_bin_scaled)
    importance = np.abs(shap_vals).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False)
    print(f"\n  Top 10 features for mechanism classification:")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:30s}  {row['mean_abs_shap']:.4f}")

    # ===================================================================
    # MULTI-CLASS: Maher 6-class
    # ===================================================================
    df_multi = df[df["maher_class"].notna()].copy()
    y_multi = df_multi["maher_class"].astype(int).values
    class_names = {
        1: "Class1_Ideal",
        2: "Class2_Unacceptable_toxic",
        3: "Class3_Modest_safe",
        4: "Class4_Slow_strong",
        5: "Class5_Slow_toxic",
        6: "Class6_Weak",
    }
    print(f"\n{'='*60}")
    print(f"MULTI-CLASS: Maher 6-Class — {len(df_multi)} compounds")
    print(f"{'='*60}")
    for c in sorted(set(y_multi)):
        n = sum(y_multi == c)
        print(f"  {class_names.get(c, c)}: {n} compounds")

    # Merge small classes for multi-class (min 3 per class), use 0-indexed
    class_map = {1: 0, 2: 1, 3: 2, 4: 2, 5: 1, 6: 3}  # 0=Ideal, 1=Toxic, 2=Moderate, 3=Weak
    y_multi_mapped = np.array([class_map[c] for c in y_multi])
    label_names = {0: "Ideal", 1: "Toxic", 2: "Moderate", 3: "Weak"}

    for c in sorted(set(y_multi_mapped)):
        n = sum(y_multi_mapped == c)
        print(f"  {label_names.get(c, c)}: {n} compounds")

    X_multi = df_multi[feature_cols].fillna(0).values.astype(np.float64)
    X_multi_s = StandardScaler().fit_transform(X_multi)

    n_folds = min(5, min(np.bincount(y_multi_mapped)))
    n_folds = max(2, n_folds)

    y_pred_multi = cross_val_predict(
        xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, reg_alpha=0.5,
            random_state=RANDOM_SEED, verbosity=0,
            eval_metric='mlogloss',
        ),
        X_multi_s, y_multi_mapped, cv=n_folds,
    )
    print(f"\n  Accuracy: {accuracy_score(y_multi, y_pred_multi):.3f}")
    print(f"  Balanced: {balanced_accuracy_score(y_multi, y_pred_multi):.3f}")

    # ===================================================================
    # Save results
    # ===================================================================
    with open(qsar_results / "mechanism_classification_report.txt", "w") as f:
        f.write(f"Binary classification (paracellular vs transcellular):\n")
        f.write(f"  N={len(df_bin)}, Para={y_bin.sum()}, Trans={len(y_bin)-y_bin.sum()}\n")
        f.write(f"  LOO-CV Accuracy: {accuracy_score(y_bin, y_pred_loo):.3f}\n")
        f.write(f"  Balanced: {balanced_accuracy_score(y_bin, y_pred_loo):.3f}\n")
        f.write(f"  MCC: {matthews_corrcoef(y_bin, y_pred_loo):.3f}\n")
        f.write(f"\n{classification_report(y_bin, y_pred_loo, target_names=['Transcellular', 'Paracellular'])}\n")
        f.write(f"\nMulti-class (Maher 6-class) Accuracy: {accuracy_score(y_multi, y_pred_multi):.3f}\n")

    print(f"\nResults saved to {qsar_results}/mechanism_classification_report.txt")


if __name__ == "__main__":
    main()
