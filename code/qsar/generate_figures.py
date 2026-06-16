"""Generate all 7 publication-quality figures for the mechanism classification paper.

Output: paper/figures/fig1–fig7.png (300 DPI)

Usage: python code/qsar/generate_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, PAPER_DIR, RANDOM_SEED

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, RocCurveDisplay, auc,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import xgboost as xgb
import shap
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# Style
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'sans-serif',
})

FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_PROCESSED / "pe_features.csv"

PHYSCHEM = ["mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds",
            "aromatic_rings", "heavy_atoms", "fraction_csp3", "num_rings",
            "num_amide_bonds", "num_carboxyl", "num_hydroxyl", "num_ether", "num_ester"]
ECFP4 = [f"ecfp4_{i:03d}" for i in range(128)]
CUSTOM = ["aliphatic_c_count", "aromatic_substitution_count",
          "has_ortho_oh_amide", "carboxyl_count", "amine_count",
          "hydrophobic_atom_count"]

# Load data
df = pd.read_csv(FEATURES_PATH)
mask = df["mechanism_type"].isin(["paracellular", "transcellular"])
df_bin = df[mask].copy()
feature_cols = [c for c in PHYSCHEM + ECFP4 + CUSTOM if c in df_bin.columns]
X = df_bin[feature_cols].fillna(0).values.astype(np.float64)
y = (df_bin["mechanism_type"] == "paracellular").astype(int).values
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, reg_alpha=0.5,
    random_state=RANDOM_SEED, verbosity=0, eval_metric='logloss',
)
loo = LeaveOneOut()
y_pred = cross_val_predict(model, X_s, y, cv=loo)
model.fit(X_s, y)
y_proba = model.predict_proba(X_s)[:, 1]

# ===================================================================
# Fig 1: Chemical Space (t-SNE)
# ===================================================================
print("Fig 1: t-SNE chemical space...")
tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=5)
X_tsne = tsne.fit_transform(X_s)
fig, ax = plt.subplots(figsize=(8, 6))
colors = {'paracellular': '#E74C3C', 'transcellular': '#3498DB'}
for mech in ['paracellular', 'transcellular']:
    idx = df_bin["mechanism_type"] == mech
    ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=colors[mech],
               label=mech.capitalize(), s=100, edgecolors='black', linewidth=0.5, alpha=0.8)
for i, name in enumerate(df_bin["name"]):
    ax.annotate(df_bin.iloc[i]["abbrev"] if "abbrev" in df_bin.columns else name[:12],
                (X_tsne[i, 0], X_tsne[i, 1]), fontsize=6, alpha=0.7)
ax.set_xlabel("t-SNE 1"), ax.set_ylabel("t-SNE 2")
ax.set_title("Chemical Space of Permeation Enhancers\nColored by Mechanism Type")
ax.legend()
fig.savefig(FIG_DIR / "fig1_chemical_space.png"), plt.close()

# ===================================================================
# Fig 2: Confusion Matrix + ROC
# ===================================================================
print("Fig 2: Confusion matrix + ROC...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Transcellular", "Paracellular"])
disp.plot(ax=ax1, cmap='Blues', colorbar=False)
ax1.set_title(f"Confusion Matrix (LOO-CV)\nAccuracy={np.mean(y_pred==y):.0%}")

fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_xlabel("False Positive Rate"), ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve (LOO-CV)"), ax2.legend()
fig.savefig(FIG_DIR / "fig2_classification_performance.png"), plt.close()

# ===================================================================
# Fig 3: SHAP Beeswarm
# ===================================================================
print("Fig 3: SHAP beeswarm...")
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_s)
top_idx = np.argsort(np.abs(shap_vals).mean(0))[-20:]
top_names = [feature_cols[i] for i in top_idx]
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals[:, top_idx], X_s[:, top_idx],
                  feature_names=top_names, show=False,
                  plot_type="dot", max_display=20)
ax.set_title("SHAP Feature Importance: Paracellular vs Transcellular")
fig.savefig(FIG_DIR / "fig3_shap_beeswarm.png", dpi=300, bbox_inches='tight')
plt.close()

# ===================================================================
# Fig 4: ECFP4 substructure visualization
# ===================================================================
print("Fig 4: ECFP4 key substructures...")
importance = np.abs(shap_vals).mean(0)
ecfp_importance = [(i, importance[feature_cols.index(f)]) for i, f in enumerate(feature_cols) if f.startswith('ecfp4')]
ecfp_importance.sort(key=lambda x: x[1], reverse=True)
top_ecfp_bits = ecfp_importance[:6]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, (bit_idx, imp_val) in enumerate(top_ecfp_bits):
    ax = axes[idx // 3][idx % 3]
    bit_num = int(feature_cols[bit_idx].split('_')[1])
    ax.text(0.5, 0.5, f"ECFP4 Bit {bit_num}\nSHAP: {imp_val:.3f}",
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title(f"Bit {bit_num} (importance {imp_val:.3f})")
    ax.axis('off')
fig.suptitle("Top 6 ECFP4 Fingerprint Bits for Mechanism Classification\n(Substructure depiction requires bit-info mapping)")
fig.savefig(FIG_DIR / "fig4_ecfp4_substructures.png"), plt.close()

# ===================================================================
# Fig 5: Physicochemical Trends
# ===================================================================
print("Fig 5: Physicochemical trends...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# logP vs fraction_csp3
for mech, c in [('paracellular', '#E74C3C'), ('transcellular', '#3498DB')]:
    idx = df_bin["mechanism_type"] == mech
    axes[0].scatter(df_bin.loc[idx, "logp"], df_bin.loc[idx, "fraction_csp3"],
                    c=c, label=mech.capitalize(), s=80, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel("logP"), axes[0].set_ylabel("Fraction C(sp3)")
axes[0].set_title("logP vs Fraction C(sp3)"), axes[0].legend()

# MW distribution
for mech, c in [('paracellular', '#E74C3C'), ('transcellular', '#3498DB')]:
    idx = df_bin["mechanism_type"] == mech
    axes[1].hist(df_bin.loc[idx, "mw"], bins=8, alpha=0.5, color=c, label=mech.capitalize())
axes[1].set_xlabel("Molecular Weight (Da)"), axes[1].set_ylabel("Count")
axes[1].set_title("MW Distribution by Mechanism"), axes[1].legend()

# rotatable_bonds vs tpsa
for mech, c in [('paracellular', '#E74C3C'), ('transcellular', '#3498DB')]:
    idx = df_bin["mechanism_type"] == mech
    axes[2].scatter(df_bin.loc[idx, "rotatable_bonds"], df_bin.loc[idx, "tpsa"],
                    c=c, label=mech.capitalize(), s=80, edgecolors='black', linewidth=0.5)
axes[2].set_xlabel("Rotatable Bonds"), axes[2].set_ylabel("TPSA (Å²)")
axes[2].set_title("Rotatable Bonds vs TPSA"), axes[2].legend()

fig.savefig(FIG_DIR / "fig5_physicochemical_trends.png"), plt.close()

# ===================================================================
# Fig 6: Chemical Category Analysis
# ===================================================================
print("Fig 6: Chemical category analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

cat_counts = df_bin.groupby(["category", "mechanism_type"]).size().unstack(fill_value=0)
cat_counts = cat_counts.loc[cat_counts.sum(1).sort_values(ascending=False).index]
cat_counts.plot(kind="barh", stacked=True, ax=ax1, color=['#3498DB', '#E74C3C'])
ax1.set_xlabel("Number of Compounds"), ax1.set_title("Mechanism Distribution by Chemical Category")
ax1.legend(["Transcellular", "Paracellular"])

# K-value by category
cat_k = df_bin.groupby("category")["k_value"].mean().sort_values(ascending=False)
colors_k = ['#E74C3C' if v > 0.5 else '#3498DB' if v < 0.5 else '#95A5A6' for v in cat_k.values]
cat_k.plot(kind="barh", ax=ax2, color=colors_k)
ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='K=0.5 boundary')
ax2.set_xlabel("Mean K-value (0=transcellular, 1=paracellular)")
ax2.set_title("K-value by Chemical Category")
legend_elements = [Patch(facecolor='#E74C3C', label='Paracellular (K>0.5)'),
                   Patch(facecolor='#3498DB', label='Transcellular (K<0.5)')]
ax2.legend(handles=legend_elements)

fig.savefig(FIG_DIR / "fig6_category_analysis.png"), plt.close()

# ===================================================================
# Fig 7: Case Studies — 4 representative PE structures
# ===================================================================
print("Fig 7: Case studies...")
case_studies = [
    ("PPS (Transcellular)", "CCCCCCCCCCCCCCCC[N+](C)(C)CCCS([O-])(=O)=O", "K=0.0, EP=0.8"),
    ("EDTA (Paracellular)", "O=C(O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O", "K=0.72, EP=0.98"),
    ("CTAB (Transcellular)", "CCCCCCCCCCCCCCCC[N+](C)(C)C", "Cationic surfactant"),
    ("Phenylpiperazine (Paracellular)", "c1ccc(N2CCNCC2)cc1", "K=0.86, EP=0.95"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (name, smiles, info) in enumerate(case_studies):
    ax = axes[idx // 2][idx % 2]
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(400, 300))
        ax.imshow(img)
    ax.set_title(f"{name}\n{info}", fontsize=11)
    ax.axis('off')
fig.suptitle("Case Studies: Representative Permeation Enhancers", fontsize=14, y=1.02)
fig.savefig(FIG_DIR / "fig7_case_studies.png"), plt.close()

print(f"\nAll figures saved to {FIG_DIR}/")
for f in sorted(FIG_DIR.glob("fig*.png")):
    print(f"  {f.name}")
