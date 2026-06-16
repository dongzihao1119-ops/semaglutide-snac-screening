# Methods

## Dataset Construction

A dataset of 53 chemically diverse permeation enhancers (PEs) was compiled from the systematic Caco-2 screening study by Whitehead et al. [1] and expanded with additional compounds from Maher et al. [2], Brayden et al. [3], and Twarog et al. [4]. The dataset spans 13 chemical categories: anionic surfactants (n=5), cationic surfactants (n=5), zwitterionic surfactants (n=3), nonionic surfactants (n=1), bile salts (n=5), fatty acids (n=5), fatty esters (n=5), sodium salts of fatty acids (n=1), fatty amines (n=2), nitrogen-containing rings (n=5), alkyl saccharides (n=4), chelating agents (n=1), and others (n=11).

For each compound, the following data were recorded: chemical name, SMILES string, chemical category, TEER reduction percentage (at 1%, 0.1%, and 0.01% w/v where available), and mechanism annotation. Mechanism type (paracellular vs transcellular) was assigned based on the K-value defined by Whitehead et al. [1]: K = (EP − LP)/EP, where EP is enhancement potential and LP is LDH potential. Compounds with K > 0.5 were classified as predominantly paracellular; K < 0.5 as predominantly transcellular. After excluding mixtures and compounds with incomplete structural data, 25 compounds with unambiguous mechanism assignment were used for binary classification.

## Molecular Feature Computation

All molecular structures were standardized using RDKit (version 2022.09) [5]: salt removal, charge neutralization, and canonicalization. Three categories of molecular features were computed for each compound:

1. **Physicochemical descriptors** (n=20): molecular weight (MW), octanol-water partition coefficient (logP), hydrogen bond donors/acceptors (HBD/HBA), topological polar surface area (TPSA), rotatable bond count, aromatic ring count, heavy atom count, fraction of sp3-hybridized carbons (fraction Csp3), ring counts, heteroatom count, and functional group counts (amide, carboxyl, hydroxyl, aromatic hydroxyl, ether, ester).

2. **Extended-connectivity fingerprints** (ECFP4, n=128): Morgan circular fingerprints with radius 2, encoded as 128-bit binary vectors.

3. **Custom PE-specific features** (n=7): aliphatic carbon count, aromatic substitution count, presence of ortho-hydroxy amide motif (salicylamide recognition), carboxyl group count, amine group count, and hydrophobic atom count.

The final feature matrix comprised 155 molecular descriptors per compound.

## Binary Classification Model

An XGBoost classifier [6] was trained to predict mechanism type (paracellular vs transcellular) from molecular features. Hyperparameters were set as follows: n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, reg_alpha=0.5. Features were standardized using scikit-learn's StandardScaler prior to model training.

Model performance was evaluated using leave-one-out cross-validation (LOO-CV), which provides an unbiased estimate of generalization performance for small datasets. Performance metrics included accuracy, balanced accuracy, Matthews correlation coefficient (MCC), precision, recall, and F1-score. A confusion matrix and receiver operating characteristic (ROC) curve with area under curve (AUC) were generated.

## Model Interpretability

SHAP (SHapley Additive exPlanations) [7] TreeExplainer was used to interpret the XGBoost classifier. Mean absolute SHAP values were computed for each feature across all predictions, providing a ranking of feature importance. SHAP beeswarm plots were generated to visualize the distribution of feature contributions across individual predictions.

## Chemical Space Visualization

t-distributed Stochastic Neighbor Embedding (t-SNE) [8] was applied to the standardized feature matrix (perplexity=5) to project the 155-dimensional feature space into two dimensions for visualization.

## Computational Environment

All analyses were performed in Python 3.9 using RDKit 2022.09, XGBoost 2.1.4, scikit-learn 1.6.1, SHAP 0.49.1, pandas 2.3.3, NumPy 1.26.4, matplotlib 3.9.4, and seaborn 0.13.2. The complete code, dataset, and trained models are available at [GitHub repository URL].

---

## References

[1] Whitehead K, Mitragotri S. Mechanistic analysis of chemical permeation enhancers for oral drug delivery. Pharm Res. 2008;25(6):1412-1419. doi:10.1007/s11095-008-9542-2

[2] Maher S, Mrsny RJ, Brayden DJ. Intestinal permeation enhancers for oral peptide delivery. Adv Drug Deliv Rev. 2016;106(Part B):277-319. doi:10.1016/j.addr.2016.06.005

[3] Brayden DJ, Gleeson J, Walsh EG. A head-to-head multi-parametric high content analysis of a series of medium chain fatty acid intestinal permeation enhancers in Caco-2 cells. Eur J Pharm Biopharm. 2014;88(3):830-839. doi:10.1016/j.ejpb.2014.10.008

[4] Twarog C, Liu K, O'Brien PJ, Dawson KA, Fattal E, Illel B, Brayden DJ. A head-to-head Caco-2 assay comparison of the mechanisms of action of the intestinal permeation enhancers: SNAC and sodium caprate (C10). Eur J Pharm Biopharm. 2020;152:95-107. doi:10.1016/j.ejpb.2020.04.023

[5] Landrum G. RDKit: Open-Source Cheminformatics Software. 2022. https://www.rdkit.org/

[6] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD. 2016. doi:10.1145/2939672.2939785

[7] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS. 2017.

[8] van der Maaten L, Hinton G. Visualizing data using t-SNE. J Mach Learn Res. 2008;9:2579-2605.
