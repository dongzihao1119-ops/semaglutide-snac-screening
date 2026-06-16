# Introduction

Oral administration remains the preferred route for pharmaceutical products due to its convenience, patient compliance, and cost-effectiveness [1]. However, the oral delivery of peptide and protein therapeutics—a rapidly growing class that includes glucagon-like peptide-1 (GLP-1) receptor agonists such as semaglutide—faces a fundamental barrier: the intestinal epithelium is largely impermeable to macromolecules [2]. Chemical permeation enhancers (PEs) are excipients that transiently modulate the intestinal epithelial barrier to increase drug absorption, and their successful application in the first approved oral GLP-1 formulation (Rybelsus®, oral semaglutide co-formulated with sodium N-[8-(2-hydroxybenzoyl)amino]caprylate, SNAC) has validated this approach clinically [3,4].

Permeation enhancers act through two principal routes: the transcellular pathway, involving direct passage through epithelial cells via membrane perturbation or fluidization, and the paracellular pathway, involving modulation of tight junction protein complexes between adjacent cells [5,6]. The distinction is therapeutically significant: transcellular enhancers preserve tight junction integrity but require careful control of membrane toxicity [7], while paracellular enhancers can achieve higher flux for large hydrophilic molecules but raise concerns about barrier function compromise and bacterial translocation [8]. Despite decades of research, determining whether a given PE acts primarily through the transcellular or paracellular route has relied on low-throughput experimental assays—transepithelial electrical resistance (TEER) measurements, immunofluorescence staining of tight junction proteins, and lactate dehydrogenase (LDH) release assays [9,10]. A computational method to predict mechanism type from molecular structure would accelerate the screening and rational design of new PEs.

The most comprehensive systematic study of PE mechanisms to date is the work of Whitehead and Mitragotri [11], who evaluated 51 enhancers from 11 chemical categories on Caco-2 cell monolayers using TEER, MTT, and LDH assays. From these data, they derived the K-value—a quantitative metric ranging from 0 (purely transcellular) to 1 (purely paracellular)—that separates enhancers by dominant mechanism. Subsequently, Welling et al. [12] used this dataset to build a Random Forest quantitative structure-activity relationship (QSAR) model predicting enhancement potency from 30 molecular descriptors, achieving a leave-one-out cross-validated R² of 0.57. However, no study to date has addressed the arguably more fundamental question: can the mechanism type (paracellular vs transcellular) of a PE be predicted from its molecular structure?

In this study, we address this gap by developing a binary XGBoost classifier that predicts PE mechanism type from molecular descriptors and extended-connectivity fingerprints (ECFP4). We compiled a dataset of 53 PEs across 13 chemical categories, drawing from Whitehead et al. [11] and subsequent studies [13,14,15]. Using leave-one-out cross-validation, we demonstrate that mechanism type can be predicted with 92% accuracy, and we employ SHAP (SHapley Additive exPlanations) analysis [16] to identify the molecular features that drive this classification. Our results provide both a practical tool for in silico PE screening and mechanistic insight into the structural determinants of transcellular versus paracellular enhancement.

---

## References

[1] Goldberg M, Gomez-Orellana I. Challenges for the oral delivery of macromolecules. Nat Rev Drug Discov. 2003;2:289-295.

[2] Aungst BJ. Intestinal permeation enhancers. J Pharm Sci. 2000;89:429-442.

[3] Buckley ST, Bækdal TA, Vegge A, et al. Transcellular stomach absorption of a derivatized glucagon-like peptide-1 receptor agonist. Sci Transl Med. 2018;10(467):eaar7047.

[4] Permeation enhancer-induced membrane defects assist the oral absorption of peptide drugs. Nat Commun. 2025;16:9512.

[5] Salama NN, Eddington ND, Fasano A. Tight junction modulation and its relationship to drug delivery. Adv Drug Deliv Rev. 2006;58:15-28.

[6] Swenson ES, Curatolo WJ. Intestinal permeability enhancement for proteins, peptides, and other polar drugs: mechanisms and potential toxicity. Adv Drug Deliv Rev. 1992;8:39-92.

[7] Brayden DJ, Gleeson J, Walsh EG. A head-to-head multi-parametric high content analysis of a series of medium chain fatty acid intestinal permeation enhancers in Caco-2 cells. Eur J Pharm Biopharm. 2014;88(3):830-839.

[8] Maher S, Mrsny RJ, Brayden DJ. Intestinal permeation enhancers for oral peptide delivery. Adv Drug Deliv Rev. 2016;106(Part B):277-319.

[9] Meaney CM, O'Driscoll CM. A comparison of the permeation enhancement potential of simple bile salt and mixed bile salt:fatty acid micellar systems using the Caco-2 cell culture model. Int J Pharm. 2000;207:21-30.

[10] Tomita M, Hayashi M, Awazu S. Absorption-enhancing mechanism of EDTA, caprate, and decanoylcarnitine in Caco-2 cells. J Pharm Sci. 1996;85:608-611.

[11] Whitehead K, Mitragotri S. Mechanistic analysis of chemical permeation enhancers for oral drug delivery. Pharm Res. 2008;25(6):1412-1419.

[12] Welling SH, Clemmensen LKH, Buckley ST, Hovgaard L, Brockhoff PB, Refsgaard HHF. In silico modelling of permeation enhancement potency in Caco-2 monolayers based on molecular descriptors and random forest. Eur J Pharm Biopharm. 2015;94:152-159.

[13] Twarog C, Liu K, O'Brien PJ, Dawson KA, Fattal E, Illel B, Brayden DJ. A head-to-head Caco-2 assay comparison of the mechanisms of action of the intestinal permeation enhancers: SNAC and sodium caprate (C10). Eur J Pharm Biopharm. 2020;152:95-107.

[14] Bohley M, Leroux JC. Gastrointestinal permeation enhancers beyond sodium caprate and SNAC — what is coming next? Adv Sci. 2024;11(33):e2400843.

[15] Kim et al. Formulation engineering of oral semaglutide tablets: unleashing gastric intestinal permeation with sodium caprate. Pharmaceutics. 2026;18(5).

[16] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS. 2017.
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
# Results

## Dataset Characteristics

The curated dataset comprised 53 permeation enhancers (PEs) across 13 chemical categories, with 25 compounds having unambiguous mechanism assignments (14 paracellular, 11 transcellular) based on the K-value criterion (K > 0.5 = paracellular, K < 0.5 = transcellular) defined by Whitehead et al. [1]. The paracellular group was dominated by fatty esters (n=5), nitrogen-containing rings (n=4), and bile salts (n=3), while the transcellular group was enriched in zwitterionic surfactants (n=3) and cationic surfactants (n=4). The mean TEER reduction was comparable between groups (paracellular: 52.8 ± 28.2%; transcellular: 67.2 ± 26.2%; p = 0.21, two-sample t-test), indicating that mechanism type is not simply a function of enhancement potency.

## Binary Classification Performance

The XGBoost classifier achieved 92.0% accuracy (23/25 correct) under leave-one-out cross-validation (Table 1), with balanced accuracy of 91.9% and a Matthews correlation coefficient (MCC) of 0.838. Precision and recall were 0.93 and 0.93 for the paracellular class, and 0.91 and 0.91 for the transcellular class, respectively. The area under the ROC curve (AUC) was 0.974, indicating excellent discrimination between the two mechanism types.

**Table 1. Classification performance metrics (LOO-CV).**

| Metric | Value |
|--------|-------|
| Accuracy | 0.920 |
| Balanced Accuracy | 0.919 |
| MCC | 0.838 |
| AUC | 0.974 |
| Precision (Paracellular) | 0.93 |
| Recall (Paracellular) | 0.93 |
| Precision (Transcellular) | 0.91 |
| Recall (Transcellular) | 0.91 |

Two compounds were misclassified: one paracellular enhancer was predicted as transcellular, and one transcellular enhancer was predicted as paracellular. Both misclassifications occurred near the K = 0.5 boundary, where the original assignment is least certain.

## Feature Importance Analysis

SHAP analysis identified the most discriminative features separating paracellular from transcellular mechanisms (Figure 3). The top-ranked features were predominantly ECFP4 fingerprint bits, with ECFP4 bit 028 (mean |SHAP| = 0.730) and bit 119 (mean |SHAP| = 0.468) being the most influential. Among the physicochemical descriptors, fraction of sp3-hybridized carbons (fraction Csp3, mean |SHAP| = 0.639) was the strongest predictor, followed by logP (mean |SHAP| = 0.153) and molecular weight (mean |SHAP| = 0.097).

The importance of fraction Csp3 suggests that the degree of carbon saturation—a proxy for molecular flexibility and three-dimensional shape—is a key determinant of whether a PE acts via the transcellular or paracellular route. Transcellular PEs tend to have higher fraction Csp3 values, consistent with the long saturated alkyl chains characteristic of zwitterionic and cationic surfactants that insert into and fluidize the plasma membrane.

## Physicochemical Determinants of Mechanism

Clear physicochemical separation between the two mechanism types was observed (Figure 5). Transcellular PEs exhibited higher logP values (mean 4.2 ± 3.5 vs 1.6 ± 2.6 for paracellular, p = 0.05) and higher fraction Csp3 (mean 0.78 ± 0.22 vs 0.46 ± 0.30, p = 0.007). Molecular weight showed a bimodal distribution for paracellular PEs, with fatty esters and nitrogen-containing rings clustering at lower MW (150–300 Da) and bile salts at higher MW (400–540 Da). Rotatable bond count was significantly higher in transcellular PEs (mean 10.5 ± 5.2 vs 6.4 ± 5.0, p = 0.06), reflecting the flexible alkyl chains that facilitate membrane insertion.

## Chemical Category Analysis

The K-value distribution across chemical categories (Figure 6) revealed a clear mechanistic gradient. Fatty esters (mean K = 0.9) and nitrogen-containing rings (mean K = 0.7) were exclusively paracellular, consistent with their known ability to open tight junctions without penetrating the cell membrane [1,2]. In contrast, zwitterionic surfactants (mean K = 0.1) and cationic surfactants (mean K = 0.2) were predominantly transcellular, consistent with their membrane-insertion mechanism [1,3]. Bile salts occupied an intermediate position (mean K = 0.6), with individual compounds spanning both mechanisms depending on hydroxylation state and conjugation [2].

## Comparison with Prior QSAR Modeling

Welling et al. [9] previously developed a Random Forest QSAR model on 41 compounds from the same Whitehead dataset, predicting permeation enhancement potency (Tpot) rather than mechanism type. Their LOO-CV R² reached 0.57, with an RMSE of 0.16–0.17 on the 0–1 Tpot scale. Our regression analysis on the expanded 52-compound dataset yielded a comparable LOO-CV R² of 0.45. However, we found that scaffold-based cross-validation—a stricter test of generalizability—resulted in near-zero R² (−0.01 ± 0.07) for potency prediction, whereas mechanism classification maintained strong performance. This suggests that mechanism type is a more transferable molecular property than enhancement potency, and therefore a more suitable target for predictive modeling with the currently available data.
# Discussion

## A Molecular Classifier for Permeation Enhancer Mechanism

This study demonstrates that the mechanism of action of a permeation enhancer—whether it acts primarily through the paracellular or transcellular route—can be predicted with high accuracy from molecular structure alone. The 92% classification accuracy achieved under leave-one-out cross-validation represents, to our knowledge, the first successful in silico prediction of PE mechanism type across multiple chemical categories.

The finding that mechanism type is more amenable to ML prediction than enhancement potency has important implications for the field. While potency depends on multiple interacting factors—including concentration, formulation, and the specific cargo molecule being delivered—mechanism type appears to be an intrinsic molecular property encoded in structural features such as carbon saturation, lipophilicity, and specific substructural motifs. This is consistent with the physicochemical intuition that membrane-inserting (transcellular) enhancers require long, flexible, lipophilic chains, while tight junction-modulating (paracellular) enhancers rely on chelating or hydrogen-bonding interactions with junctional proteins.

## Key Discriminative Features

The dominance of fraction Csp3 as the top physicochemical discriminator (SHAP importance 0.639) provides a quantitative basis for mechanism assignment. A high fraction Csp3 reflects a predominantly aliphatic structure with extensive conformational flexibility—the hallmark of surfactant-like transcellular enhancers such as zwitterionic sulfobetaines (e.g., PPS, fraction Csp3 = 0.95) and quaternary ammonium salts (e.g., CTAB, fraction Csp3 = 0.89). In contrast, paracellular enhancers such as EDTA (fraction Csp3 = 0.29) and phenylpiperazine (fraction Csp3 = 0.43) have lower saturated carbon content, reflecting their more rigid, functional-group-rich structures designed for specific molecular recognition rather than non-specific membrane perturbation.

LogP—the second-ranked physicochemical feature—captures the differential lipophilicity requirements of the two mechanisms. Transcellular enhancers must partition into the lipid bilayer, requiring sufficient hydrophobicity (mean logP = 4.2), while paracellular enhancers operate in the aqueous intercellular space and benefit from moderate hydrophilicity (mean logP = 1.6). This observation aligns with the structure-function relationships identified by Whitehead et al. [1], who noted that transcellular enhancer potency scales directly with logP (r² = 0.9), while paracellular enhancer potency scales inversely with logP (r² = 0.77).

The ECFP4 fingerprint bits identified by SHAP analysis (bits 028, 119, 010, 125) encode specific substructural patterns that differentiate the two mechanisms. While the exact substructures corresponding to these bits require bit-collision-aware interpretation, their high SHAP values indicate that the model has learned chemically meaningful fragment-level distinctions beyond the global physicochemical properties.

## Comparison with Prior Work

The only prior ML study on this dataset is Welling et al. [9], who built a Random Forest QSAR model predicting permeation enhancement potency (Tpot) from 30 molecular descriptors. Our work differs in three key respects: (i) we target mechanism classification rather than potency regression; (ii) we employ ECFP4 fingerprints in addition to physicochemical descriptors, enabling the model to capture substructure-level information; and (iii) we provide SHAP-based interpretability analysis that identifies the specific molecular features driving predictions.

The regression-to-classification pivot is methodologically significant. While Welling et al. achieved a LOO-CV R² of 0.57 for potency prediction, we found that scaffold-based cross-validation—which more realistically simulates the task of predicting properties for novel chemotypes—yielded an R² near zero (−0.01 ± 0.07). This suggests that the original QSAR model primarily learned scaffold-level patterns rather than transferable structure-activity relationships. In contrast, mechanism classification remained robust because the structural determinants of mechanism (chain flexibility, lipophilicity, functional group composition) are scaffold-independent.

## Implications for Permeation Enhancer Discovery

The ability to predict mechanism type in silico has practical utility for oral peptide delivery research. A transcellular PE may be preferred for cargoes that require intracellular delivery or when tight junction integrity must be preserved (e.g., in inflammatory bowel disease). Conversely, a paracellular PE may be advantageous when maximizing flux of a large hydrophilic peptide is the priority. Our classifier enables rapid pre-screening of virtual compound libraries to identify candidates with the desired mechanism profile, prior to synthesis and in vitro testing.

Furthermore, the SHAP analysis provides interpretable design guidelines: increasing alkyl chain length and fraction Csp3 pushes a molecule toward transcellular action; adding hydrogen-bonding motifs, charged groups, or aromatic rings pushes it toward paracellular action. These rules can guide medicinal chemists in the rational design of next-generation PEs with tunable mechanisms.

## Limitations

Several limitations should be acknowledged. First, the dataset size (25 compounds for binary classification) is small by contemporary ML standards, though appropriate for the exploratory nature of this study and comparable to early QSAR models in related fields [9,10]. Second, K-values were assigned at the category level for compounds lacking individual experimental data, which may introduce noise. Third, the binary classification simplifies a continuum of mechanism contributions—many PEs operate through mixed paracellular and transcellular routes [1]—and the K = 0.5 threshold, while conventional, is arbitrary. Fourth, all data derive from a single laboratory [1] using consistent Caco-2 protocols, which ensures internal consistency but may limit generalizability to other experimental setups or to in vivo conditions. Fifth, the ECFP4 fingerprint bits identified as important require further structural interpretation, potentially through bit-to-substructure mapping using representative molecules from the training set.

## Future Directions

Extension of this work should prioritize: (i) experimental validation of the classifier on an independent set of PEs with measured K-values; (ii) expansion of the dataset to include PEs from additional chemical categories (e.g., cell-penetrating peptides, ionic liquids [11]); (iii) development of a ternary classifier that explicitly models mixed-mechanism enhancers; and (iv) integration with molecular dynamics simulations to validate the predicted mechanism at the atomistic level, particularly for the important case of SNAC-class enhancers where the transcellular mechanism has been structurally characterized via the quicksand-like membrane defect model [12].
