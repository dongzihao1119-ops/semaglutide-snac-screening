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
