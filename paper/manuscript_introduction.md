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
