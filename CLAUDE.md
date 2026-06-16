# CLAUDE.md — 司美格鲁肽 SNAC 衍生物吸收促进剂 AI 虚拟筛选

## ⚠️ 重要声明：数据验证状态

**Phase 0 文献验证已完成（2026-06-15）。**原始 docx 中的多个关键数据已被证伪或确认不存在。本文档已根据验证结果更新。被证伪的项目包括：

- ❌ "127 个 SNAC 类似物"——不存在。实际具名类似物 <20 个。
- ❌ "C10 + 对位 F = 2.3x 活性提升（2026.5）"——文献中找不到。
- ❌ "IF 6-12"——目标期刊 IF 实际在 3.9-5.7 范围。
- ❌ "SNAC 专利 2025 过期"——混淆了化合物专利和制剂专利。
- ✅ SNAC 300mg/片——确认。
- ✅ 树模型在小数据集优于 GNN——确认。
- ✅ 机制文献方向正确——但年份/期刊细节已被纠正。

**详见 Phase 0 验证结果部分。**

---

## Project Identity

**我们在做什么**：用机器学习对渗透促进剂（permeation enhancers, PEs）的作用机制进行系统性分类和 SAR 分析。核心产出是一个能以 92% 准确率预测 PE 是走 transcellular 还是 paracellular 路径的分子分类器。

**科学缺口**：尽管渗透促进剂已被研究数十年，其机制分类（transcellular vs paracellular）仍主要靠低通量实验（TEER、免疫荧光、LDH），缺乏基于分子结构的 in silico 预测工具。我们利用 Whitehead 2008 的 51 个 PE 系统筛选数据（含 EP/LP/K 机制指标）和 Maher 2016 的分类框架，首次建立了跨 13 个化学类别的 PE 机制预测模型。

**论文叙事**（2026-06-16 修正版）：
1. 口服肽类药物（如司美格鲁肽）依赖 PE 实现吸收，但 PE 的作用机制（transcellular vs paracellular）决定其安全性和适用场景
2. Whitehead 2008 在 Pharm Res 发表了 51 个 PE 的系统性 Caco-2 数据，但仅做了描述性统计分析
3. 我们收集并扩充了该数据集（53 个化合物），计算了 158 个分子描述符和 ECFP4 指纹
4. 二元 XGBoost 分类器在 LOO-CV 下以 **92% 准确率（MCC=0.84）** 预测 paracellular vs transcellular 机制
5. SHAP 分析揭示了关键区分特征：ECFP4 子结构（特定官能团组合）、fraction_csp3（饱和碳比例）、logP
6. 这是首次用 in silico 方法实现跨类别 PE 机制分类，可在合成前预筛 PE 候选分子

**不做的事情**：不发明新的 ML 架构、不要求实验验证、不声称临床适用性。

**成功的定义**：投稿 JCR Q1 计算化学/药学期刊（IF 4-6），核心卖点是 92% 准确率的机制分类器 + 可解释的 SHAP 分析。

---

## Team

- **Person A**（生物分子信息学）：数据收集与整理、特征工程（定制化 SNAC 特征）、分子动力学模拟、生物学结果解释、论文写作（Introduction、Results 生物部分、Discussion）
- **Person B**（计算机科学与技术）：项目基础设施、QSAR 模型训练与调优、大规模虚拟筛选、ChemBERTa 微调、NSGA-II 多目标优化、论文写作（Methods、Results 计算部分）

**共同负责**：文献调研、图表设计、稿件修改。

**工作模式**：两人共用一台电脑，完全同步工作。
- **不需要异步协作**：人在旁边，直接沟通。不需要 GitHub Issues、不需要 daily commit。
- **Git 用法的定位**：Git 用于版本控制和备份，不是协作沟通工具。每次完成一个有意义的阶段性成果后提交一次即可。
- **代码审核**：直接坐在一起看屏幕。一个人写、一个人看，当场发现当场改。
- **优势**：决策快、沟通零延迟、知识完全共享。利用好这个优势。

---

## Project Directory Structure

```
司美格鲁肽/
├── CLAUDE.md                          # 本文件
├── README.md                          # 项目简介（对外展示用）
├── .gitignore
├── environment.yml                    # Conda 环境精确锁定
├── requirements.txt                   # pip 备选方案
│
├── code/                              # 生产代码 — 所有 .py 脚本，可导入、可重复运行
│   ├── data/
│   │   ├── collect_snac_analogs.py    # 从文献中提取 127+ SNAC 类似物数据
│   │   ├── collect_negative_set.py    # 从 ChEMBL 提取 ~2000 阴性对照
│   │   ├── standardize_molecules.py   # RDKit 分子标准化流程
│   │   ├── compute_descriptors.py     # 计算 200 个特征：理化 + ECFP4 + 定制
│   │   └── fetch_zinc15.py            # 从 ZINC15 下载 1000 万子集
│   ├── qsar/
│   │   ├── train_qsar.py              # XGBoost/LightGBM 训练 + 交叉验证
│   │   ├── evaluate_qsar.py           # 评估指标、SHAP 分析、外部验证
│   │   ├── screen_zinc.py             # 对 1000 万化合物批量预测
│   │   └── filter_candidates.py       # 多条件筛选流程
│   ├── generative/
│   │   ├── finetune_chemberta.py      # 在 top 50 类似物上微调 ChemBERTa
│   │   ├── generate_novel.py          # 生成 10 万+ 新型衍生物
│   │   └── nsga2_optimize.py          # NSGA-II 多目标优化
│   ├── md/
│   │   ├── prep_systems.py            # 准备配体-蛋白-膜体系
│   │   ├── run_simulation.py          # OpenMM MD 生产模拟
│   │   └── analyze_trajectories.py    # 结合自由能、RMSD、关键接触分析
│   └── utils/
│       ├── config.py                  # 全局配置：路径、超参数、阈值常量
│       ├── logging_setup.py           # 统一日志
│       └── plot_utils.py              # 共享绘图函数
│
├── notebooks/                         # 探索分析 — 所有 .ipynb，不可被导入
│   ├── 01_data_exploration.ipynb      # 数据集初步探索
│   ├── 02_feature_analysis.ipynb      # 特征相关性、重要性分析
│   ├── 03_model_comparison.ipynb      # XGBoost vs LightGBM vs RF 对比
│   ├── 04_shap_interpretation.ipynb   # SHAP 特征重要性瀑布图/蜂群图
│   ├── 05_screening_results.ipynb     # 筛选命中可视化、化学空间分布
│   ├── 06_md_analysis.ipynb           # MD 轨迹可视化
│   ├── 07_chemberta_output.ipynb      # 生成分子分析
│   └── 08_paper_figures.ipynb         # 最终发表级图表
│
├── data/
│   ├── raw/                           # ⚠️ 永不修改 — 原始下载数据
│   │   ├── snac_analogs.smi           # 127+ SMILES + 活性标签
│   │   ├── negative_set.smi           # ~2000 阴性对照
│   │   └── zinc15_subset.smi          # 1000 万 ZINC15 化合物
│   ├── processed/                     # RDKit 标准化后、含特征的数据
│   │   ├── analogs_clean.csv          # 清洗后的 SNAC 类似物 + 描述符
│   │   ├── negatives_clean.csv        # 清洗后的阴性对照 + 描述符
│   │   ├── zinc15_features.h5         # 预计算特征（筛选用）
│   │   └── train_test_split.pkl       # 训练/验证/测试集索引（固定随机种子）
│   └── external/                      # 外部参考数据
│       ├── uniprot_tight_junction.fasta  # Occludin, Claudin 序列
│       ├── alphafold_structures/         # 下载的 AlphaFold 结构
│       └── snac_mechanism_2025.pdf       # 关键参考文献
│
├── models/                            # 训练好的模型文件（git-ignored，>10MB）
│   ├── qsar_activity_xgboost.json     # XGBoost 活性预测模型
│   ├── qsar_toxicity_lightgbm.txt     # LightGBM 毒性预测模型
│   ├── chemberta_finetuned/           # 微调后的 ChemBERTa checkpoint
│   └── scaler.pkl                     # 特征标准化器
│
├── results/                           # 所有模型输出
│   ├── qsar/
│   │   ├── cv_results.csv             # 交叉验证指标
│   │   ├── shap_values.csv            # SHAP 特征重要性
│   │   ├── external_test_predictions.csv
│   │   └── top_candidates.csv         # 1000 万筛选后的命中分子
│   ├── generative/
│   │   ├── generated_molecules.smi    # ChemBERTa 生成结果
│   │   └── pareto_front.smi          # NSGA-II 帕累托前沿前 100
│   └── md/
│       ├── rmsd_plots/
│       ├── binding_energies.csv
│       └── contact_maps/
│
├── paper/                             # 稿件和投稿材料
│   ├── manuscript.docx                # 正文（LaTeX 或 Word）
│   ├── figures/                       # 最终发表级图表（300+ DPI）
│   │   ├── fig1_pipeline_overview.png
│   │   ├── fig2_qsar_performance.png
│   │   ├── fig3_shap_analysis.png
│   │   ├── fig4_chemical_space.png
│   │   ├── fig5_md_results.png
│   │   ├── fig6_pareto_front.png
│   │   └── fig7_sar_summary.png
│   ├── supplementary/                 # 补充材料
│   ├── cover_letter.md
│   └── target_journals.md             # 目标期刊对比和策略
│
└── docs/                              # 过程文档
    ├── literature_review.md           # 关键文献笔记
    ├── data_dictionary.md             # 所有 CSV 的列定义
    ├── environment_setup.md           # 环境配置步骤
    ├── compute_notes.md               # GPU 租用日志和费用记录
    └── meeting_notes/                 # 每周进度记录
```

---

## Phased Plan (Revised June 2026 — Mechanism Classification Paper)

### ✅ Phase 1: Dataset + Pipeline (June 1-16, 2026 — COMPLETE)
53 compounds, 13 categories, classification model at 92% accuracy.

### Phase 2: Paper Writing & Refinement (June 17 – August 2026)
**Target: Submit to Journal of Cheminformatics (IF 5.7) by end of August.**

| 周次 | 任务 | 产出 |
|------|------|------|
| Jun 17-23 | Write Methods section | 完整方法描述 |
| Jun 24-30 | Generate all 7 figures | 300+ DPI PNG |
| Jul 1-14 | Write Results + Discussion | 初稿 |
| Jul 15-21 | Write Introduction | 完整初稿 |
| Jul 22-31 | Internal review + revision | 第二稿 |
| Aug 1-14 | Polish + format + cover letter | 投稿就绪 |
| Aug 15-31 | SUBMIT | 确认邮件 |

### ❌ Cancelled (no longer needed)
- Phase 2 (old): ZINC15 virtual screening — not needed for classification paper
- Phase 3 (old): MD simulations — not needed for classification paper
- Phase 4 (old): ChemBERTa generative design — not needed for classification paper

### Phase 0: 文献验证与数据确认（June 2026 Week 1 — 所有工作的前提）
**目标：逐条验证 docx 中的所有关键声明，把 `[待验证]` 变成确认事实或划掉。本阶段不做任何编码。**

**为什么需要 Phase 0**：原始 docx 只提供思路方向，里面的具体数字、文献引用、数据规模均不可直接采信。如果基础假设错了，整个项目会白做。

#### Phase 0 验证结果（2026-06-15 完成）

> 以下结果通过 WebSearch/WebFetch 完成。**高置信度项目仅标注，红色标记需特别注意的事项。**

| # | 声明 | 验证结果 | 置信度 |
|---|------|---------|--------|
| 0.1 | SNAC 300mg/片 | ✅ **确认**。Rybelsus 所有规格（3/7/14mg semaglutide）均含 300mg SNAC，PIONEER III 期验证。 | 高 |
| 0.2 | SNAC 专利 2025 年过期 | ❌ **严重不准确**。Semaglutide 化合物专利在印度 2026.3 过期，但 SNAC 制剂专利：印度至 2031、美国至 ~2033、欧盟至 2039。docx 混淆了化合物专利和制剂专利。 | 高 |
| 0.3 | 机制文献在 2025 Nature Communications | ⚠️ **部分准确**。2025 NC 确有一篇 "Permeation enhancer-induced membrane defects assist the oral absorption of peptide drugs"（Article 9512），提出 quicksand-like 膜缺陷机制。但 SNAC 机制的首个完整解析是 **Buckley et al. (2018) Science Translational Medicine**（DOI: 10.1126/scitranslmed.aar7047），docx 漏掉了这篇奠基性文献。 | 高 |
| 0.4 | 127+ SNAC 类似物有活性数据 | ❌ **未找到证据**。已知 SNAC 化学系列主要包括 SNAC 本身、**5-CNAC** 和 **4-CNAB** 等少数几个命名类似物。未找到任何包含 127 个 SNAC 类似物系统性活性数据的文献。这个数字很可能是 docx 编造的。**实际能找到的具名类似物可能不到 20 个。** | 高 |
| 0.5 | 2026.5 C10+对位 F = 2.3x 活性提升 | ❌ **高度可疑，可能不存在**。2026.5 Pharmaceutics 确有 Kim et al. 论文比较 C10 vs SNAC，但结论是 C10 "等效"（not 2.3x 更好）。**无任何论文提到对位氟取代或 2.3 倍提升。** 这个声称很可能是 docx 编造的。 | 高 |
| 0.6 | ChEMBL 有 ~2000 个 absorption enhancer 阴性对照 | ⚠️ **未直接确认数量**。ChEMBL 确实有渗透性 assay 数据，但具体 absorption enhancer 标签的化合物数量需要在 ChEMBL 网页上直接查询。2000 这个数字暂无独立来源。建议用 agent-browser 直接在 ChEMBL 查询。 | 中 |
| 0.7 | ZINC15 有 1000 万+ drug-like 化合物 | ⚠️ **近似合理但需精确化**。ZINC 数据库 wiki 称总规模约 2.3 亿物质，drug-like 子集通常在数千万级别。1000 万作为预筛选后的子集是合理的。但需去 ZINC15 网站确认最新统计。 | 中 |
| 0.8 | 目标期刊 IF 6-12 | ❌ **全部低于 docx 声称值**。2024 JCR（2025.6 发布）：J. Cheminformatics **5.7**（docx 说 7）、EJPS **4.7**（docx 说 6）、JCIM **5.3**（docx 说 6）、Mol. Pharmaceutics **4.5**（docx 说 5）、Sci. Reports **3.9**（docx 说 4）。**无一达到 IF 6。** | 高 |
| 0.9 | XGBoost/LightGBM 在小数据集上优于 GNN | ✅ **基本确认，但需细化**。Boehringer Ingelheim/Merck 2024 研究（JCIM 2025）定量确定了交叉点：**<500 数据点 → RF/XGBoost 胜；500-2000 → 灰色区域；>2000 → GNN 胜。** 同时 2026 BBB 基准测试发现 RF 在外部 scaffold-split 验证上 AUC 最高（0.841），高于 GATv2（0.801）。 | 高 |
| 0.10 | 文献中实际能找到多少可用化合物？ | ❌ **远少于 docx 声称**。综合 0.4 和 0.5 结果：SNAC 化学系列具名类似物仅 3-5 个。但 Bohley & Leroux (2024) 综述表明渗透促进剂涵盖多个化学类别（中链脂肪酸、酰基肉碱、胆盐、EDTA 等）。**从跨类别化合物中收集活性数据是可行的，但不会集中在 SNAC 系列。** | 高 |

### 🚨 Phase 0 关键发现

**docx 存在多处数据编造或严重失实**：

1. **"127 个 SNAC 类似物"** —— 不存在。实际具名类似物 <20 个。
2. **"C10 + 对位 F = 2.3x 活性提升"** —— 文献中找不到。这是论文叙事的核心支柱，塌了。
3. **"IF 6-12"** —— 提议的目标期刊 IF 均低于 6。最高的是 J. Cheminformatics 5.7。
4. **"SNAC 专利 2025 年过期"** —— 混淆了化合物专利和制剂专利。SNAC 制剂专利在主要市场持续到 2030s。

**项目可以继续，但必须大幅调整**：
- 论文叙事不能依赖"127 个类似物"和"C10+F=2.3x"这两个不存在的数据点
- 数据集建设策略需要改变：不应限于 SNAC 系列，而应扩大到跨类别的渗透促进剂
- 目标期刊需要重新选择（IF 4-6 区间更现实，而非 6-12）
- 建议的替代论文叙事方向见下方

### Phase 0 退出决定

- [ ] Person A 确认理解上述验证结果
- [ ] Person B 确认理解上述验证结果
- [ ] 双方同意调整后的项目方向（见下方 Phase 0 调整）

---

## ⚠️ Phase 0 项目方向调整（基于验证结果）

### 论文叙事重建

**原叙事（已崩塌）**：127+ SNAC 类似物 → QSAR → 1000 万筛选 → 找到活性 1.5x 的分子 → 发表 IF 6-12

**新叙事（建议）**：

1. **背景问题不变**：口服司美格鲁肽依赖 SNAC 300mg，生物利用度 <1%，寻找更好的吸收促进剂有价值
2. **数据策略改变**：从"SNAC 系列专项"转为**跨类别渗透促进剂的系统性 AI 筛选**
   - 渗透促进剂已有多类：中链脂肪酸（C8/C10）、酰基肉碱、胆盐、EDTA 等
   - Bohley & Leroux (2024) *Advanced Science* 综述提供了完整的化学空间
   - 收集跨类别 PE 的活性数据（Caco-2 TEER/Papp），而非仅限于 SNAC 系列
3. **锚定文献**：Buckley et al. (2018) *Science Translational Medicine* + 2025 *Nature Communications* Article 9512 作为机制基础
4. **创新点重新表述**：
   - 首个跨类别 PE 活性的系统性 QSAR 模型
   - 将 quicksand-like 膜缺陷机制与分子描述符关联
   - 虚拟筛选发现新型骨架的 PE 候选分子（非 SNAC 衍生物）
5. **ChemBERTa 生成方向调整**：不限于 SNAC 母核改造，而是跨类别的全新 PE 分子从头设计

### 目标期刊调整

| 排名 | 期刊 | IF (2024 JCR) | 理由 |
|------|------|--------------|------|
| 1 | **Journal of Cheminformatics** | 5.7 | 计算化学+AI，最契合；Q1 |
| 2 | **JCIM** | 5.3 | ACS 期刊，计算方向强；Q1 |
| 3 | **EJPS** | 4.7 | PE 文献集中地；Q1 药学 |
| 4 | **Molecular Pharmaceutics** | 4.5 | 药物递送方向；Q1 |
| 5 | **Scientific Reports** | 3.9 | 保底；Q1 综合 |

**现实预期**：IF 5-6 为目标，IF 4-5 可接受。docx 说的 IF 6-12 在当前期刊格局下不现实。

### 数据策略调整

- **不再追求**"127 个 SNAC 类似物"——这个数字不存在
- **改为**跨类别 PE 数据集：SNAC 类（5-10 个）+ 中链脂肪酸类（10-20 个）+ 胆盐类（10-20 个）+ 酰基肉碱类（5-10 个）+ 其他（10-20 个）
- 预期总量：**50-100 个已知 PE 活性化合物**，而非声称的 150 个阳性 + 2000 个阴性
- 如数据量不足 500，QSAR 使用 RF/XGBoost（有 benchmark 支持），不碰 GNN
- 数据量达到后才考虑 GNN

### 其他调整

- SNAC 专利过期 ≠ 仿制药无门槛（制剂专利仍在），但**不影响我们做计算研究的合理性**——我们做的是筛选方法，不是仿制
- C10 论文（Kim et al. 2026 Pharmaceutics）可以作为"领域在积极探索 SNAC 替代方案"的证据，但不引用不存在的 2.3x 数据
- 分子动力学模拟的靶点从 tight junction 蛋白改为**脂质膜体系**（与 2025 NC 的 quicksand 机制一致）

---

### Phase 1: Foundation (June 2026 — Month 1)
**目标：在已有数据上跑通 QSAR 模型，搭好项目骨架。**
**前提：Phase 0 已完成，所有 `[待验证]` 声明已确认或修改。**

| 周次 | 交付物 | 负责人 | 验证方式 |
|------|--------|--------|----------|
| W1 (Jun 1-7) | Git repo 初始化、conda 环境可用、RDKit+XGBoost 可导入 | B | `python -c "from rdkit import Chem; print('OK')"` |
| W1 (Jun 1-7) | 文献调研：收集全部 127+ SNAC 类似物的 SMILES 和活性数据 | A | `data/raw/snac_analogs.smi` 含 ≥100 条 |
| W2 (Jun 8-14) | RDKit 分子标准化流程 | A | `code/data/standardize_molecules.py` 无错误运行 |
| W2 (Jun 8-14) | ChEMBL 阴性对照提取（~2000 化合物）| A | `data/raw/negative_set.smi` |
| W3 (Jun 15-21) | 特征计算：200 个描述符（理化 + ECFP4 + 定制）| A | `data/processed/analogs_clean.csv` 含 200 列特征 |
| W3 (Jun 15-21) | 基线 QSAR 模型：XGBoost + 5 折 CV | B | `results/qsar/cv_results.csv`，训练集 R² ≥ 0.75 |
| W4 (Jun 22-28) | 模型对比：XGBoost vs LightGBM vs Random Forest | B | Notebook `03_model_comparison.ipynb` |
| W4 (Jun 22-28) | 最佳模型 SHAP 分析 | B | Notebook `04_shap_interpretation.ipynb` |
| W4 (Jun 22-28) | **Phase 1 里程碑**：QSAR 模型测试集 R²≥0.85（活性）、R²≥0.80（毒性）| Both | `results/qsar/external_test_predictions.csv` |

**Phase 1 退出标准**：QSAR 模型在外部测试集上达到 R² ≥ 0.85（活性）和 R² ≥ 0.80（毒性）。如果不达标，延长 1-2 周 —— 模型不行不进下一阶段。

### Phase 2: Large-Scale Virtual Screening (July 2026 — Month 2)
**目标：筛选 1000 万 ZINC15 化合物，锁定前 500-1000 个候选分子。**

| 周次 | 交付物 | 负责人 | 验证方式 |
|------|--------|--------|----------|
| W5 (Jun 29-Jul 5) | 下载 ZINC15 子集（1000 万 drug-like 化合物）| B | `data/raw/zinc15_subset.smi`，文件大小验证规模 |
| W5 (Jun 29-Jul 5) | 批量特征计算流程优化（并行处理）| B | 100 万化合物 <2 小时完成特征计算 |
| W6 (Jul 6-12) | 本地 CPU 对 1000 万化合物批量预测（多天跑批）| B | 全部预测结果写入 `results/qsar/` |
| W7 (Jul 13-19) | 多条件筛选：活性>1.5x SNAC、CC50>1000μM、pKa 5.5-7.0、logP 1.0-3.0 | B | `results/qsar/top_candidates.csv` <1000 条 |
| W8 (Jul 20-26) | 按 Murcko 骨架聚类，分析化学多样性 | A | Notebook `05_screening_results.ipynb` |
| W8 (Jul 20-26) | 选定前 100 个候选分子进入 MD 验证 | Both | `results/qsar/md_candidates_100.csv` |

**Phase 2 退出标准**：100 个结构多样、预测活性 >1.5x SNAC 的候选分子。如果 <200 个分子通过筛选，放宽 pKa 范围至 5.0-7.5 或 logP 至 0.5-3.5。

### Phase 3: Molecular Dynamics Validation (August–September 2026 — Months 3-4)
**目标：对前 100 个候选分子进行 MD 模拟验证，获取结合/插入能力数据。**

| 周次 | 交付物 | 负责人 | 验证方式 |
|------|--------|--------|----------|
| W9 (Jul 27-Aug 2) | 搭建 OpenMM 模拟流程 | A | 1 个测试体系跑通 |
| W9 (Jul 27-Aug 2) | 下载 AlphaFold 结构：occludin、claudin | A | PDB 文件到位 |
| W10-11 (Aug 3-16) | 首批 20 个候选分子 MD（学习流程、调试）| A | 每个体系 50ns，RMSD 稳定 |
| W11 (Aug 10-16) | 云 GPU 并行 MD 环境搭建（AutoDL）| B | 多个模拟同时运行 |
| W12-15 (Aug 17-Sep 13) | 剩余 80 个候选分子 MD（云 GPU 并行）| A（执行）B（运维）| 全部 100 个体系完成 |
| W16 (Sep 14-20) | 轨迹分析：结合自由能（MM/PBSA）、RMSD、关键接触 | A | `results/md/binding_energies.csv` |
| W16 (Sep 14-20) | 按 MD 预测的结合能力 + 插入能力综合排名 | Both | 最终排名列表 |

**Phase 3 退出标准**：至少 80/100 个候选分子完成模拟（20% 失败率可接受，力场兼容性问题）。至少 10 个候选分子的 MD 预测结合力优于 SNAC 基线。

**GPU 预算**：AutoDL A100 约 ¥3-5/小时，100 个体系 × 50ns ≈ 50-100 GPU 小时，预算 ¥500-1000。并行运行以控制时间。

### Phase 4: Generative AI Molecular Design (September–October 2026 — Months 4-5) ⚠️ 加分项/可放弃
**目标：生成 ZINC15 数据库中不存在的新型 SNAC 衍生物。本阶段是可选的。**

**触发条件**：Phase 1-3 按计划推进 AND 距论文截止日期 >4 周。如果 Phase 3 延迟 >2 周，跳过 Phase 4 直接进入 Phase 5。

| 周次 | 交付物 | 负责人 | 验证方式 |
|------|--------|--------|----------|
| W17 (Sep 21-27) | 准备 ChemBERTa 微调数据集（top 50 类似物 + 结构域标签）| A | 带标签的 SMILES 数据集 |
| W17-18 (Sep 21-Oct 4) | 在 HuggingFace 上微调 ChemBERTa 2.0 | B | `models/chemberta_finetuned/`，loss 曲线合理 |
| W19 (Oct 5-11) | 约束生成 10 万+ 新型分子 | B | `results/generative/generated_molecules.smi` |
| W20 (Oct 12-18) | NSGA-II 多目标优化（活性、毒性、可合成性）| B | 确定帕累托最优集合 |
| W20 (Oct 12-18) | 通过 QSAR 模型交叉验证 top 50 生成分子 | A | 新型分子活性预测值合理 |

**Phase 4 放弃标准**：微调 3 天 GPU 时间内 loss 不收敛，或 >90% 生成分子有效性不合格 → 放弃，在论文中注明 "generative design attempted but limited by training data size"。

### Phase 5: Paper Writing and Submission (September–November 2026 — Months 4-6)
**目标：完成稿件，投稿。**

| 周次 | 交付物 | 负责人 | 验证方式 |
|------|--------|--------|----------|
| W16 (Sep 14-20) | **与 Phase 3 并行开始写 Methods** | Both | Methods 初稿 |
| W18 (Sep 28-Oct 4) | 全部图表完成（7 张主图 + 补充材料）| Both | 300+ DPI PNG 在 `paper/figures/` |
| W19-20 (Oct 5-18) | 完整初稿：Introduction、Methods、Results | Both | `paper/manuscript.docx` 完整 |
| W21 (Oct 19-25) | 内部审稿和修改 | Both | 互相审读对方的章节 |
| W22 (Oct 26-Nov 1) | 语言润色、图表定稿、SI 整理 | Both | 稿件就绪 |
| W23 (Nov 2-8) | Cover letter、期刊格式调整、终审 | Both | 投稿资料包完整 |
| W24 (Nov 9-15) | **投稿** | Both | 确认邮件 |

**写作原则**：不等实验全部做完再开始写。Phase 2 期间开始写 Methods，Phase 3 期间开始写 Introduction。防止最后一个月灾难性加班。

---

## Technical Architecture

### Current Pipeline (implemented)

```
Whitehead 2008 51 PEs    Maher 2016 review       Brayden 2014 MCFA
        |                       |                      |
        +-----------------------+----------------------+
                            |
                    data/raw/pe_compounds.csv (56 compounds)
                            |
                    RDKit standardization
                            |
                    158 features per compound:
                    20 physchem + 128 ECFP4 + 7 custom + 3 metadata
                            |
              +-------------+-------------+
              |                           |
     Binary Classifier              QSAR Regression
     (paracellular vs              (TEER reduction %
     transcellular)                prediction)
     XGBoost, LOO-CV               XGBoost, LOO-CV
     Accuracy: 92%                 R²: 0.45
     MCC: 0.84                     (informative but
     ✅ PUBLISHABLE                 limited by data)
              |
         SHAP Analysis
    Top features: ECFP4 bits,
    fraction_csp3, logP, MW
```

### Paper Figure Plan (7 figures)

1. **Fig 1: Dataset overview** — Chemical space (t-SNE) of 53 PEs colored by mechanism type
2. **Fig 2: Classification performance** — Confusion matrix + ROC curve for binary classifier
3. **Fig 3: SHAP beeswarm** — Top 20 features driving paracellular vs transcellular prediction
4. **Fig 4: Key ECFP4 substructures** — RDKit depiction of the fingerprint bits that most differentiate mechanisms
5. **Fig 5: Physicochemical trends** — logP vs fraction_csp3 scatter, colored by mechanism, with decision boundary
6. **Fig 6: Chemical category analysis** — Bar chart of mechanism distribution across 13 categories
7. **Fig 7: Case studies** — 4 representative PEs (PPS, EDTA, SNAC, C10) with annotated structures showing key discriminative features

### Paper data provenance

- **Training data**: Whitehead et al. (2008) Pharm Res 25:1412-1419 — 51 compounds, Caco-2 TEER/MTT/LDH
- **QSAR precedent**: Welling et al. (2015) EJPB 94:152-159 — Random Forest on same dataset, 41 compounds
- **Classification framework**: Maher et al. (2016) Adv Drug Deliv Rev 106:277-319 — 6-class PE framework
- **Novelty vs Welling 2015**: They did regression (potency prediction); we do classification (mechanism prediction). They used only molecular descriptors; we add ECFP4 fingerprints + SHAP interpretation.

---

## Data Plan

> ⚠️ 以下所有数字（127+、2000、1000 万等）均来自 docx，**在 Phase 0 完成验证前不可采信**。Phase 0 结束后更新本节。

### 数据源详情

#### 1. SNAC 类似物（`[待验证]` 127+ `[待验证]` 个化合物）
- **主要来源** `[待验证]`：European Journal of Pharmaceutical Sciences（2023-2026）、Journal of Controlled Release（2023-2026）
- **提取方式**：从论文表格/补充材料手动提取，或使用 WebPlotDigitizer 从图表中提取
- **每个化合物必要字段**：SMILES、Caco-2 TEER 降低率（%）、司美格鲁肽 Papp（cm/s）、CC50（μM）、pKa（实验值优先，无则计算值）、logP（实验值优先）
- **目标**：≥100 个化合物有完整的活性数据（`[待验证]`：实际文献中能找到多少？Phase 0 摸底后更新）
- **备用方案**：如果文献中找到的 <100 个，从 ChEMBL "absorption enhancer" 标签 + PubChem bioassay 数据补充

#### 2. 阴性对照（`[待验证]` ~2000 `[待验证]` 个化合物）
- **来源** `[待验证]`：ChEMBL 34，搜索 "absorption enhancer" assay 中标注为 inactive/negative 的化合物
- **阴性标准**：TEER 降低率 <10% 或 Papp < 1e-6 cm/s 或 assay 标为 "inactive"
- **多样性要求**：MW 范围（150-400 Da）和 logP 范围（-1 到 5）必须与阳性集一致，避免模型仅凭分子大小就能区分
- `[待验证]`：ChEMBL 中 "absorption enhancer" 标签实际有多少化合物？Phase 0 确认后更新

#### 3. ZINC15 筛选库（`[待验证]` 1000 万 `[待验证]` 化合物）
- **来源** `[待验证]`：ZINC15 drug-like subset（http://zinc15.docking.org/）
- **下载方式**：使用 ZINC15 tranche 文件或 ZINC API
- **预筛选标准**（QSAR 预测前）：MW 150-400、logP -1 到 5、可旋转键 ≤10、TPSA < 140
- **存储**：特征矩阵用 HDF5，SMILES 用压缩文本。1000 万化合物约 2-5GB

### 数据质量规则
- **所有 SMILES 必须通过 RDKit 检验**（`Chem.SanitizeMol`）——失败则拒绝并记录
- **去除盐型**：只保留最大片段（`Chem.GetMolFrags`）
- **去除重复**：按 canonical SMILES 去重（InChI key 备用）
- **所有活性值必须有单位**——不接受无单位的裸数字
- **确定性数据分割**：全项目统一 `random_state=42`

---

## Division of Work

### Person A（生物分子信息学）
**核心职责**：
1. **数据收集**（Phase 1）：找到并提取所有 SNAC 类似物数据。理解生物学含义，判断哪些数据可靠。
2. **特征工程**（Phase 1）：设计 20 个 SNAC 专属特征（烷基链长度检测、取代基位置识别、酰胺键描述符）。你知道什么化学特征重要。
3. **分子动力学模拟**（Phase 3）：搭建、运行、分析所有 MD 模拟。这是最接近生物学的计算任务。
4. **生物学解释**（Phase 5）：写 Introduction、Results 生物部分、Discussion。解释为什么这些 SAR 规则在化学上合理。
5. **图表设计**（Phase 5）：化学结构图、结合模式可视化、SAR 总结图。

**需要学习**：
- RDKit 基础（1-2 天）：分子操作教程
- OpenMM 基础（3-5 天）：蛋白质-配体模拟教程
- SHAP 可解释性（1 天）：理解特征重要性的含义

### Person B（计算机科学与技术）
**核心职责**：
1. **基础设施**（Phase 1）：Git repo、conda 环境、项目结构、云 GPU 配置
2. **QSAR 模型训练**（Phase 1）：模型训练、超参数调优、交叉验证、模型对比
3. **大规模筛选**（Phase 2）：1000 万化合物的批量预测、高效数据加载、并行化
4. **ChemBERTa 微调**（Phase 4）：HuggingFace 训练、生成流程、NSGA-II 优化
5. **Methods 写作**（Phase 5）：计算方法部分、所有代码文档
6. **可复现性**（全程）：版本锁定、随机种子设置、配置管理

**需要学习**：
- RDKit 基础（1 天）：指纹计算、描述符计算 API
- 药物发现概念（2-3 天）：pKa、logP、TEER 的含义及为什么重要
- 分子动力学概念（1 天）：MM/PBSA 做什么（不需要会跑）

---

## Publication Strategy

### 目标期刊（按优先级排序）

> ⚠️ IF 数据需在 Phase 0 确认最新值，期刊选择需结合文献调研过程中对领域发文偏好的实际了解来调整。

| 排名 | 期刊 | IF `[待验证]` | 理由 | 投稿要求 |
|------|------|-------------|------|----------|
| 1 | Journal of Cheminformatics | ~7 | 计算化学 + AI，完美契合 | 代码+数据公开 |
| 2 | European Journal of Pharmaceutical Sciences | ~6 | SNAC 文献集中地 | 接受 in-silico 研究 |
| 3 | Journal of Chemical Information and Modeling (JCIM) | ~6 | ACS 期刊，计算方向 | 期望较强的 MD 部分 |
| 4 | Molecular Pharmaceutics | ~5 | 药物递送方向 | 强调生物学相关性 |
| 5 | Scientific Reports | ~4 | 保底，接收率高 | 科学合理即可，新颖性要求低 |

**策略**：先投 Journal of Cheminformatics。被拒则改投 EJPS，再被拒投 JCIM，最后 Scientific Reports。

### 图表要求（7 张主图）

1. **Pipeline 总览图**：类 Graphical Abstract 的工作流
2. **QSAR 性能图**：预测 vs 实际散点图，标注 R² 和 RMSE
3. **SHAP 分析图**：top 20 特征的蜂群图，附化学解释
4. **化学空间图**：t-SNE/UMAP 着色标注预测活性，突出 SNAC 位置
5. **MD 结果图**：RMSD 稳定性 + 结合自由能柱状图（top 10 vs SNAC 基线）
6. **SAR 总结图**：标注 SNAC 结构在哪个位置做何种改造能提升哪种性质（如 Phase 4 完成则展示帕累托前沿）
7. **Case Study 图**：3-4 个最佳候选分子 + 2D 结构 + 预测性质表格

### 写作时间线
- **Methods**：Week 10 开始（Phase 3 期间）——计算方法步骤已经确定
- **Introduction**：Week 12 开始 —— 起草 "gap" 和 "our approach" 段落
- **Results**：每个 Phase 产出最终结果时立即写
- **Discussion**：Week 18，全部结果确定后
- **至少 2 轮内部审稿**

---

## Risk Management

### 高风险项

| 风险 | 概率 | 影响 | 缓解方案 |
|------|------|------|----------|
| QSAR 测试集 R² < 0.85 | 中 | 高（模型不好论文发不了）| 延长 Phase 1。尝试图神经网络（GNN）。从专利中收集更多数据。如果毒性模型差但活性模型好，可以放弃毒性模型。 |
| ZINC15 下载/处理失败 | 低 | 中 | 提前下载多个 tranche。用 PubChem 做备份（~1 亿化合物）。降到 100 万子集的备选方案。 |
| MD 模拟不稳定（体系崩溃）| 中 | 中 | 准备调试清单。先用 10ns 短模拟测试。必要时简化膜模型。 |
| ChemBERTa 生成大量无效 SMILES（>90%）| 高 | 低（Phase 4 是加分项）| 放弃 Phase 4，在论文中注明 "exploratory"。对发表不致命。 |
| 云 GPU 租不到或超预算 | 低 | 中 | 提前注册 AutoDL 账号 + 备用平台。总预算上限 ¥2000。如果超预算，MD 从 100 个减到 50 个。 |
| 其中一人 2 周以上无法工作 | 低 | 高 | 双方充分文档化代码，对方可以接手关键路径任务。 |

### 时间线风险

**最大风险是 Month 5 的冲突**：MD 分析 + 论文写作 + 可能修改全部挤在一起。
- **缓解**：Month 4 开始写论文，不等到 Month 5
- Phase 4 作为弹性缓冲区 —— 任何环节延迟，直接跳过
- **硬性截止日**：如果 10 月 1 日 Phase 3 未完成，跳过 Phase 4 立即启动论文

### 科学严谨性风险
- **数据泄露**：训练/测试集分割必须按化合物骨架（scaffold-based split），不能用随机分割。随机分割在同系物很多时会严重高估模型性能。用 Murcko scaffold 聚类确保测试集骨架不在训练集中出现。
- **127 个化合物的小数据集过拟合**：缓解方案 → 只用树模型（小数据更好）；充分的交叉验证；用 ChEMBL 不同 assay 来源的数据做外部验证。
- **MD 力场精度**：AMBER/OpenMM 力场可能对特殊 SNAC 衍生物（尤其含卤素）建模不准确。缓解方案 → 先用已知 SNAC 实验数据做基线验证。在论文中诚实报告力场局限。

---

## Reproducibility & Open Science

### 环境锁定
- **主要**：`environment.yml`，每个包精确版本号
- **备选**：`pip freeze > requirements.txt`
- **验证**：新建 conda 环境后 `import rdkit; import xgboost; import openmm` 必须无错误
- **OS**：macOS（开发）+ Linux（GPU 服务器）。代码必须在两者上都可运行。

### 随机种子（确定性训练）
```python
# config.py
RANDOM_SEED = 42
import random
random.seed(RANDOM_SEED)
import numpy as np
np.random.seed(RANDOM_SEED)
import torch
torch.manual_seed(RANDOM_SEED)
# XGBoost/LightGBM 接受 seed 参数
```

### 数据溯源
- 每个 `data/raw/` 文件必须有一个对应的 `.source` 文本文件，记录：URL、下载日期、查询参数、许可证
- 所有数据转换用脚本完成（不手动改 Excel）。可追溯：raw → 标准化 → 特征 → 模型输入。

### 代码发布计划
- **开发期间**：GitHub private repo（仅限两人）
- **投稿时公开**：MIT license（代码），CC-BY 4.0（数据）
- 生成 Zenodo DOI 关联论文引用的确切版本
- 公开前清理 repo：删除死代码、添加 docstring、写 README

### 不开源的内容
- 云 GPU 凭证（.env 文件，加入 .gitignore）
- ZINC15 原始下载（太大 —— 提供下载脚本）
- `docs/meeting_notes/` 个人笔记

---

## Coding Standards

### Notebook vs Script 规则（关键）

```
┌──────────────────────────────────────────────────────┐
│  NOTEBOOKS (.ipynb)：探索、可视化、论文图表、        │
│  一次性分析。永远不被 import。输出结果随 git 提交。   │
│                                                      │
│  SCRIPTS (.py)：生产代码、数据处理流程、模型训练、    │
│  批量筛选。可导入、可测试、可复现。提交时包含         │
│  函数 docstring。                                    │
└──────────────────────────────────────────────────────┘
```

**规则**：运行超过 3 次的东西写成 script。产出论文图表的写成 notebook（内部调用 script 的函数）。Notebook 从 `code/` 导入函数并做可视化——不包含核心逻辑。

### Git Workflow
- **分支命名**：`phase1/data-curation`、`phase1/qsar-training`、`phase2/screening` 等
- **Commit message**：英文，祈使语气。例如 "Add RDKit standardization for SNAC analogs"
- **大文件不入库**：.gitignore 排除 `models/`（>10MB）、`data/raw/zinc15_*`（>100MB）、`results/md/trajectories/`（.dcd/.xtc 文件）

### .gitignore 模板
```
models/
data/raw/zinc15*
results/md/trajectories/
*.dcd
*.xtc
*.pdb
.env
__pycache__/
.ipynb_checkpoints/
.DS_Store
```

### 文档规范
- **每个 script**：模块 docstring 说明 Purpose、Input、Output、Example usage
- **每个函数**：Args 和 Returns（Google 风格 docstring）
- **每个 notebook**：顶部 markdown cell 说明目的和预期运行时间
- **Config**：所有超参数、路径、阈值在 `code/utils/config.py` 中，禁止硬编码

### Code Review 清单（双方通用）
- [ ] 可导入：`from code.qsar.train_qsar import train_model` 能跑
- [ ] 确定性：相同输入 → 相同输出
- [ ] 有日志：关键步骤打印/记录进度
- [ ] 可重跑：中断后重新运行不会损坏数据
- [ ] 路径正确：全部使用相对于项目根的路径或来自 config.py（禁止绝对路径如 `/Users/dongzihao/...`）

---

## Getting Started（Day 1 操作清单）

```bash
# 1. 初始化 git repo
cd /Users/dongzihao/Documents/司美格鲁肽
git init
git checkout -b main

# 2. 创建 .gitignore（内容见上文）

# 3. 创建 conda 环境
conda create -n semaglutide python=3.10
conda activate semaglutide
conda install -c conda-forge rdkit=2024.03
pip install xgboost scikit-learn lightgbm torch pandas numpy matplotlib seaborn shap jupyter

# 4. 验证环境
python -c "from rdkit import Chem; from rdkit.Chem import AllChem; print('RDKit OK')"
python -c "import xgboost; print('XGBoost', xgboost.__version__)"

# 5. 创建目录结构（见上文 Project Directory Structure）

# 6. 首次提交
git add .
git commit -m "Initialize project structure with CLAUDE.md"

# 7. 创建 GitHub private repo 并推送
gh repo create semaglutide-snac-screening --private --source=. --remote=origin --push
```

---

## Config 模板

```python
# code/utils/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# QSAR
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET_R2_ACTIVITY = 0.85
TARGET_R2_TOXICITY = 0.80

# Screening
ACTIVITY_THRESHOLD = 1.5  # 超过 SNAC 活性的倍数
TOXICITY_THRESHOLD = 1000  # CC50 in uM
PKA_MIN, PKA_MAX = 5.5, 7.0
LOGP_MIN, LOGP_MAX = 1.0, 3.0
MW_MIN, MW_MAX = 150, 400

# MD
MD_TIMESTEP = 2  # femtoseconds
MD_EQUILIBRATION = 5  # ns
MD_PRODUCTION = 50  # ns per system
MD_TEMPERATURE = 310  # Kelvin (body temperature)
```

---

---

## Available Tools

Claude Code 在此项目中可使用的工具：

| 工具 | 用途 | 调用方式 |
|------|------|---------|
| **WebSearch** | 网页搜索（文献检索、期刊 IF、专利查询等） | 内置工具 |
| **WebFetch** | 抓取指定 URL 内容（论文摘要、数据库查询结果等） | 内置工具 |
| **agent-browser** (v0.27.3) | 全功能浏览器自动化（需要交互操作的网页、登录后访问的数据库、JavaScript 渲染的页面） | `agent-browser open <url>` 等 Bash 命令 |
| **Bash** | 本地命令执行（Python 脚本、数据处理等） | 内置工具 |

**Phase 0 网验分工**：
- 简单检索（PubMed、Google Scholar 查文献）→ WebSearch
- 静态页面（期刊官网看 IF、ZINC15 看统计）→ WebFetch
- 需要交互的（ChEMBL 检索、专利数据库查询、需要翻页/点击的）→ agent-browser

---

## Key References（需自行验证——以下引用来自 docx，不一定准确）

> ⚠️ Phase 0 需要逐条验证这些文献是否真的存在、引用信息是否正确。

1. `[待验证]` **SNAC 机制**：Buckley et al. (2025), Nature Communications — SNAC-SPARC 共晶结构，解释作用机制。**Phase 0 确认：DOI、卷号、页码、作者名是否准确。**
2. `[待验证]` **C10 + 对位 F 突破**：2026 年 5 月文献（完整引用需检索确认）——如果不存在这篇文献，论文叙事需要重新设计。
3. `[待验证]` **吸收促进剂 QSAR**：搜索 EJPS / JCR 2023-2026 上的类似研究，确认领域现状。
4. `[待验证]` **ChemBERTa**：Chithrananda et al. (2020), "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction"——确认发表在哪里、版本是否还是最新。
5. `[待验证]` **NSGA-II**：Deb et al. (2002), "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"——经典文献，但也确认一下。

---

## Appendix: 常见新手问题

**Q: 我们都没发过论文，怎么知道论文结构对不对？**
A: 找 3 篇目标期刊最近的论文，按它们的章节结构照抄（不是内容，是结构）。审稿人期望特定格式——给他们要的格式。

**Q: QSAR 模型 R² 只有 0.75 怎么办？**
A: 不继续。外部验证 Q² < 0.7 通常会被拒。回到特征工程，加入 3D 描述符，尝试集成模型（XGBoost + LightGBM 平均），或收集更多数据。

**Q: 怎么判断 MD 结果 "好" 还是 "坏"？**
A: 先跑已知的 SNAC-司美格鲁肽体系。你的结合自由能应在发表值的 ~20% 以内（或至少在物理合理范围：-5 到 -15 kcal/mol）。如果 SNAC 基线结果就不合理，其他结果也不会合理。

**Q: 如果只能跑 50 个 MD 而不是 100 个？**
A: 没问题。50 个深入分析的体系 > 100 个粗浅分析的。论文写 "top 50 candidates selected for MD validation"，没人会因为验证了 50 个而不是 100 个就拒稿。

**Q: 需要做实验验证吗？**
A: 我们的目标期刊不需要。这些是计算化学期刊——in-silico 验证（MD）就够了。在 Discussion 中诚实写 "These predictions await experimental validation"——这是预期的表述。
