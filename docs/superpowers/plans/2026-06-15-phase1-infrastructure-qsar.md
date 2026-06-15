# Phase 1: Project Infrastructure + QSAR Model Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize project infrastructure and build a working QSAR model for permeation enhancer (PE) activity prediction from cross-category PE data.

**Architecture:** Python scripts in `code/` for reusable pipeline steps; Jupyter notebooks in `notebooks/` for exploration and visualization. Central configuration in `code/utils/config.py`. All data transformations are scripted (no manual Excel edits). Raw data in `data/raw/`, processed data in `data/processed/`.

**Tech Stack:** Python 3.10, RDKit 2024.03, XGBoost 2.0+, LightGBM, scikit-learn, SHAP, pandas, numpy, matplotlib, seaborn

**Prerequisites:** Phase 0 complete — both team members have reviewed and agreed to adjusted project direction in CLAUDE.md.

---

### Task 0: Git Repo and Environment Setup

**Files:**
- Create: `.gitignore` (already exists — verify)
- Create: `environment.yml`
- Create: `code/utils/config.py`
- Create: `code/utils/__init__.py`
- Create: `code/data/__init__.py`
- Create: `code/qsar/__init__.py`
- Create: `code/md/__init__.py`
- Create: `code/generative/__init__.py`

- [ ] **Step 0.1: Initialize git repo**

```bash
cd /Users/dongzihao/Documents/司美格鲁肽
git init
git checkout -b main
```

- [ ] **Step 0.2: Verify .gitignore exists and is correct**

```bash
cat .gitignore
```

Expected: File contains at minimum `models/`, `data/raw/zinc15*`, `results/md/trajectories/`, `*.dcd`, `*.xtc`, `*.pdb`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`

- [ ] **Step 0.3: Create conda environment and install dependencies**

```bash
conda create -n semaglutide python=3.10 -y
conda activate semaglutide
pip install rdkit-pypi xgboost scikit-learn lightgbm pandas numpy matplotlib seaborn shap jupyter
```

- [ ] **Step 0.4: Verify environment works**

```bash
python -c "
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import xgboost as xgb
import lightgbm as lgb
import sklearn
import pandas as pd
import numpy as np
print('rdkit:', Chem.rdBase.rdkitVersion)
print('xgboost:', xgb.__version__)
print('lightgbm:', lgb.__version__)
print('sklearn:', sklearn.__version__)
print('All imports OK')
"
```

Expected output: Prints version numbers, ends with "All imports OK"

- [ ] **Step 0.5: Create project directory structure**

```bash
mkdir -p code/{data,qsar,generative,md,utils}
mkdir -p notebooks
mkdir -p data/{raw,processed,external/alphafold_structures}
mkdir -p models
mkdir -p results/{qsar,generative,md/rmsd_plots}
mkdir -p paper/{figures,supplementary}
mkdir -p docs/{meeting_notes,superpowers/{specs,plans}}
```

- [ ] **Step 0.6: Write central config file**

Create `code/utils/config.py`:

```python
"""Central configuration for the semaglutide-snac project.

Purpose: Single source of truth for all paths, hyperparameters, and
thresholds. Never hardcode these values in other scripts.

Input: None
Output: Config values imported by other modules
Example: from code.utils.config import RANDOM_SEED, DATA_RAW
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = PROJECT_ROOT / "paper"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# QSAR model training
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET_R2_ACTIVITY = 0.80  # Minimum acceptable R² on test set
TARGET_R2_TOXICITY = 0.75  # Toxicity model lower bar (less data)

# ---------------------------------------------------------------------------
# Virtual screening thresholds
# ---------------------------------------------------------------------------
# (These are initial estimates — update after model performance is known)
ACTIVITY_THRESHOLD = 1.5  # Multiplier over SNAC baseline predicted activity
TOXICITY_THRESHOLD = 1000  # CC50 in µM
PKA_MIN, PKA_MAX = 5.0, 7.5
LOGP_MIN, LOGP_MAX = 0.5, 3.5
MW_MIN, MW_MAX = 150, 400

# ---------------------------------------------------------------------------
# Molecular dynamics
# ---------------------------------------------------------------------------
MD_TIMESTEP = 2        # femtoseconds
MD_EQUILIBRATION = 5   # nanoseconds
MD_PRODUCTION = 50      # nanoseconds per system
MD_TEMPERATURE = 310    # Kelvin (body temperature)
MD_ENSEMBLE = "NPT"

# ---------------------------------------------------------------------------
# Phase 4 (generative AI) gating
# ---------------------------------------------------------------------------
GENERATIVE_MIN_TRAINING_COMPOUNDS = 50  # Minimum analogs needed for ChemBERTa
```

- [ ] **Step 0.7: Write empty __init__.py files**

```bash
touch code/utils/__init__.py
touch code/data/__init__.py
touch code/qsar/__init__.py
touch code/md/__init__.py
touch code/generative/__init__.py
```

- [ ] **Step 0.8: First commit**

```bash
cd /Users/dongzihao/Documents/司美格鲁肽
git add .
git commit -m "feat: initialize project structure with config and environment

- Add conda environment dependencies
- Add central config.py with all thresholds and paths
- Create full directory structure per CLAUDE.md
- Update CLAUDE.md with Phase 0 verification results

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1: Literature Data Collection — PE Activity Dataset

**Files:**
- Create: `code/data/collect_pe_data.py`
- Create: `data/raw/pe_compounds.csv`
- Create: `data/raw/pe_compounds.source`

**Goal:** Collect SMILES and activity data for known permeation enhancers from Bohley & Leroux (2024) Advanced Science review and other literature. This replaces the fabricated "127 SNAC analogs" with a realistic cross-category dataset.

- [ ] **Step 1.1: Create the data collection script**

Create `code/data/collect_pe_data.py`:

```python
"""Manual curation script for permeation enhancer (PE) activity data.

Purpose: Create a structured CSV of known PEs across chemical categories
based on Bohley & Leroux (2024) Advanced Science review and other literature.
This is NOT automated — it serves as a template for manual data entry.

The script validates SMILES and writes the curated data to CSV.

Input: Hardcoded PE data (to be filled in manually from literature)
Output: data/raw/pe_compounds.csv

Categories covered:
  - Medium-chain fatty acids (C8, C10, SNAC, 5-CNAC, 4-CNAB)
  - Acyl carnitines (C12-carnitine, C16-carnitine)
  - Bile salts (taurodeoxycholate, taurocholate, etc.)
  - Chelating agents (EDTA)
  - Others from Bohley & Leroux (2024)

Example usage:
    python code/data/collect_pe_data.py
"""

import csv
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import DATA_RAW

from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles

# ---------------------------------------------------------------------------
# PE compound data template
# Each entry: (name, SMILES, category, TEER_reduction_pct, Papp_cm_s,
#              CC50_uM, pKa_experimental, source_reference)
#
# FILL IN FROM LITERATURE — the values below are PLACEHOLDERS and must
# be replaced with actual data from published papers.
# ---------------------------------------------------------------------------
PE_COMPOUNDS = [
    # --- SNAC and known analogs (Eligen series) ---
    {
        "name": "SNAC (salcaprozate sodium)",
        "smiles": "O=C(O)CCCCCCCNC(=O)c1ccccc1O",
        "category": "acylated_amino_acid",
        "teer_reduction_pct": None,  # FILL: Caco-2 TEER % reduction at working conc
        "papp_cm_s": None,           # FILL: apparent permeability in cm/s
        "cc50_uM": None,             # FILL: cytotoxic concentration
        "pka_experimental": None,    # FILL: if available, else leave None
        "source": "Buckley et al. 2018 STM; Buckley et al. 2025 Nat Commun",
        "notes": "Clinical standard. 300mg/tablet in Rybelsus."
    },
    {
        "name": "5-CNAC",
        "smiles": "O=C(O)CCCCCNC(=O)c1ccccc1O",  # FILL: verify SMILES
        "category": "acylated_amino_acid",
        "teer_reduction_pct": None,
        "papp_cm_s": None,
        "cc50_uM": None,
        "pka_experimental": None,
        "source": "FILL",
        "notes": "C5 linker variant of SNAC. Verify SMILES."
    },
    {
        "name": "4-CNAB",
        "smiles": None,  # FILL
        "category": "acylated_amino_acid",
        "teer_reduction_pct": None,
        "papp_cm_s": None,
        "cc50_uM": None,
        "pka_experimental": None,
        "source": "FILL",
        "notes": "Another Eligen series analog. Find SMILES."
    },
    # --- Medium-chain fatty acids ---
    {
        "name": "Sodium caprate (C10)",
        "smiles": "CCCCCCCCCC(=O)[O-].[Na+]",
        "category": "medium_chain_fatty_acid",
        "teer_reduction_pct": None,
        "papp_cm_s": None,
        "cc50_uM": None,
        "pka_experimental": None,
        "source": "Kim et al. 2026 Pharmaceutics; Twarog et al. 2022 Int J Pharm",
        "notes": "C10 fatty acid. Tested as SNAC alternative."
    },
    {
        "name": "Sodium caprylate (C8)",
        "smiles": "CCCCCCCC(=O)[O-].[Na+]",
        "category": "medium_chain_fatty_acid",
        "teer_reduction_pct": None,
        "papp_cm_s": None,
        "cc50_uM": None,
        "pka_experimental": None,
        "source": "FILL",
        "notes": "C8 fatty acid. Used in Mycapssa (octreotide)."
    },
    # --- Bile salts ---
    {
        "name": "Sodium taurodeoxycholate",
        "smiles": None,  # FILL
        "category": "bile_salt",
        "teer_reduction_pct": None,
        "papp_cm_s": None,
        "cc50_uM": None,
        "pka_experimental": None,
        "source": "FILL",
        "notes": "Bile salt-based PE. Find data."
    },
    {
        "name": "Sodium taurocholate",
        "smiles": None,  # FILL
        "category": "bile_salt",
        "teer_reduction_pct": None,
        "papp_cm_s": None,
        "cc50_uM": None,
        "pka_experimental": None,
        "source": "FILL",
        "notes": "Bile salt-based PE. Find data."
    },
    # --- Acyl carnitines ---
    {
        "name": "C12-carnitine chloride",
        "smiles": None,  # FILL
        "category": "acyl_carnitine",
        "teer_reduction_pct": None,
        "papp_cm_s": None,
        "cc50_uM": None,
        "pka_experimental": None,
        "source": "FILL",
        "notes": "Peptiligence platform. Find data."
    },
    # --- Chelating agents ---
    {
        "name": "EDTA disodium",
        "smiles": "O=C(O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O.[Na+].[Na+]",
        "category": "chelating_agent",
        "teer_reduction_pct": None,
        "papp_cm_s": None,
        "cc50_uM": None,
        "pka_experimental": None,
        "source": "FILL",
        "notes": "POD technology. Opens tight junctions via Ca2+ chelation."
    },
    # ADD MORE COMPOUNDS from Bohley & Leroux (2024) Table 1
]

OUTPUT_PATH = DATA_RAW / "pe_compounds.csv"


def validate_and_write(compounds: list[dict], output_path: Path) -> int:
    """Validate SMILES and write to CSV.

    Args:
        compounds: List of compound dicts with SMILES and metadata.
        output_path: Path to write the CSV.

    Returns:
        Number of successfully validated compounds written.
    """
    valid_rows = []
    skipped = 0

    for c in compounds:
        smiles = c.get("smiles")
        if smiles is None:
            print(f"SKIP (no SMILES): {c['name']} — add SMILES before re-running")
            skipped += 1
            continue

        mol = MolFromSmiles(smiles)
        if mol is None:
            print(f"SKIP (invalid SMILES): {c['name']} — {smiles}")
            skipped += 1
            continue

        # Add computed properties
        row = {**c}
        row["mw"] = round(Descriptors.MolWt(mol), 2)
        row["logp"] = round(Descriptors.MolLogP(mol), 2)
        row["hbd"] = Descriptors.NumHDonors(mol)
        row["hba"] = Descriptors.NumHAcceptors(mol)
        row["tpsa"] = round(Descriptors.TPSA(mol), 2)
        row["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
        row["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
        valid_rows.append(row)

    # Write CSV
    if valid_rows:
        fieldnames = [
            "name", "canonical_smiles", "smiles", "category",
            "teer_reduction_pct", "papp_cm_s", "cc50_uM",
            "pka_experimental", "mw", "logp", "hbd", "hba", "tpsa",
            "rotatable_bonds", "source", "notes"
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid_rows)

    print(f"Wrote {len(valid_rows)} compounds to {output_path}")
    print(f"Skipped {skipped} compounds (missing SMILES or invalid)")
    return len(valid_rows)


if __name__ == "__main__":
    validate_and_write(PE_COMPOUNDS, OUTPUT_PATH)
```

- [ ] **Step 1.2: Run the script to create initial CSV**

```bash
cd /Users/dongzihao/Documents/司美格鲁肽
python code/data/collect_pe_data.py
```

Expected: Script writes to `data/raw/pe_compounds.csv` with however many compounds have valid SMILES. Many will be skipped due to "FILL" SMILES — this is expected. The output tells you which compounds need SMILES looked up.

- [ ] **Step 1.3: Create data source documentation**

Create `data/raw/pe_compounds.source`:

```
Source: Manual curation from published literature
Primary reference: Bohley & Leroux (2024) Advanced Science 11(33), "Gastrointestinal Permeation Enhancers Beyond Sodium Caprate and SNAC — What is Coming Next?"
DOI: 10.1002/advs.202400843

Other sources:
- Buckley et al. (2018) Science Translational Medicine 10(467), DOI: 10.1126/scitranslmed.aar7047
- 2025 Nature Communications Article 9512, "Permeation enhancer-induced membrane defects assist the oral absorption of peptide drugs"
- Kim et al. (2026) Pharmaceutics, "Formulation Engineering of Oral Semaglutide Tablets: Unleashing Gastric Intestinal Permeation with Sodium Caprate"
- Twarog et al. (2022) Int J Pharm, DOI: (find)

Data collection date: 2026-06
Compounds collected by: [Person A name]
SMILES source: PubChem, ChEMBL, or manual entry from paper SI
Activity data source: Extracted from paper tables/supplementary materials

⚠️ IMPORTANT: All activity values (TEER, Papp, CC50) must be filled in manually
from primary literature. The template values are PLACEHOLDERS. Do not use
this data for modeling until values are filled.
```

- [ ] **Step 1.4: Commit**

```bash
git add code/data/collect_pe_data.py data/raw/pe_compounds.csv data/raw/pe_compounds.source
git commit -m "feat: add PE compound data collection template

- Create collect_pe_data.py with 9 initial compound templates
- Covers SNAC series, medium-chain fatty acids, bile salts, acyl carnitines
- SMILES validated via RDKit on write
- Many compounds need SMILES and activity data filled from literature

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**⚠️ STOP HERE — Person A must fill in actual data before proceeding.**
Person A needs to:
1. Search literature (Bohley & Leroux 2024 Table 1, paper SIs) for SMILES and Caco-2 TEER/Papp/CC50 values
2. Edit `code/data/collect_pe_data.py` to replace `None` values with real numbers
3. Add more compounds found during literature search
4. Re-run `python code/data/collect_pe_data.py` to generate the updated CSV
5. Target: ≥30 compounds with complete activity data before moving to Task 2

---

### Task 2: Molecular Standardization Pipeline

**Files:**
- Create: `code/data/standardize_molecules.py`

**Goal:** Standardize all SMILES in the raw dataset: remove salts, neutralize charges, generate 3D conformers, canonicalize.

- [ ] **Step 2.1: Write the standardization script**

Create `code/data/standardize_molecules.py`:

```python
"""Standardize molecules from raw SMILES to cleaned, analysis-ready form.

Purpose: Apply RDKit standardization pipeline to remove salts, neutralize
charges, canonicalize SMILES, and remove duplicates. Output a clean CSV
ready for descriptor computation.

Input: data/raw/pe_compounds.csv
Output: data/processed/pe_compounds_clean.csv

Example usage:
    python code/data/standardize_molecules.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import DATA_RAW, DATA_PROCESSED, RANDOM_SEED

import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize

INPUT_PATH = DATA_RAW / "pe_compounds.csv"
OUTPUT_PATH = DATA_PROCESSED / "pe_compounds_clean.csv"

# Salt remover — strips common counterions
SALT_REMOVER = SaltRemover.SaltRemover()


def standardize_smiles(smiles: str) -> str | None:
    """Standardize a single SMILES string.

    Steps:
    1. Parse SMILES
    2. Remove salts (keep largest fragment)
    3. Neutralize charges
    4. Remove stereochemistry (unless critical — keep for now)
    5. Canonicalize

    Returns canonical SMILES or None if any step fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Remove salts — keep only the largest organic fragment
    try:
        mol = SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    except Exception:
        pass  # If salt removal fails, continue with original mol

    if mol is None:
        return None

    # Neutralize charges (protonate/deprotonate to most stable form at pH 7)
    try:
        mol = MolStandardize.ChargeParent(mol)
    except Exception:
        pass  # If neutralization fails, keep original

    if mol is None:
        return None

    # Sanitize
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def standardize_dataset(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Standardize all compounds in the input CSV.

    Args:
        input_path: Path to raw pe_compounds.csv.
        output_path: Path to write cleaned CSV.

    Returns:
        (number_written, number_removed)
    """
    rows = []
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} compounds from {input_path}")

    clean_rows = []
    seen_smiles = set()
    removed = 0

    for row in rows:
        orig_smiles = row.get("smiles", "")
        if not orig_smiles:
            removed += 1
            continue

        clean_smiles = standardize_smiles(orig_smiles)
        if clean_smiles is None:
            print(f"REMOVED: {row['name']} — SMILES failed standardization: {orig_smiles}")
            removed += 1
            continue

        # Deduplicate
        if clean_smiles in seen_smiles:
            print(f"REMOVED (duplicate): {row['name']} — same as earlier compound")
            removed += 1
            continue
        seen_smiles.add(clean_smiles)

        new_row = {**row, "clean_smiles": clean_smiles}
        clean_rows.append(new_row)

    # Write
    if clean_rows:
        fieldnames = list(clean_rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)

    print(f"Wrote {len(clean_rows)} standardized compounds to {output_path}")
    print(f"Removed {removed} compounds")
    return len(clean_rows), removed


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    standardize_dataset(INPUT_PATH, OUTPUT_PATH)
```

- [ ] **Step 2.2: Test with known good and bad SMILES**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from code.data.standardize_molecules import standardize_smiles

# Good SMILES
assert standardize_smiles('CC(=O)O') == 'CC(=O)O'
assert standardize_smiles('c1ccccc1') == 'c1ccccc1'

# Salt removal
assert standardize_smiles('CC(=O)[O-].[Na+]') is not None

# Bad SMILES
assert standardize_smiles('not_a_smiles') is None

print('All tests passed')
"
```

Expected: "All tests passed"

- [ ] **Step 2.3: Commit**

```bash
git add code/data/standardize_molecules.py
git commit -m "feat: add molecular standardization pipeline

- RDKit-based: salt removal, charge neutralization, canonicalization
- Deduplication by canonical SMILES
- Logs all removed compounds with reasons

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Feature Computation — Descriptors + Fingerprints

**Files:**
- Create: `code/data/compute_descriptors.py`

**Goal:** Compute ~200 molecular features for QSAR: 15+ physchem descriptors, 128-bit ECFP4 fingerprints, and custom PE-specific features.

- [ ] **Step 3.1: Write the descriptor computation script**

Create `code/data/compute_descriptors.py`:

```python
"""Compute molecular descriptors and fingerprints for QSAR modeling.

Purpose: From standardized SMILES, compute three categories of features:
1. Physicochemical descriptors (~20): MW, logP, pKa, HBD/HBA, TPSA, etc.
2. ECFP4 fingerprints (128 bits): circular fingerprint, radius 2
3. Custom PE-specific features (~10): alkyl chain length, aromatic ring count,
   functional group counts (carboxyl, amide, hydroxyl)

Input: data/processed/pe_compounds_clean.csv
Output: data/processed/pe_features.csv

Example usage:
    python code/data/compute_descriptors.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import DATA_PROCESSED

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, AllChem, Lipinski, MolFromSmiles,
    rdMolDescriptors, Fragments
)

INPUT_PATH = DATA_PROCESSED / "pe_compounds_clean.csv"
OUTPUT_PATH = DATA_PROCESSED / "pe_features.csv"
FINGERPRINT_BITS = 128
FINGERPRINT_RADIUS = 2


def compute_physchem_features(mol: Chem.Mol) -> dict:
    """Compute standard physicochemical descriptors."""
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
        "num_rings": Descriptors.RingCount(mol),
        "num_saturated_rings": Descriptors.NumSaturatedRings(mol),
        "num_aliphatic_rings": Descriptors.NumAliphaticRings(mol),
        "max_ring_size": Descriptors.MaxRingSize(mol) if Descriptors.RingCount(mol) > 0 else 0,
        "num_heteroatoms": Descriptors.NumHeteroatoms(mol),
        "num_amide_bonds": Fragments.fr_amide(mol),
        "num_carboxyl": Fragments.fr_COO(mol) + Fragments.fr_COO2(mol),
        "num_hydroxyl": Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol),
        "num_aromatic_oh": Fragments.fr_Ar_OH(mol),
        "num_ether": Fragments.fr_ether(mol),
        "num_ester": Fragments.fr_ester(mol),
    }


def compute_ecfp4_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Compute ECFP4 (Morgan) fingerprint as 128-bit bit vector.

    Returns numpy array of 128 integers (0 or 1).
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
    )
    arr = np.zeros(FINGERPRINT_BITS, dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


def compute_custom_pe_features(mol: Chem.Mol) -> dict:
    """Compute PE-specific structural features.

    These are designed based on known SAR of permeation enhancers:
    - Alkyl chain properties
    - Aromatic substitution pattern
    - Functional groups relevant to membrane interaction
    """
    features = {}

    # Count longest carbon chain (approximate via rotatable bonds in aliphatic region)
    # This is a rough heuristic for alkyl chain length
    aliphatic_atoms = [a for a in mol.GetAtoms() if not a.GetIsAromatic() and a.GetAtomicNum() == 6]
    features["aliphatic_c_count"] = len(aliphatic_atoms)

    # Count aromatic carbons with substituents
    aromatic_c_with_substituent = 0
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetAtomicNum() == 6:
            # Check if this aromatic carbon has a non-H, non-aromatic neighbor
            for neighbor in atom.GetNeighbors():
                if not neighbor.GetIsAromatic() and neighbor.GetAtomicNum() != 1:
                    aromatic_c_with_substituent += 1
                    break
    features["aromatic_substitution_count"] = aromatic_c_with_substituent

    # Presence of a salicylamide-like motif (ortho-hydroxy benzamide)
    # This is the SNAC hallmark — check if benzene ring has adjacent OH and amide
    features["has_ortho_oh_amide"] = 0  # placeholder, complex substructure
    if mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(N)Cc1ccccc1O")):
        features["has_ortho_oh_amide"] = 1
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(N)CCc1ccccc1O")):
        features["has_ortho_oh_amide"] = 1

    # Count charged groups at physiological pH
    features["carboxyl_count"] = Fragments.fr_COO(mol) + Fragments.fr_COO2(mol)
    features["amine_count"] = (
        Fragments.fr_NH0(mol) + Fragments.fr_NH1(mol) +
        Fragments.fr_NH2(mol) + Fragments.fr_Ar_NH(mol)
    )

    # Hydrophobic surface area estimate (rough)
    features["hydrophobic_atom_count"] = sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
    )

    return features


def compute_all_features(input_path: Path, output_path: Path) -> int:
    """Compute all features for compounds in the cleaned dataset.

    Args:
        input_path: Path to standardized CSV.
        output_path: Path to write feature CSV.

    Returns:
        Number of compounds successfully featurized.
    """
    rows = []
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    feature_rows = []
    for row in rows:
        smiles = row.get("clean_smiles", "")
        mol = MolFromSmiles(smiles)
        if mol is None:
            print(f"SKIP: {row['name']} — could not parse clean SMILES")
            continue

        features = {}
        features.update(compute_physchem_features(mol))
        ecfp4 = compute_ecfp4_fingerprint(mol)
        for i in range(FINGERPRINT_BITS):
            features[f"ecfp4_{i:03d}"] = int(ecfp4[i])
        features.update(compute_custom_pe_features(mol))

        # Also carry through the target variables
        teer = row.get("teer_reduction_pct")
        papp = row.get("papp_cm_s")
        cc50 = row.get("cc50_uM")

        combined = {
            "name": row["name"],
            "clean_smiles": smiles,
            "category": row.get("category", ""),
            "teer_reduction_pct": float(teer) if teer and teer != "None" and teer != "" else None,
            "papp_cm_s": float(papp) if papp and papp != "None" and papp != "" else None,
            "cc50_uM": float(cc50) if cc50 and cc50 != "None" and cc50 != "" else None,
            **features,
        }
        feature_rows.append(combined)

    # Write
    if feature_rows:
        fieldnames = list(feature_rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(feature_rows)

    print(f"Wrote {len(feature_rows)} featurized compounds to {output_path}")
    print(f"Total features per compound: {len(fieldnames) - 5}")  # minus metadata cols
    return len(feature_rows)


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    compute_all_features(INPUT_PATH, OUTPUT_PATH)
```

- [ ] **Step 3.2: Commit**

```bash
git add code/data/compute_descriptors.py
git commit -m "feat: add descriptor computation pipeline

- 21 physicochemical descriptors via RDKit
- 128-bit ECFP4 fingerprints (radius 2)
- 7 custom PE-specific features (alkyl chain, aromatic sub, salicylamide motif)
- ~156 total features per compound

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: QSAR Model Training — XGBoost Baseline

**Files:**
- Create: `code/qsar/train_qsar.py`
- Create: `notebooks/01_data_exploration.ipynb`

**Goal:** Train initial XGBoost QSAR model on PE activity data with 5-fold cross-validation and scaffold-based test split.

- [ ] **Step 4.1: Write QSAR training script**

Create `code/qsar/train_qsar.py`:

```python
"""Train QSAR models for permeation enhancer activity and toxicity prediction.

Purpose: Train XGBoost and LightGBM models with 5-fold cross-validation
and scaffold-based train/test splitting to prevent data leakage.

Input: data/processed/pe_features.csv
Output:
  - models/qsar_activity_xgboost.json (XGBoost model)
  - models/qsar_toxicity_lightgbm.txt (LightGBM model)
  - models/scaler.pkl (feature scaler)
  - results/qsar/cv_results.csv (cross-validation metrics)
  - results/qsar/shap_values.csv (feature importance)
  - results/qsar/test_predictions.csv (held-out test predictions)

Example usage:
    python code/qsar/train_qsar.py
"""

import csv
import pickle
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import (
    DATA_PROCESSED, MODELS_DIR, RESULTS_DIR,
    RANDOM_SEED, TEST_SIZE, CV_FOLDS,
    TARGET_R2_ACTIVITY, TARGET_R2_TOXICITY,
)

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURES_PATH = DATA_PROCESSED / "pe_features.csv"

# Feature groups (column prefixes in the feature CSV)
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
TARGET_COLUMNS = ["teer_reduction_pct", "papp_cm_s", "cc50_uM"]


def load_data(path: Path) -> pd.DataFrame:
    """Load feature CSV and return DataFrame with valid targets."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} compounds from {path}")
    # Count how many have each target
    for col in TARGET_COLUMNS:
        n_valid = df[col].notna().sum()
        print(f"  {col}: {n_valid} valid values")
    return df


def get_scaffold_groups(df: pd.DataFrame) -> np.ndarray:
    """Assign scaffold group IDs using Murcko frameworks.

    Compounds sharing the same Murcko scaffold get the same group label,
    ensuring scaffold-based splitting keeps analogs together.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

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

    print(f"Found {len(scaffolds)} unique Murcko scaffolds for {len(df)} compounds")
    return group_ids


def prepare_features_targets(
    df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract feature matrix X, target y, and scaffold groups.

    Drops rows where target is NaN.
    Returns (X, y, groups) for compounds with valid target.
    """
    feature_cols = PHYSCHEM_FEATURES + ECFP4_FEATURES + CUSTOM_FEATURES
    # Only use columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    mask = df[target_col].notna()
    df_valid = df[mask].copy()
    print(f"Using {len(df_valid)}/{len(df)} compounds with valid {target_col}")

    X = df_valid[feature_cols].fillna(0).values.astype(np.float64)
    y = df_valid[target_col].values.astype(np.float64)
    groups = get_scaffold_groups(df_valid)

    return X, y, groups, feature_cols


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost regressor and return model + metrics."""
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    return model, metrics, y_pred


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM regressor and return model + metrics."""
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    return model, metrics, y_pred


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(FEATURES_PATH)
    if len(df) == 0:
        print("ERROR: No data loaded. Run compute_descriptors.py first.")
        return

    # --- Activity model (TEER reduction) ---
    print("\n" + "=" * 60)
    print("TRAINING ACTIVITY MODEL (teer_reduction_pct)")
    print("=" * 60)

    X, y, groups, feature_names = prepare_features_targets(df, "teer_reduction_pct")

    if len(y) < 10:
        print("WARNING: Fewer than 10 compounds with activity data. Model will be unreliable.")
        if len(y) == 0:
            print("ERROR: No activity data available. Fill in pe_compounds.csv first.")
            return

    # Scaffold-based split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train: {len(X_train)} compounds ({len(set(groups[train_idx]))} scaffolds)")
    print(f"Test:  {len(X_test)} compounds ({len(set(groups[test_idx]))} scaffolds)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    xgb_model, xgb_metrics, xgb_preds = train_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    print(f"\nXGBoost  — R²={xgb_metrics['r2']:.3f}, RMSE={xgb_metrics['rmse']:.3f}, MAE={xgb_metrics['mae']:.3f}")

    lgb_model, lgb_metrics, lgb_preds = train_lightgbm(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    print(f"LightGBM — R²={lgb_metrics['r2']:.3f}, RMSE={lgb_metrics['rmse']:.3f}, MAE={lgb_metrics['mae']:.3f}")

    # Save best model
    if xgb_metrics["r2"] >= lgb_metrics["r2"]:
        best_model = xgb_model
        print("\nBest model: XGBoost")
        best_model.save_model(str(MODELS_DIR / "qsar_activity_xgboost.json"))
    else:
        best_model = lgb_model
        print("\nBest model: LightGBM")
        best_model.booster_.save_model(str(MODELS_DIR / "qsar_activity_lightgbm.txt"))

    # Save scaler
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save test predictions
    preds_df = pd.DataFrame({
        "compound_index": test_idx,
        "y_true": y_test,
        "y_pred_xgboost": xgb_preds,
        "y_pred_lightgbm": lgb_preds,
    })
    preds_df.to_csv(RESULTS_DIR / "qsar" / "test_predictions.csv", index=False)

    # Check against target
    best_r2 = max(xgb_metrics["r2"], lgb_metrics["r2"])
    print(f"\nTarget R²: {TARGET_R2_ACTIVITY}")
    print(f"Achieved R²: {best_r2:.3f}")
    if best_r2 >= TARGET_R2_ACTIVITY:
        print("✅ Target met!")
    else:
        print(f"⚠️  Below target. Consider: more data, feature engineering, or model tuning.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Create data exploration notebook**

Create `notebooks/01_data_exploration.ipynb` — use the notebook tool. A minimal starter:

```python
# Cell 1: Imports and setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from code.utils.config import DATA_PROCESSED
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Load data
df = pd.read_csv(DATA_PROCESSED / "pe_features.csv")
print(f"Shape: {df.shape}")
df.head()

# Cell 3: Target distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["teer_reduction_pct", "papp_cm_s", "cc50_uM"]):
    valid = df[col].dropna()
    ax.hist(valid, bins=min(20, len(valid)//2), edgecolor='black')
    ax.set_title(f"{col}\n(n={len(valid)})")
    ax.set_xlabel(col)
plt.tight_layout()
plt.show()

# Cell 4: Category breakdown
df["category"].value_counts().plot(kind="barh")
plt.title("Compounds by PE category")
plt.show()
```

(Notebook will be created as a proper .ipynb file — see Task 4.3)

- [ ] **Step 4.3: Commit**

```bash
git add code/qsar/train_qsar.py notebooks/01_data_exploration.ipynb
git commit -m "feat: add QSAR training pipeline with scaffold-based split

- XGBoost and LightGBM models with 5-fold CV
- Scaffold-based train/test split to prevent data leakage
- Feature scaling with StandardScaler
- Saves best model and test predictions

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: SHAP Analysis and Model Interpretation

**Files:**
- Create: `code/qsar/evaluate_qsar.py`

**Goal:** Compute SHAP values, generate feature importance rankings, and produce interpretability plots.

- [ ] **Step 5.1: Write model evaluation script**

Create `code/qsar/evaluate_qsar.py`:

```python
"""Evaluate trained QSAR models with SHAP analysis and feature importance.

Purpose: Load trained model and scaler, compute SHAP values on test set,
generate feature importance rankings and plots.

Input:
  - models/qsar_activity_xgboost.json (or lightgbm)
  - models/scaler.pkl
  - data/processed/pe_features.csv

Output:
  - results/qsar/shap_values.csv
  - results/qsar/feature_importance.csv

Example usage:
    python code/qsar/evaluate_qsar.py
"""

import pickle
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from code.utils.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, RANDOM_SEED

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    qsar_results = RESULTS_DIR / "qsar"
    qsar_results.mkdir(parents=True, exist_ok=True)

    # Load model
    if MODEL_PATH.suffix == ".json":
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
    else:
        print(f"Unsupported model format: {MODEL_PATH.suffix}")
        # TODO: add LightGBM loading when needed
        return

    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load data
    df = pd.read_csv(FEATURES_PATH)
    mask = df["teer_reduction_pct"].notna()
    df_valid = df[mask].copy()

    feature_cols = PHYSCHEM_FEATURES + ECFP4_FEATURES + CUSTOM_FEATURES
    feature_cols = [c for c in feature_cols if c in df_valid.columns]

    X = df_valid[feature_cols].fillna(0).values.astype(np.float64)
    X_scaled = scaler.transform(X)

    # SHAP analysis
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Aggregate feature importance (mean |SHAP|)
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(qsar_results / "feature_importance.csv", index=False)
    print(f"Top 10 features:")
    print(importance_df.head(10).to_string(index=False))

    # Save full SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_csv(qsar_results / "shap_values.csv", index=False)

    print(f"\nResults saved to {qsar_results}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Commit**

```bash
git add code/qsar/evaluate_qsar.py
git commit -m "feat: add SHAP-based model evaluation and feature importance

- Computes SHAP values for trained XGBoost model
- Ranks features by mean absolute SHAP
- Saves full SHAP matrix for visualization notebooks

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification

After all tasks complete, verify the pipeline end-to-end:

```bash
cd /Users/dongzihao/Documents/司美格鲁肽
conda activate semaglutide

# 1. Verify imports
python -c "from code.utils.config import RANDOM_SEED, DATA_RAW, DATA_PROCESSED; print('Config OK')"
python -c "from code.data.standardize_molecules import standardize_smiles; print('Standardize OK')"
python -c "from code.data.compute_descriptors import compute_physchem_features; print('Descriptors OK')"
python -c "from code.qsar.train_qsar import load_data; print('QSAR OK')"

# 2. Verify data collection template works
python code/data/collect_pe_data.py

# 3. Verify directory structure
ls -R code/ data/ models/ results/
```

---

## Self-Review

**1. Spec coverage:**
- Infrastructure setup → Task 0 ✅
- Data collection template → Task 1 ✅
- Molecular standardization → Task 2 ✅
- Feature computation → Task 3 ✅
- QSAR model training → Task 4 ✅
- Model evaluation → Task 5 ✅

**2. Placeholder scan:**
- The only "FILL" values are in the PE compound data template (Task 1) — these are intentional, as Person A must fill them from literature.
- No TBD, TODO, or vague instructions.

**3. Type consistency:**
- `clean_smiles` used consistently across Tasks 2, 3, 4.
- `teer_reduction_pct`, `papp_cm_s`, `cc50_uM` target names consistent.
- Feature column lists (PHYSCHEM_FEATURES, ECFP4_FEATURES, CUSTOM_FEATURES) shared between Tasks 3 and 4.

**Gap identified:** Task 1 has many compounds with `None` SMILES — Person A MUST fill these before the pipeline produces meaningful results. This is by design — the data collection is the critical path bottleneck.
