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
from code.utils.config import DATA_RAW, DATA_PROCESSED

from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize

INPUT_PATH = DATA_RAW / "pe_compounds.csv"
OUTPUT_PATH = DATA_PROCESSED / "pe_compounds_clean.csv"

SALT_REMOVER = SaltRemover.SaltRemover()


def standardize_smiles(smiles: str):
    """Standardize a single SMILES string.

    Steps: parse → remove salts → neutralize → sanitize → canonicalize.
    Returns canonical SMILES or None if any step fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        mol = SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    except Exception:
        pass

    if mol is None:
        return None

    try:
        mol = MolStandardize.ChargeParent(mol)
    except Exception:
        pass

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} compounds from {INPUT_PATH}")

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
            print(f"REMOVED: {row['name']} — SMILES failed: {orig_smiles}")
            removed += 1
            continue

        if clean_smiles in seen_smiles:
            print(f"REMOVED (dup): {row['name']}")
            removed += 1
            continue
        seen_smiles.add(clean_smiles)

        new_row = {**row, "clean_smiles": clean_smiles}
        clean_rows.append(new_row)

    if clean_rows:
        fieldnames = list(clean_rows[0].keys())
        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)

    print(f"Wrote {len(clean_rows)} compounds → {OUTPUT_PATH}")
    print(f"Removed {removed}")


if __name__ == "__main__":
    main()
