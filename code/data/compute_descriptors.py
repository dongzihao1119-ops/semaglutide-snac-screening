"""Compute molecular descriptors and fingerprints for QSAR modeling.

Purpose: From standardized SMILES, compute three categories of features:
1. Physicochemical descriptors (~20): MW, logP, HBD/HBA, TPSA, etc.
2. ECFP4 fingerprints (128-bit): circular fingerprint, radius 2
3. Custom PE-specific features (~7): alkyl chain C count, aromatic sub,
   salicylamide motif, carboxyl/amine counts

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
    Descriptors, AllChem, Lipinski, MolFromSmiles, Fragments,
)

INPUT_PATH = DATA_PROCESSED / "pe_compounds_clean.csv"
OUTPUT_PATH = DATA_PROCESSED / "pe_features.csv"
FINGERPRINT_BITS = 128
FINGERPRINT_RADIUS = 2


def compute_physchem_features(mol):
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
        # max_ring_size not available in rdkit-pypi 2022.09, use RingCount instead
        "max_ring_size": Descriptors.RingCount(mol),
        "num_heteroatoms": Descriptors.NumHeteroatoms(mol),
        "num_amide_bonds": Fragments.fr_amide(mol),
        "num_carboxyl": Fragments.fr_COO(mol) + Fragments.fr_COO2(mol),
        "num_hydroxyl": Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol),
        "num_aromatic_oh": Fragments.fr_Ar_OH(mol),
        "num_ether": Fragments.fr_ether(mol),
        "num_ester": Fragments.fr_ester(mol),
    }


def compute_ecfp4_fingerprint(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS,
    )
    arr = np.zeros(FINGERPRINT_BITS, dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


def compute_custom_pe_features(mol):
    features = {}
    aliphatic_c = [a for a in mol.GetAtoms()
                   if not a.GetIsAromatic() and a.GetAtomicNum() == 6]
    features["aliphatic_c_count"] = len(aliphatic_c)

    aromatic_sub = 0
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetAtomicNum() == 6:
            for nbr in atom.GetNeighbors():
                if not nbr.GetIsAromatic() and nbr.GetAtomicNum() != 1:
                    aromatic_sub += 1
                    break
    features["aromatic_substitution_count"] = aromatic_sub

    features["has_ortho_oh_amide"] = 0
    if mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(N)Cc1ccccc1O")):
        features["has_ortho_oh_amide"] = 1
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(N)CCc1ccccc1O")):
        features["has_ortho_oh_amide"] = 1

    features["carboxyl_count"] = Fragments.fr_COO(mol) + Fragments.fr_COO2(mol)
    features["amine_count"] = (
        Fragments.fr_NH0(mol) + Fragments.fr_NH1(mol) +
        Fragments.fr_NH2(mol) + Fragments.fr_Ar_NH(mol)
    )
    features["hydrophobic_atom_count"] = sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
    )
    return features


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} compounds")

    feature_rows = []
    for row in rows:
        smiles = row.get("clean_smiles", "")
        mol = MolFromSmiles(smiles)
        if mol is None:
            print(f"SKIP: {row['name']} — bad clean SMILES")
            continue

        feats = {}
        feats.update(compute_physchem_features(mol))
        ecfp4 = compute_ecfp4_fingerprint(mol)
        for i in range(FINGERPRINT_BITS):
            feats[f"ecfp4_{i:03d}"] = int(ecfp4[i])
        feats.update(compute_custom_pe_features(mol))

        def _to_float(val):
            if val and val != "None" and val != "":
                return float(val)
            return None

        combined = {
            "name": row["name"],
            "clean_smiles": smiles,
            "category": row.get("category", ""),
            "teer_reduction_pct": _to_float(row.get("teer_reduction_pct")),
            "papp_cm_s": _to_float(row.get("papp_cm_s")),
            "cc50_uM": _to_float(row.get("cc50_uM")),
            # Carry through mechanism metadata
            "mechanism_type": row.get("mechanism_type", ""),
            "k_value": row.get("k_value", ""),
            "maher_class": row.get("maher_class", ""),
            **feats,
        }
        feature_rows.append(combined)

    if feature_rows:
        fieldnames = list(feature_rows[0].keys())
        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(feature_rows)

    n_feats = len(fieldnames) - 5  # minus metadata cols
    print(f"Wrote {len(feature_rows)} compounds → {OUTPUT_PATH}")
    print(f"Features per compound: {n_feats}")


if __name__ == "__main__":
    main()
