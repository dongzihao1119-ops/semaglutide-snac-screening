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
MD_PRODUCTION = 50     # nanoseconds per system
MD_TEMPERATURE = 310   # Kelvin (body temperature)
MD_ENSEMBLE = "NPT"

# ---------------------------------------------------------------------------
# Phase 4 (generative AI) gating
# ---------------------------------------------------------------------------
GENERATIVE_MIN_TRAINING_COMPOUNDS = 50  # Minimum analogs needed for ChemBERTa
