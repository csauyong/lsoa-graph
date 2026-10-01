from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "datasets"
OUTPUTS_DIR = ROOT / "outputs"

# ── Input data ───────────────────────────────────────────────────────────────
LSOA_BOUNDARIES = DATASETS_DIR / "lsoa_boundaries_2021_ew.geojson"
CENSUS_FEATURES  = DATASETS_DIR / "census_node_features.csv"
EPC_DATA         = DATASETS_DIR / "epc_2016_2024.csv"

# ── Intermediate outputs ──────────────────────────────────────────────────────
GRAPH_PT              = OUTPUTS_DIR / "graph.pt"
LSOA_FEATURES_PQ      = OUTPUTS_DIR / "lsoa_features.parquet"
FOLD_MAP_PNG          = OUTPUTS_DIR / "fold_map.png"
FOLDS_PT              = OUTPUTS_DIR / "folds.pt"
RESULTS_BASELINES_CSV = OUTPUTS_DIR / "results_baselines.csv"
RESULTS_GNN_CSV       = OUTPUTS_DIR / "results_gnn.csv"

# ── Feature / target columns (confirmed at stage 3) ──────────────────────────
TARGET_COL = "median_energy_efficiency"

FEATURE_COLS = [
    "Median Age", "Mean Household Size", "Mean Household Deprivation",
    "Ethnicity: Asian", "Ethnicity: Black", "Ethnicity: Mixed",
    "Ethnicity: Other", "Ethnicity: White",
    "Religion: Buddhist", "Religion: Christian", "Religion: Hindu",
    "Religion: Jewish", "Religion: Muslim", "Religion: None",
    "Religion: Not answered", "Religion: Other", "Religion: Sikh",
    "NS-SeC: Higher managerial", "NS-SeC: Lower supervisory",
    "NS-SeC: Semi-routine", "NS-SeC: Routine", "NS-SeC: Never worked",
    "NS-SeC: Students", "NS-SeC: Lower managerial", "NS-SeC: Intermediate",
    "NS-SeC: Small employers", "NS-SeC: Does Not Apply",
    "Industry: Agriculture, energy and water", "Industry: Manufacturing",
    "Industry: Construction",
    "Industry: Distribution, hotels and restaurants",
    "Industry: Transport and communication",
    "Industry: Financial and professional", "Industry: Public sector",
    "Industry: Other", "Industry: Does Not Apply",
]

# Column used as the similarity metric for IMD k-NN heterogeneous edges
IMD_SIMILARITY_COL = "Mean Household Deprivation"

# ── Hyperparameters ───────────────────────────────────────────────────────────
RANDOM_SEED   = 42
N_CV_FOLDS    = 5

# GNN
GNN_HIDDEN_DIM  = 64
GNN_NUM_LAYERS  = 2
GNN_DROPOUT     = 0.3
GNN_LR          = 1e-3
GNN_EPOCHS      = 200
GNN_BATCH_SIZE  = 256

# IMD k-NN heterogeneous edges
IMD_KNN_K = 10
