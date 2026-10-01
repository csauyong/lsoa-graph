import numpy as np, pandas as pd, ptload
from pathlib import Path
from scipy.sparse import coo_matrix, diags

D = str(Path(__file__).resolve().parent.parent / "outputs") + "/"
FEATURE_COLS = [
    "Median Age", "Mean Household Size", "Mean Household Deprivation",
    "Ethnicity: Asian", "Ethnicity: Black", "Ethnicity: Mixed", "Ethnicity: Other", "Ethnicity: White",
    "Religion: Buddhist", "Religion: Christian", "Religion: Hindu", "Religion: Jewish", "Religion: Muslim",
    "Religion: None", "Religion: Not answered", "Religion: Other", "Religion: Sikh",
    "NS-SeC: Higher managerial", "NS-SeC: Lower supervisory", "NS-SeC: Semi-routine", "NS-SeC: Routine",
    "NS-SeC: Never worked", "NS-SeC: Students", "NS-SeC: Lower managerial", "NS-SeC: Intermediate",
    "NS-SeC: Small employers", "NS-SeC: Does Not Apply",
    "Industry: Agriculture, energy and water", "Industry: Manufacturing", "Industry: Construction",
    "Industry: Distribution, hotels and restaurants", "Industry: Transport and communication",
    "Industry: Financial and professional", "Industry: Public sector", "Industry: Other",
    "Industry: Does Not Apply",
]
TARGET = "median_energy_efficiency"

def load():
    df = pd.read_parquet(D + "lsoa_features.parquet")
    g = ptload.load(D + "graph.pt")
    assert list(df.LSOA21CD) == g["lsoa_codes"]
    folds = [(tr.astype(int), te.astype(int)) for tr, te in ptload.load(D + "folds.pt")]
    return df, g["edge_index"].astype(int), folds

def row_std_W(ei, n):
    W = coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    W.data[:] = 1.0  # binary
    rs = np.asarray(W.sum(1)).ravel()
    inv = np.where(rs > 0, 1.0 / np.maximum(rs, 1), 0.0)
    return diags(inv) @ W, rs

def morans_i(x, Wr):
    """Moran's I with a row-standardised W (islands contribute 0)."""
    z = x - x.mean()
    S0 = Wr.sum()
    return float(len(x) / S0 * (z @ (Wr @ z)) / (z @ z))

def worst_set(y, q=0.10):
    """True worst areas: SAP at or below the national 10th percentile."""
    thr = np.quantile(y, q)
    return y <= thr, thr

def targeting(y, score, worst_mask, rng_seed=0):
    """Select the k areas with the LOWEST score (k = size of true worst set).
    Ties in score broken at random. Returns recall (= precision) and mean true SAP of the picked set."""
    k = int(worst_mask.sum())
    rng = np.random.default_rng(rng_seed)
    order = np.lexsort((rng.random(len(score)), score))
    pick = np.zeros(len(y), bool); pick[order[:k]] = True
    return (pick & worst_mask).sum() / k, y[pick].mean()
