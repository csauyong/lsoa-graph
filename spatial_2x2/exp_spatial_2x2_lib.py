"""
Where does the spatial signal in LSOA home energy efficiency live?

2 x 2 design:
  validation   : random 5-fold (interpolation)  vs  the original spatially blocked 5 folds (transfer)
  spatial info : neighbours' FEATURES (SLX lags into XGBoost; GraphSAGE)
                 vs neighbours' OUTCOMES (residual correction from neighbouring training LSOAs)
Nulls          : the same spatial methods on a node-permuted graph (identical structure, geography scrambled)

Target: LSOA median SAP (median_energy_efficiency), England, 33,755 LSOAs, 36 Census 2021 features.
XGBoost settings identical to scripts/05_train_baselines.py; GraphSAGE identical to scripts/06_train_gnn.py.
"""
import time, numpy as np, pandas as pd
from scipy.sparse import coo_matrix, diags, identity
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from common import *

SEED = 42
GNN_SEEDS = [0, 1, 2]
df, ei, blocked = load()
n = len(df)
X = df[FEATURE_COLS].values.astype(np.float64)
y = df[TARGET].values.astype(np.float64)
rand = [(tr, te) for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X)]
SCHEMES = {"random": rand, "blocked": blocked}

perm = np.random.default_rng(SEED).permutation(n)
GRAPHS = {"contiguity": ei, "permuted": perm[ei]}   # permuted: same graph shape, nodes relabelled at random

def adj(e):
    A = coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(n, n)).tocsr(); A.data[:] = 1.0
    A2 = (A @ A).tocsr(); A2.data[:] = 1.0
    A2 = A2 - A2.multiply(A) - A2.multiply(identity(n, format="csr")); A2.eliminate_zeros()
    return A, A2

def rowstd(A):
    rs = np.asarray(A.sum(1)).ravel()
    return diags(np.where(rs > 0, 1 / np.maximum(rs, 1), 0)) @ A

ADJ = {k: adj(v) for k, v in GRAPHS.items()}
SLX = {}
for k, (A, _) in ADJ.items():
    W = rowstd(A); WX = W @ X; WWX = W @ WX
    SLX[k] = np.hstack([X, WX, WWX])           # own + 1st- and 2nd-order neighbour means (2-hop, like 2-layer GraphSAGE)

def xgb():
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbosity=0)

def oof_train_residuals(F, tr):
    """Cross-fitted (inner random 5-fold) residuals for training LSOAs."""
    r = np.full(n, np.nan)
    for itr, ite in KFold(5, shuffle=True, random_state=SEED).split(tr):
        m = xgb().fit(F[tr[itr]], y[tr[itr]])
        r[tr[ite]] = y[tr[ite]] - m.predict(F[tr[ite]])
    return r

def neighbour_means(r, trmask, A, A2):
    """Mean training-set residual among 1st-order and 2nd-order neighbours (0 where none)."""
    v = np.where(trmask, r, 0.0); c = trmask.astype(float)
    out = []
    for M in (A, A2):
        s = M @ v; k = M @ c
        out.append(np.where(k > 0, s / np.maximum(k, 1), 0.0))
    return np.column_stack(out)

def residual_correction(r, tr, te, A, A2):
    trmask = np.zeros(n, bool); trmask[tr] = True
    M = neighbour_means(r, trmask, A, A2)
    beta, *_ = np.linalg.lstsq(M[tr], r[tr], rcond=None)     # fitted on training LSOAs only
    return M[te] @ beta, beta, (M[te, 0] != 0).mean()


ADJ_CONTIG = (*ADJ['contiguity'], ei)
