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
from sage_jax import run_sage

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

oof = {}      # (scheme, method) -> full-length OOF prediction vector
betas, coverage, timing = [], [], {}
for scheme, folds in SCHEMES.items():
    t0 = time.time()
    P = {}
    def put(name, te, p):
        P.setdefault(name, np.full(n, np.nan))[te] = p
    for f, (tr, te) in enumerate(folds):
        # --- neighbours' features ---
        m0 = xgb().fit(X[tr], y[tr]); base = m0.predict(X[te]); put("XGBoost", te, base)
        for g in GRAPHS:
            m1 = xgb().fit(SLX[g][tr], y[tr]); put(f"XGBoost+SLX [{g}]", te, m1.predict(SLX[g][te]))
        for g in GRAPHS:
            ps = [run_sage(X, y, GRAPHS[g], tr, te, seed=s)[0] for s in GNN_SEEDS]
            put(f"GraphSAGE [{g}]", te, np.mean(ps, axis=0))
            for s, p in zip(GNN_SEEDS, ps):
                put(f"GraphSAGE [{g}] seed{s}", te, p)
        # --- neighbours' outcomes ---
        r0 = oof_train_residuals(X, tr)
        for g in GRAPHS:
            corr, beta, cov = residual_correction(r0, tr, te, *ADJ[g])
            put(f"XGBoost+residual [{g}]", te, base + corr)
            betas.append((scheme, f, g, "XGBoost", *beta)); coverage.append((scheme, f, g, cov))
        rS = oof_train_residuals(SLX["contiguity"], tr)
        baseS = xgb().fit(SLX["contiguity"][tr], y[tr]).predict(SLX["contiguity"][te])
        corr, beta, _ = residual_correction(rS, tr, te, *ADJ["contiguity"])
        put("XGBoost+SLX+residual [contiguity]", te, baseS + corr)
        betas.append((scheme, f, "contiguity", "XGBoost+SLX", *beta))
        print(f"{scheme} fold {f} done ({time.time()-t0:.0f}s)", flush=True)
    for k, v in P.items():
        assert not np.isnan(v).any(), (scheme, k)
        oof[(scheme, k)] = v
    timing[scheme] = time.time() - t0

# ---------------- metrics ----------------
Wr, _ = row_std_W(ei, n)
worst, thr = worst_set(y)
fold_of = {s: np.concatenate([np.full(len(te), i) for i, (_, te) in enumerate(fs)])[np.argsort(np.concatenate([te for _, te in fs]))]
           for s, fs in SCHEMES.items()}
rows = []
for (scheme, method), p in oof.items():
    e = p - y
    fr = [np.sqrt(np.mean(e[fold_of[scheme] == i] ** 2)) for i in range(5)]
    rec, picked = targeting(y, p, worst)
    rows.append(dict(scheme=scheme, method=method, RMSE=np.sqrt(np.mean(e ** 2)),
                     RMSE_fold_mean=np.mean(fr), RMSE_fold_sd=np.std(fr, ddof=1),
                     R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
                     residual_MoransI=morans_i(e, Wr), recall_worst=rec, mean_SAP_picked=picked,
                     spearman=spearmanr(p, y)[0]))
res = pd.DataFrame(rows)
res.to_csv("results_2x2_summary.csv", index=False)
pd.DataFrame(betas, columns=["scheme", "fold", "graph", "base", "beta_1st_order", "beta_2nd_order"]).to_csv("results_2x2_betas.csv", index=False)
pd.DataFrame(coverage, columns=["scheme", "fold", "graph", "share_test_with_train_neighbour"]).to_csv("results_2x2_coverage.csv", index=False)
long = pd.DataFrame({f"{s}|{m}": v for (s, m), v in oof.items()}); long.insert(0, "LSOA21CD", df.LSOA21CD.values); long.insert(1, "y_true", y)
long.to_parquet("results_2x2_oof_predictions.parquet", index=False)
print(f"\nworst set: SAP <= {thr} ({worst.sum()} LSOAs, {worst.mean():.1%}); target Moran's I = {morans_i(y, Wr):.3f}")
print("timing (s):", {k: round(v) for k, v in timing.items()})
pd.set_option("display.width", 200)
print(res[~res.method.str.contains("seed")].sort_values(["scheme", "RMSE"]).round(3).to_string(index=False))
