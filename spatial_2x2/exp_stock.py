"""Does housing stock explain the spatial clustering that Census composition leaves behind?
Adds LSOA-level GeoDS property characteristics (shares of dwelling type, construction era, bedrooms,
gardens/balconies; mean internal area) to the XGBoost model, under both validation schemes, and
re-runs the neighbour-outcome residual correction on top. Same XGBoost settings and folds as
exp_spatial_2x2.py. Counts in properties_LSOA21.csv are all multiples of 12, so only shares are used."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from common import *
from exp_spatial_2x2_lib import xgb, oof_train_residuals, residual_correction, ADJ_CONTIG, n, y, SCHEMES, df

p = pd.read_csv(Path(__file__).resolve().parent.parent / "properties_LSOA21.csv").set_index("lsoa21").reindex(df.LSOA21CD)
tot = p.total_properties
S = pd.DataFrame({
    "share_detached": p.subtype_detached / tot, "share_semi": p.subtype_semi_detached / tot,
    "share_terraced": p.subtype_terraced / tot, "share_flat": p.subtype_flat / tot,
    "era_period": p.era_period / tot, "era_early_century": p.era_early_century / tot,
    "era_mid_century": p.era_mid_century / tot, "era_modern": p.era_modern / tot,
    "share_garden": p.has_garden / tot, "share_balcony": p.has_balcony / tot,
    "mean_internal_area": p.mean_internal_area,
    "bed_0_1": (p.bedroom_0 + p.bedroom_1) / tot, "bed_2": p.bedroom_2 / tot, "bed_3": p.bedroom_3 / tot,
    "bed_4": p.bedroom_4 / tot, "bed_5plus": p.bedroom_5plus / tot,
})
Xc = df[FEATURE_COLS].values.astype(float); Xs = S.values.astype(float); Xcs = np.hstack([Xc, Xs])
FS = {"Census": Xc, "Stock": Xs, "Census+Stock": Xcs}
Wr, _ = row_std_W(ADJ_CONTIG[2], n)
worst, _ = worst_set(y)
rows, oof = [], {}
for scheme, folds in SCHEMES.items():
    P = {}
    for f, (tr, te) in enumerate(folds):
        for name, F in FS.items():
            P.setdefault(name, np.full(n, np.nan))[te] = xgb().fit(F[tr], y[tr]).predict(F[te])
        r = oof_train_residuals(Xcs, tr)
        corr, beta, _ = residual_correction(r, tr, te, ADJ_CONTIG[0], ADJ_CONTIG[1])
        P.setdefault("Census+Stock+residual", np.full(n, np.nan))[te] = P["Census+Stock"][te] + corr
        print(scheme, f, "beta", np.round(beta, 3), flush=True)
    for k, v in P.items():
        e = v - y; rec, picked = targeting(y, v, worst)
        rows.append(dict(scheme=scheme, method=k, RMSE=np.sqrt(np.mean(e**2)), R2=1 - np.sum(e**2) / np.sum((y - y.mean())**2),
                         residual_MoransI=morans_i(e, Wr), recall_worst=rec, mean_SAP_picked=picked, spearman=spearmanr(v, y)[0]))
        oof[f"{scheme}|{k}"] = v
res = pd.DataFrame(rows); res.to_csv("results_stock_summary.csv", index=False)
o = pd.DataFrame(oof); o.insert(0, "LSOA21CD", df.LSOA21CD.values); o.to_parquet("results_stock_oof_predictions.parquet", index=False)
S.assign(LSOA21CD=df.LSOA21CD.values).to_parquet("stock_features_lsoa.parquet", index=False)
print(res.round(3).to_string(index=False))
