"""Worst-area targeting check on the ORIGINAL Myriad predictions (spatially blocked folds).
Question: of the LSOAs with the lowest median SAP, how many does each rule put in its bottom set?"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import *

df, ei, folds = load()
y = df[TARGET].values.astype(float)
worst, thr = worst_set(y)                     # SAP <= national 10th percentile (ties -> 11.7% of LSOAs)
b = pd.read_csv(D + "results_baselines.csv"); g = pd.read_csv(D + "results_gnn.csv")
g["model"] = g.model + " [" + g.graph_type + "]"
P = (pd.concat([b[["model", "lsoa_code", "y_pred"]], g[["model", "lsoa_code", "y_pred"]]])
       .pivot(index="lsoa_code", columns="model", values="y_pred").reindex(df.LSOA21CD))
dep = df["Mean Household Deprivation"].values
rows = [("Random selection (expected)", worst.mean(), y.mean(), np.nan)]
r, m = targeting(y, -dep, worst); rows.append(("Census household deprivation, most deprived first", r, m, spearmanr(dep, y)[0]))
for c in ["Ridge", "RandomForest", "XGBoost", "GraphSAGE [contiguity]", "GraphSAGE [heterogeneous]",
          "GAT [contiguity]", "GAT [heterogeneous]"]:
    r, m = targeting(y, P[c].values, worst); rows.append((c, r, m, spearmanr(P[c].values, y)[0]))
out = pd.DataFrame(rows, columns=["rule", "share_of_worst_areas_found", "mean_true_SAP_of_selected", "spearman_with_SAP"])
out.to_csv("results_targeting_myriad.csv", index=False)
print(f"Worst set: SAP <= {thr:.0f} -> {worst.sum()} LSOAs ({worst.mean():.1%}); their mean SAP {y[worst].mean():.2f}; England mean {y.mean():.2f}")
print(out.round(3).to_string(index=False))
