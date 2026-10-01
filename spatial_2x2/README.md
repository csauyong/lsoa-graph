# spatial_2x2 — where does the spatial signal in LSOA home energy efficiency live?

Extension of the lsoa-graph GNN ablation for the WSED 2027 submission (Oct 2026).
Target: LSOA median SAP (`median_energy_efficiency`), England, 33,755 LSOAs, the same 36 Census 2021 features.
XGBoost settings are identical to `scripts/05_train_baselines.py`. GraphSAGE is identical to `scripts/06_train_gnn.py`.

## 1. Finding the least efficient neighbourhoods (original Myriad predictions, blocked folds)

"Worst areas" means the LSOAs at or below the national 10th percentile of median SAP: SAP ≤ 62. Because of ties at 62, that is 3,935 LSOAs (11.7%), with a mean SAP of 58.8 against 67.7 for England. Each rule picks its 3,935 lowest-ranked LSOAs.

| Rule | Share of worst areas found | Mean true SAP of the areas picked |
|---|---|---|
| Random selection (expected) | 11.7% | 67.7 |
| Census household deprivation, most deprived first | **8.7%** | 68.4 |
| Ridge | 43.5% | 62.3 |
| Random forest | 46.8% | 61.9 |
| **XGBoost** | **49.0%** | 61.7 |
| GraphSAGE (contiguity) | 48.2% | 61.7 |
| GraphSAGE (heterogeneous) | 47.1% | 61.9 |
| GAT (contiguity / heterogeneous) | 36.7% / 24.2% | 63.2 / 65.3 |

Ranking by deprivation does worse than chance. More deprived LSOAs have slightly *higher* median SAP (Spearman +0.10), which is consistent with the Energy and Buildings paper.

Caveat: this uses the Census 2021 household-deprivation measure that is in the feature set, **not IMD**. Re-run with the IMD rank (or its income domain) before quoting it as "deprivation-based targeting".

## 2. The 2 × 2: random vs blocked validation × neighbours' features vs neighbours' outcomes

- **Random split:** 5-fold, seed 42. This is interpolation, i.e. filling gaps between known areas.
- **Blocked:** the original 5 folds. These are contiguous blocks: 99.4% of contiguity edges fall within a fold, and only **1.8%** of held-out LSOAs border a training LSOA. This is transfer to unseen regions.

The spatial methods tested:

- **Neighbours' features (SLX):** XGBoost on own features plus the 1st- and 2nd-order neighbour means of all 36 features. This gives the same 2-hop view as a 2-layer GraphSAGE.
- **GraphSAGE:** run on CPU (see Reproduction). The figures are the mean of 3 single-seed runs, not a seed ensemble.
- **Neighbours' outcomes (residual correction):** a regression-kriging-style step, built as follows.
  - Cross-fitted training residuals come from an inner random 5-fold.
  - For every LSOA, compute the mean residual of its 1st-order and 2nd-order *training* neighbours.
  - Fit the coefficient on training LSOAs only.
  - Add the correction to the XGBoost prediction.
- **Scrambled-graph nulls:** the same methods on a node-permuted copy of the contiguity graph. The graph shape and degrees are identical, but the geography is scrambled.

Residual Moran's I uses Queen contiguity (row-standardised) on the pooled out-of-fold residuals. The target's own Moran's I is 0.352.

| Validation | Method | RMSE | R² | Residual Moran's I | Worst areas found |
|---|---|---|---|---|---|
| Random | XGBoost (own features) | 3.926 | 0.520 | 0.176 | 55.2% |
| Random | + neighbours' features (SLX) | 3.777 | 0.555 | 0.120 | 58.3% |
| Random | GraphSAGE | 4.031 | 0.494 | 0.195 | 52.7% |
| Random | + neighbours' outcomes | 3.745 | 0.563 | **−0.046** | 59.9% |
| Random | + both | **3.678** | **0.578** | −0.032 | **61.2%** |
| Blocked | XGBoost (own features) | 4.215 | 0.446 | 0.233 | 49.0% |
| Blocked | + neighbours' features (SLX) | 4.152 | 0.463 | 0.234 | 50.2% |
| Blocked | GraphSAGE | 4.264 | 0.433 | 0.259 | 48.6% |
| Blocked | + neighbours' outcomes | 4.215 | 0.446 | 0.229 | 49.1% |
| Blocked | + both | **4.150** | **0.463** | 0.231 | **50.3%** |

Nulls (scrambled graph):

- **SLX:** worse than plain XGBoost in 5/5 folds under both schemes (RMSE 4.006 random, 4.262 blocked).
- **Residual correction:** coefficient ≈ 0 (−0.01 vs 0.41–0.48 on the real graph), so it produces no change.
- **GraphSAGE:** RMSE 4.188 random and 4.375 blocked, against 4.031 and 4.264 on the real graph.

Fold consistency against plain XGBoost:

- **Random split:** SLX, residual correction and both combined are better in 5/5 folds.
- **Blocked:** SLX is better in 4/5 folds. Residual correction is a coin-flip (2/5, all differences under 0.01).
- **GraphSAGE:** a single seed never beats XGBoost under the random split (0/5 folds for every seed). Under blocked folds it beats XGBoost in 0/5, 1/5 and 3/5 folds for the three seeds, and is worse on average.

## Reading

1. **The leftover spatial clustering is in neighbours' outcomes.** Borrowing residuals from neighbouring training areas removes it completely under interpolation (Moran's I 0.176 → −0.046, RMSE −4.6%). It does nothing for unseen regions, because held-out areas almost never have labelled neighbours. This is the place-specific part (stock, local schemes) that Census data does not carry.
2. **Neighbours' Census composition carries a small signal that transfers** (RMSE −1.5% blocked, −3.8% random). The scrambled-graph null confirms that this is geography, not extra model capacity.
3. **GraphSAGE's problem is the network, not the graph.** The real graph helps GraphSAGE against a scrambled one, but GraphSAGE starts below XGBoost. XGBoost fed the same neighbour information beats GraphSAGE under both schemes.
4. **Heterogeneous deprivation-similarity edges and GAT:** see the July analysis (both hurt; GAT undertrained). They were not re-run here.

## Files

| File | What it holds |
|---|---|
| `exp_spatial_2x2.py` | Full experiment. About 28 min on 2 CPU cores. |
| `targeting_existing_predictions.py` | Section 1, run on `../outputs/results_*.csv` |
| `sage_jax.py` | CPU (JAX) re-implementation of the GraphSAGE in `06_train_gnn.py`. It reproduces Myriad fold by fold within seed noise (seed 42: micro RMSE 4.318 vs Myriad 4.298). |
| `common.py` | Loading, Moran's I and the targeting metric |
| `ptload.py` | Reads the `.pt` files without torch |
| `results_2x2_summary.csv` | All methods, including per-seed GraphSAGE |
| `table_2x2_main.csv` | The table above |
| `results_2x2_oof_predictions.parquet` | Out-of-fold predictions for every scheme × method |
| `results_2x2_betas.csv` | Residual-correction coefficients by fold |
| `results_2x2_coverage.csv` | Share of test LSOAs with a training neighbour |
| `results_targeting_myriad.csv` | Section 1 table |

Check: blocked-fold XGBoost here equals the Myriad `results_baselines.csv` XGBoost predictions exactly (max |diff| = 0.0).

## Reproduction

```
pip install -r requirements.txt
python targeting_existing_predictions.py
python exp_spatial_2x2.py
```

Paths resolve relative to this folder (`../outputs/`, `../properties_LSOA21.csv`). Run the scripts from `spatial_2x2/`; results are written there.

No hyperparameter tuning was done for any model. All settings are as in the original scripts, which keeps the comparison like-for-like but is not a tuned benchmark.

## 3. Housing stock (GeoDS), added Oct 2026

`exp_stock.py` adds 16 LSOA-level GeoDS stock features to XGBoost, using `../properties_LSOA21.csv`. The features are shares by dwelling type, construction era and bedrooms, the shares with a garden or balcony, and mean internal area. The counts in that file are all multiples of 12, so only shares and means are used. The folds and settings are the same as above.

| Validation | Features | RMSE | R² | Residual Moran's I | Worst areas found |
|---|---|---|---|---|---|
| Random | Census | 3.926 | 0.520 | 0.176 | 55.2% |
| Random | Stock | 3.598 | 0.597 | 0.160 | 60.1% |
| Random | Census + stock | 3.223 | 0.676 | 0.122 | 66.1% |
| Random | Census + stock + neighbours' outcomes | 3.139 | 0.693 | −0.035 | 68.8% |
| Blocked | Census | 4.215 | 0.446 | 0.233 | 49.0% |
| Blocked | Stock | 3.732 | 0.566 | 0.191 | 54.8% |
| Blocked | Census + stock | 3.384 | 0.643 | 0.174 | 60.4% |
| Blocked | Census + stock + neighbours' outcomes | 3.384 | 0.643 | 0.170 | 60.4% |

Adding stock cuts the residual clustering by about a quarter but does not remove it. The neighbour-residual coefficient falls from 0.41 to 0.30 (random split), so a place-specific component remains beyond both Census and stock. SAP is partly computed from dwelling age and type, so some of the stock gain is expected by construction.

Extra files:

| File | What it holds |
|---|---|
| `exp_stock.py` | The stock experiment |
| `exp_spatial_2x2_lib.py` | Shared setup imported by `exp_stock.py` |
| `results_stock_summary.csv` | Summary results |
| `results_stock_oof_predictions.parquet` | Out-of-fold predictions |
| `stock_features_lsoa.parquet` | The 16 derived features (from licensed GeoDS data, so do not share) |

The WSED paper built from these results is in `../../WSED-2027/`.
