# lsoa-graph: spatial models of neighbourhood home energy efficiency in England

This is the code and results for *"Locating England's least energy-efficient neighbourhoods with open data: deprivation, housing stock and spatial machine learning"* (World Sustainable Energy Days 2027, Young Energy Researchers Conference). It is part of Chun Sang Au Yong's PhD at the UCL Institute for Environmental Design and Engineering.

- **Unit:** 2021 LSOAs in England (33,755 areas).
- **Outcome:** median SAP rating of Energy Performance Certificates.
- **Features:** 36 Census 2021 features, plus 16 GeoDS housing-stock features for the stock analysis.

## Layout

| Path | Contents |
|---|---|
| `config.py` | Paths, feature list, hyperparameters |
| `scripts/05_train_baselines.py` | Ridge, random forest and XGBoost on the spatially blocked folds |
| `scripts/06_train_gnn.py` | GraphSAGE and GAT on the contiguity graph and on contiguity + deprivation-similarity edges |
| `job_05_06.sh`, `hpc_setup.sh` | UCL Myriad job script (SGE, A100) and environment setup |
| `logs/` | Myriad run logs |
| `outputs/` | `graph.pt` (Queen contiguity), `folds.pt` (5 spatially blocked folds), `lsoa_features.parquet` (Census features and median SAP per LSOA), `results_*.csv` (out-of-fold predictions from stages 05–06) |
| `spatial_2x2/` | CPU analyses behind the paper. See `spatial_2x2/README.md`. |

`spatial_2x2/` covers four analyses:

- worst-area targeting;
- random vs blocked validation × neighbours' features vs neighbours' outcomes;
- a CPU re-implementation of the GraphSAGE model;
- the housing-stock extension.

## Reproduce the paper's results

```
pip install -r spatial_2x2/requirements.txt
cd spatial_2x2
python targeting_existing_predictions.py   # deprivation vs model targeting (seconds)
python exp_spatial_2x2.py                  # main comparison (~30 min on 2 CPU cores)
python exp_stock.py                        # needs ../properties_LSOA21.csv (GeoDS, licensed)
```

## Data

- **EPC:** Energy Performance of Buildings Data, England and Wales (DLUHC), open data.
- **Census 2021:** Office for National Statistics, Open Government Licence.
- **GeoDS property characteristics:** licensed, and **not included** in this repository. To run `exp_stock.py`, place `properties_LSOA21.csv` in this folder. `.gitignore` excludes it and its derived features.

## Not included

- The stage 01–04 scripts that built the graph, features and folds. The files they produced are in `outputs/`.
- Raw EPC and Census inputs (`datasets/`).

TODO: add the stage 01–04 scripts if they are still on Myriad.

## Licence

The code is released under the MIT licence (see `LICENSE`). Data files in `outputs/` and `spatial_2x2/` are derived from EPC and Census 2021 data and remain subject to those sources' licences.
