"""
Stage 6 — Train GNN models (GraphSAGE, GAT).

Ablation:
  - graph_type="contiguity"    : Queen's contiguity edges only
  - graph_type="heterogeneous" : contiguity + IMD-similarity k-NN edges

For each (graph_type, model, fold):
  - Full-batch training with train/test masks
  - StandardScaler fit on training nodes only (features AND target)
  - Predict on held-out test nodes; predictions saved in native target units

Output:
  outputs/results_gnn.csv
    columns: fold, model, graph_type, lsoa_code, y_true, y_pred
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import add_self_loops, coalesce
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LSOA_FEATURES_PQ, GRAPH_PT, FOLDS_PT, RESULTS_GNN_CSV,
    OUTPUTS_DIR,
    FEATURE_COLS, TARGET_COL, IMD_SIMILARITY_COL,
    GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_DROPOUT, GNN_LR, GNN_EPOCHS,
    IMD_KNN_K, RANDOM_SEED,
)

# GAT was still descending at GNN_EPOCHS in the contiguity run; give it more
# budget. Set to 1 for an identical-epoch comparison (and raise GNN_LR instead).
EPOCH_MULTIPLIER = {"GraphSAGE": 1, "GAT": 2}

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_DIR.mkdir(exist_ok=True)
print(f"Device: {DEVICE}")

# ── 1. Load features and graph ────────────────────────────────────────────────
print("Loading features...")
df = pd.read_parquet(LSOA_FEATURES_PQ)
lsoa_codes = df["LSOA21CD"].tolist()
X_raw = df[FEATURE_COLS].values.astype(np.float32)
y_raw = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
print(f"  X: {X_raw.shape}  y: {y_raw.shape}")

print("Loading graph...")
graph_data = torch.load(GRAPH_PT, weights_only=False)
edge_index_contiguity = graph_data["edge_index"]
print(f"  Contiguity edges: {edge_index_contiguity.shape[1]:,}")

# ── 2. Build IMD k-NN edges ───────────────────────────────────────────────────
# Similarity metric: Mean Household Deprivation (already in features parquet)
print(f"Building IMD k-NN edges using '{IMD_SIMILARITY_COL}'...")
imd_scores = df[IMD_SIMILARITY_COL].values.reshape(-1, 1).astype(np.float32)
print(f"  Similarity values: min={imd_scores.min():.4f}  max={imd_scores.max():.4f}")
nn = NearestNeighbors(n_neighbors=IMD_KNN_K + 1, metric="euclidean", n_jobs=-1)
nn.fit(imd_scores)
distances, indices = nn.kneighbors(imd_scores)

knn_rows, knn_cols = [], []
for i, neighbours in enumerate(indices):
    for j in neighbours[1:]:   # skip self (index 0)
        knn_rows.append(i)
        knn_cols.append(j)
        knn_rows.append(j)     # add reverse edge — undirected
        knn_cols.append(i)

edge_index_knn = torch.tensor([knn_rows, knn_cols], dtype=torch.long)
n_nodes = len(lsoa_codes)
edge_index_knn = coalesce(edge_index_knn, num_nodes=n_nodes)
print(f"  IMD k-NN edges (k={IMD_KNN_K}): {edge_index_knn.shape[1]:,}")

edge_index_hetero = coalesce(
    torch.cat([edge_index_contiguity, edge_index_knn], dim=1),
    num_nodes=n_nodes,
)
print(f"  Heterogeneous edges (contiguity + k-NN): {edge_index_hetero.shape[1]:,}")

EDGE_INDICES = {
    "contiguity":    edge_index_contiguity,
    "heterogeneous": edge_index_hetero,
}

# ── 3. Model definitions ──────────────────────────────────────────────────────
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index).squeeze(-1)


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, heads=4):
        super().__init__()
        assert hidden_dim % heads == 0, (
            f"GNN_HIDDEN_DIM ({hidden_dim}) must be divisible by heads ({heads}); "
            f"otherwise layer widths silently mismatch."
        )
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads,
                                  dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads,
                                      heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim, 1, heads=1, concat=False,
                                  dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index).squeeze(-1)


def make_models(in_dim):
    return {
        "GraphSAGE": GraphSAGE(in_dim, GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_DROPOUT),
        "GAT":       GAT(in_dim, GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_DROPOUT),
    }


# ── 4. Training loop ──────────────────────────────────────────────────────────
print("Loading folds...")
folds = torch.load(FOLDS_PT, weights_only=False)
print(f"  {len(folds)} folds")

records = []

for graph_type, edge_index in EDGE_INDICES.items():
    # Add self-loops so island nodes (e.g. Isles of Scilly) can aggregate
    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=n_nodes)
    edge_index_sl = edge_index_sl.to(DEVICE)

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train_idx_np = train_idx.numpy()
        test_idx_np  = test_idx.numpy()

        # Scale FEATURES on train fold only
        x_scaler = StandardScaler()
        X_scaled = X_raw.copy()
        X_scaled[train_idx_np] = x_scaler.fit_transform(X_raw[train_idx_np])
        X_scaled[test_idx_np]  = x_scaler.transform(X_raw[test_idx_np])

        # Scale TARGET on train fold only — fixes GAT convergence and makes
        # losses comparable. Predictions inverse-transformed before saving.
        y_scaler = StandardScaler()
        y_scaler.fit(y_raw[train_idx_np])
        y_scaled = y_scaler.transform(y_raw).astype(np.float32).ravel()

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(DEVICE)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx]   = True
        train_mask = train_mask.to(DEVICE)
        test_mask  = test_mask.to(DEVICE)

        # Native-unit target for test nodes, for RMSE reporting
        y_true_native = y_raw[test_idx_np].ravel()

        print(f"\n[{graph_type}] Fold {fold_id}  "
              f"train={train_idx.shape[0]:,}  test={test_idx.shape[0]:,}")

        for model_name, model in make_models(len(FEATURE_COLS)).items():
            model = model.to(DEVICE)
            optimiser = torch.optim.Adam(model.parameters(), lr=GNN_LR)
            n_epochs = GNN_EPOCHS * EPOCH_MULTIPLIER[model_name]

            # Training
            model.train()
            for epoch in range(n_epochs):
                optimiser.zero_grad()
                out = model(x_tensor, edge_index_sl)
                loss = F.mse_loss(out[train_mask], y_tensor[train_mask])
                loss.backward()
                optimiser.step()
                if (epoch + 1) % 50 == 0:
                    print(f"  {model_name} epoch {epoch+1:3d}  "
                          f"train_loss={loss.item():.4f}")

            # Inference
            model.eval()
            with torch.no_grad():
                preds_scaled = model(x_tensor, edge_index_sl)[test_mask].cpu().numpy()

            # Back to native units
            preds_native = y_scaler.inverse_transform(
                preds_scaled.reshape(-1, 1)
            ).ravel()

            test_rmse = float(np.sqrt(np.mean((preds_native - y_true_native) ** 2)))
            print(f"  {model_name} TEST RMSE (native units): {test_rmse:.4f}")

            for i, idx in enumerate(test_idx_np):
                records.append({
                    "fold":       fold_id,
                    "model":      model_name,
                    "graph_type": graph_type,
                    "lsoa_code":  lsoa_codes[idx],
                    "y_true":     float(y_true_native[i]),
                    "y_pred":     float(preds_native[i]),
                })

# ── 5. Save ───────────────────────────────────────────────────────────────────
results = pd.DataFrame(records)
results.to_csv(RESULTS_GNN_CSV, index=False)
print(f"\nSaved: {RESULTS_GNN_CSV}")
print(f"  Rows:        {len(results):,}")
print(f"  Models:      {results['model'].unique().tolist()}")
print(f"  Graph types: {results['graph_type'].unique().tolist()}")
print(f"  Folds:       {sorted(results['fold'].unique().tolist())}")

# ── 6. Summary table ──────────────────────────────────────────────────────────
results["se"] = (results["y_true"] - results["y_pred"]) ** 2
summary = (results.groupby(["graph_type", "model"])["se"]
                  .mean().pow(0.5).rename("test_RMSE").reset_index())
print("\nTest RMSE by graph type and model:")
print(summary.to_string(index=False))
