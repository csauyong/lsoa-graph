#!/bin/bash -l
# Run once on Myriad after transferring files.
# Sets up the conda environment and directory structure.

set -e

cd ~/Scratch/lsoa-graph
mkdir -p outputs logs

module -q load python/miniconda3/4.10.3

conda create -n epc_gnn python=3.11 -y
conda activate epc_gnn

pip install \
    geopandas \
    libpysal \
    esda \
    xgboost \
    scikit-learn \
    pandas \
    numpy"<2" \
    pyarrow \
    matplotlib \
    requests \
    "numba>=0.60"

# PyTorch + PyG — CPU build (change cu118/cu121 if GPU node has CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric

echo "Environment ready."
python -c "import torch, geopandas, libpysal, xgboost, torch_geometric; print('All imports OK')"
