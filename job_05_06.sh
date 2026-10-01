#!/bin/bash -l
#$ -N epc_gnn_baselines
#$ -l h_rt=8:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -pe smp 4
#$ -ac allow=L
#$ -wd /home/ucbvauy/Scratch/lsoa-graph
#$ -o /home/ucbvauy/Scratch/lsoa-graph/logs/gnn_$JOB_ID.out
#$ -e /home/ucbvauy/Scratch/lsoa-graph/logs/gnn_$JOB_ID.err
#$ -m bea
#$ -M ucbvauy@ucl.ac.uk

# ── Modules (must match what the venv was built with) ──
module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load cuda/12.2.2/gnu-10.2.0
module load python/3.11.4-gnu-10.2.0

# ── Neutralise central-bundle overrides so the venv isn't shadowed ──
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# ── Activate the GPU virtualenv ──
source ~/envs/geo-gpu/bin/activate

# ── Sanity check: confirm CUDA is visible, fail fast if not ──
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

cd /home/ucbvauy/Scratch/lsoa-graph
python scripts/05_train_baselines.py && python scripts/06_train_gnn.py
