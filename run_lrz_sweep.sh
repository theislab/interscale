#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem=200GB

# Shell logic
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="logs/${TIMESTAMP}_${SLURM_JOB_ID}.out"

mkdir -p logs
exec > >(tee "${LOG_FILE}") 2>&1

# Set wandb API key
export WANDB_API_KEY="45b9c9a439c12187aa03a740a0cacad57dcf958f"

# Default config path
DEFAULT_CONFIG_ICB="/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/"
DEFAULT_CONFIG_LRZ="/dss/dsshome1/05/di93tig/1_projects/GT-long-range-niches/src/config_files/"

LEGNINI_CONFIG="Legnini_23/legnini23_graph_sample_LocalModel_gnn.yaml"
SCHUERCH_CONFIG="Schuerch20/schuerch20_graph_sample_GlobalModel.yaml"
SCHUERCH_SWEEP="Schuerch20/schurch20_hyperparam_sweep.yaml"

echo "Current working directory: $(pwd)"
ls -l /dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/schuerch20.h5ad
ls -ld /dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data

srun -N1 --ntasks-per-node=1 --container-mounts=/dss/dsshome1/05/di93tig/1_projects/GT-long-range-niches:/workspace,/dss/dssfs03:/dss/dssfs03 \
     --container-image='/dss/dsshome1/05/di93tig/InterScale.sqsh' \
     python src/InterScale/main_sweep.py  --cfg "$DEFAULT_CONFIG_LRZ$SCHUERCH_CONFIG" --model_type "LocalModel" --sweep_cfg "$DEFAULT_CONFIG_LRZ$SCHUERCH_SWEEP"  --sweep_goal hyperparameter --prediction_task classification


# # srun necessary for running lightning on SLURM
# python src/InterScale/main.py --cfg "$DEFAULT_CONFIG_LRZ$LEGNINI_CONFIG"