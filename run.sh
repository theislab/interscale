#!/bin/bash
#SBATCH --job-name=gt_long_range
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300GB
#SBATCH --time=5:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source activate GT_long_range_env

# Default config path
DEFAULT_CONFIG="/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/"
LEGNINI_CONFIG="Legnini_23/legnini23_graph_sample_LocalModel_gnn.yaml"

# srun necessary for running lightning on SLURM
python src/InterScale/main.py --cfg "$DEFAULT_CONFIG$LEGNINI_CONFIG"