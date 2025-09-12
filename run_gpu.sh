#!/bin/bash
#SBATCH --job-name=InterScale_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --nice=10000

source activate GT_long_range_env

# Default config path
DEFAULT_CONFIG="/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/"
LEGNINI_CONFIG="Legnini23/legnini23_genes_sample_GlobalModel.yaml"
XENIUM_PIG_PANCREAS_CONFIG="Xenium_pig_pancreas/pancreas_regr_sw_CombinedComponent.yaml"

# srun necessary for running lightning on SLURM
python src/InterScale/main.py --cfg "$DEFAULT_CONFIG$LEGNINI_CONFIG" --model_type GlobalModel
