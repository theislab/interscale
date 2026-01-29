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
LEGNINI_CONFIG="Legnini_23/legnini23_genes_sample_GlobalModel.yaml"
MELTON25="melton25/regr_combined.yaml"
#MELTON25="melton25/node_class_combined.yaml"
XENIUM_PIG_PANCREAS_CONFIG="Xenium_pig_pancreas/pancreas_regr_sw_CombinedComponent.yaml"
DAMOND19="damond19/damond19_class_graph_Combined_condition.yaml"

# srun necessary for running lightning on SLURM
python src/InterScale/main.py --cfg "$DEFAULT_CONFIG$DAMOND19" --model_type CombinedModel
