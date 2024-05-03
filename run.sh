#!/bin/bash
#SBATCH --job-name=gt_long_range
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300GB
#SBATCH --time=10:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source activate exphormer_mamba

# srun necessary for running lightning on SLURM
srun python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_gnn_ct_lung5.yaml
srun python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_gnntrans_ct_lung5.yaml
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_cosmx_human_lung_0_niche_gnn.yaml