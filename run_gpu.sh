#!/bin/bash
#SBATCH --job-name=gt_long_range_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=240GB
#SBATCH --time=5:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --nice=10000

source activate GT_long_range_env #exphormer_geome

# srun necessary for running lightning on SLURM
#srun python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_cosmx_human_lung_ct.yaml
srun python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/pancreas_gnntrans_ct.yaml
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_cosmx_human_lung_ct_gnn.yaml
