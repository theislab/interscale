#!/bin/bash
#SBATCH --job-name=CV_Pancreas_GNN
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300GB
#SBATCH --time=10:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source activate GT_long_range_env

## COSMX PANCREAS

# GNN (K fold = 5)
#python src/graph_transformer_long_range_niches/main_cv.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_gnn_ct_sw_2_cv.yaml
#python src/graph_transformer_long_range_niches/main_cv.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_gnntrans_ct_sw_2_cv.yaml
python src/graph_transformer_long_range_niches/main_cv.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_pcatrans_ct_sw_2_cv.yaml