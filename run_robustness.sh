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

source activate GT_long_range_env

python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/He23/he23_gnntrans_niche_r_10.yaml 
python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/He23/he23_gnntrans_niche_r_30.yaml 