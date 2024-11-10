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

# srun necessary for running lightning on SLURM
## HE23 LUNG
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/He23/he23_gnntrans_graph_condition.yaml #he23_gnn_ct_lung5.yaml #he23_gnn_ct.yaml #he23_gnn_niche_lung5.yaml
#python src/graph_transformer_long_range_niches/main_cv.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/He23/he23_gnntrans_ct.yaml
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/He23/he23_gnntrans_niche.yaml
#srun python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/He23/he23_gnntrans_niche.yaml #he23_gnntrans_ct_lung5.yaml #he23_gnntrans_ct.yaml #he23_gnntrans_niche_lung5.yaml
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/He23/he23_gnn_niche.yaml

## PANCREAS
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/pancreas_gnntrans_graph_condition.yaml
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/pancreas_gnn_graph_condition.yaml
#python src/graph_transformer_long_range_niches/main_cv.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/pancreas_gnntrans_ct.yaml
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/pancreas_gnn_ct.yaml
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/pancreas_pcatrans_ct.yaml

## BASELINE
#srun python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_fcnn_niche_lung5.yaml

## VISIUM BREAST CANCER
#python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/VisiumBreastCancer/visium_gnntrans_genes.yaml
python src/graph_transformer_long_range_niches/main.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/VisiumBreastCancer/visium_gnn_genes.yaml