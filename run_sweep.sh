#!/bin/bash
#SBATCH --job-name=InterScale_sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300GB
#SBATCH --time=30:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source activate GT_long_range_env


# HYPERPARAMETER SWEEP
#python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_gnntrans_ct_sw_2.yaml --model_type gnn-transformer
#python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_gnn_ct_sw_2.yaml --model_type gnn
#python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_pcatrans_ct_sw_2.yaml --model_type pca-transformer

# ROBUSTNESS SWEEP
#python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_gnntrans_ct_sw_2.yaml --model_type gnn-transformer --sweep_goal robustness
#python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/Legnini_23/legnini23_InterScale_genes_sample_gnn.yaml --model_type gnn --sweep_goal experiment --prediction_task regression
 
# EXPERIMENT SWEEP
#python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/Legnini_23/legnini23_genes_sample_gnntrans.yaml --model_type gnn-transformer --sweep_goal experiment --prediction_task regression
#python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/Legnini_23/legnini23_genes_sample_gnntrans_crosscell.yaml --model_type gnn-transformer --sweep_goal experiment --prediction_task regression
python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_gnntrans_genes_sw_2.yaml --model_type gnn-transformer --sweep_goal experiment --prediction_task regression
#python src/graph_transformer_long_range_niches/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/CosmX_Pancreas/pancreas_gnn_genes.yaml --model_type gnn --sweep_goal experiment --prediction_task regression
