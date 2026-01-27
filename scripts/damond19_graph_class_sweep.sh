#!/bin/bash
#SBATCH --job-name=InterScale_sweep_graph_clas
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source activate GT_long_range_env

# Default config path
DEFAULT_CONFIG_ICB="/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/"
DEFAULT_CONFIG_LRZ="/dss/dsshome1/05/di93tig/1_projects/GT-long-range-niches/src/config_files/"

DAMOND_SWEEP="damond19/sweep/hyperparameter.yaml"
DAMOND_CFG="damond19/damond19_class_graph_Combined_condition.yaml"

# Legnini graph sweep for Local, Global and Combined Model
#python src/InterScale/main_sweep.py --cfg "$DEFAULT_CONFIG_ICB$DAMOND_CFG" --sweep_cfg "$DEFAULT_CONFIG_ICB$DAMOND_SWEEP" --model_type LocalModel --sweep_goal hyperparameter --prediction_task classification
python src/InterScale/main_sweep.py --cfg "$DEFAULT_CONFIG_ICB$DAMOND_CFG" --sweep_cfg "$DEFAULT_CONFIG_ICB$DAMOND_SWEEP" --model_type GlobalModel --sweep_goal hyperparameter --prediction_task classification
python src/InterScale/main_sweep.py --cfg "$DEFAULT_CONFIG_ICB$DAMOND_CFG" --sweep_cfg "$DEFAULT_CONFIG_ICB$DAMOND_SWEEP" --model_type CombinedModel --sweep_goal hyperparameter --prediction_task classification
