#!/bin/bash
#SBATCH --job-name=InterScale_sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source activate GT_long_range_env

python src/InterScale/main_sweep.py --cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/Legnini_23/legnini23_genes_sample_GlobalModel.yaml --sweep_cfg /home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/Legnini_23/sweep/loss.yaml --model_type GlobalModel --sweep_goal loss --prediction_task regression
python src/InterScale/main_sweep.py \
                            --cfg "${DEFAULT_CONFIG_LRZ}${COSMX_PANCREAS_CONFIG}" \
                                 --sweep_cfg "${DEFAULT_CONFIG_LRZ}Cosmx_pancreas/sweep/robustness.yaml" \
                                      --model_type "CombinedModel" \
                                           --sweep_goal robustness \
                                                --prediction_task classification
