#!/bin/bash
<<<<<<< HEAD
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -J InterScale
=======
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
>>>>>>> e4357f3a4533dad82219312865f6eaf124a545e9

# Shell logic
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="logs/${TIMESTAMP}_${SLURM_JOB_ID}.out"

mkdir -p logs
exec > >(tee "${LOG_FILE}") 2>&1

<<<<<<< HEAD
=======
## Enroot container
#srun --pty --container-image='/dss/dsshome1/05/di93tig/InterScale_scvi.sqsh' bash # will execute bash on a container created out of your custom Enroot container image

>>>>>>> e4357f3a4533dad82219312865f6eaf124a545e9
# Set wandb API key
export WANDB_API_KEY="45b9c9a439c12187aa03a740a0cacad57dcf958f"

# Default config path
DEFAULT_CONFIG_ICB="/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/"
DEFAULT_CONFIG_LRZ="/dss/dsshome1/05/di93tig/1_projects/GT-long-range-niches/src/config_files/"

<<<<<<< HEAD
#LEGNINI_CONFIG="Legnini23/legnini23_regr_node_Combined_genes.yaml"
LEGNINI_CONFIG="Legnini23/legnini23_class_graph_Combined_condition.yaml"
SCHUERCH_CONFIG="Schuerch20/schuerch20_graph_sample_GlobalModel.yaml"
COSMX_PANCREAS_CONFIG="Cosmx_pancreas/clas_graph.yaml"
XENIUM_PIG_CONFIG="/Xenium_pig/pig_genes_sw_combined.yaml"
=======
LEGNINI_CONFIG="Legnini_23/legnini23_graph_sample_LocalModel_gnn.yaml"
SCHUERCH_CONFIG="Schuerch20/schuerch20_graph_sample_GlobalModel.yaml"
COSMX_PANCREAS_CONFIG=""
>>>>>>> e4357f3a4533dad82219312865f6eaf124a545e9

echo "Current working directory: $(pwd)"
ls -l /dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/schuerch20.h5ad
ls -ld /dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data

# Memory monitoring
echo "=== System Memory Info ==="
free -h
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_MEM_PER_CPU: $SLURM_MEM_PER_CPU"
echo "=========================="

srun -N1 --ntasks-per-node=1 --container-mounts=/dss/dsshome1/05/di93tig/1_projects/GT-long-range-niches:/workspace,/dss/dssfs03:/dss/dssfs03 \
<<<<<<< HEAD
	             --container-image='/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/custom-enroot-image/InterScale.sqsh' \
                          python src/InterScale/main.py  --cfg "$DEFAULT_CONFIG_LRZ$COSMX_PANCREAS_CONFIG" --model_type "CombinedModel"

#srun -N1 --ntasks-per-node=1 --container-mounts=/dss/dsshome1/05/di93tig/1_projects/GT-long-range-niches:/workspace,/dss/dssfs03:/dss/dssfs03 \
	           #  --container-image='/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/custom-enroot-image/InterScale.sqsh'
		  #python src/InterScale/main_sweep.py --cfg "$DEFAULT_CONFIG_LRZ$LEGNINI_CONFIG" --sweep_cfg "{$DEFAULT_CONFIG_LR}Cosmx_pancreas/sweep_config_loss.yaml" --model_type "GlobalModel"  --sweep_goal loss --prediction_task regression
=======
     --container-image='/dss/dsshome1/05/di93tig/InterScale.sqsh' \
     python src/InterScale/main.py  --cfg "$DEFAULT_CONFIG_LRZ$SCHUERCH_CONFIG" --model_type "LocalModel"
>>>>>>> e4357f3a4533dad82219312865f6eaf124a545e9

# # srun necessary for running lightning on SLURM
# python src/InterScale/main.py --cfg "$DEFAULT_CONFIG_LRZ$LEGNINI_CONFIG"
