import wandb
import torch
from sklearn.model_selection import train_test_split

# from graph_transformer_long_range_niches.pp.geome_utils import prepare_geome_dataset, load_pyg_data

def log_data(datasets, names, cfg, run):
    """Load dataset and log it as artifact to WandB.
    """

    print("Log data artifact...")

    assert run is wandb.run

    artifact_name = f"{cfg.dataset.name}_{cfg.dataset.prediction_obs}_{cfg.dataset.library_key[-1]}"

    # create Artifact
    raw_data = wandb.Artifact(
        artifact_name, type="dataset",
        description=cfg.dataset.description,
        metadata={"source": cfg.dataset.h5ad_data,
                    "sizes": [len(dataset) for dataset in datasets]})

    for name, data in zip(names, datasets):
        # Store a new file in the artifact, and write something into its contents.
        with raw_data.new_file(name + ".pt", mode="wb") as file:
            torch.save(data, file)

    # ✍️ Save the artifact to W&B.
    run.log_artifact(raw_data)

# def load_and_log(cfg):
#     """Load dataset and log it as artifact to WandB.
#     """

#     with wandb.init(project=cfg.wandb.project_name, job_type="load-data", name = 'data_'+cfg.dataset.name) as run:
        
#         datasets, names = load_pyg_data(cfg)  # separate code for loading the datasets

#         # create Artifact
#         raw_data = wandb.Artifact(
#             cfg.dataset.name, type="dataset",
#             description=cfg.dataset.description,
#             metadata={"source": cfg.dataset.h5ad_data,
#                       "sizes": [len(dataset) for dataset in datasets]})

#         for name, data in zip(names, datasets):
#             # Store a new file in the artifact, and write something into its contents.
#             with raw_data.new_file(name + ".pt", mode="wb") as file:
#                 torch.save(data, file)

#         # ✍️ Save the artifact to W&B.
#         run.log_artifact(raw_data)

