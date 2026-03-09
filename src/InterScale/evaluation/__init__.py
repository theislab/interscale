"""Evaluation tools."""

from ._gene_loadings import gene_loadings
from ._gene_set_covariance import gene_set_covariance, spatial_covariance_plot
from .clustering import leiden_cluster_embeddings, plot_clustering_metrics
from .gene_rank_analysis import predict_gene_r2
from .graph_classification import calculate_pr_auc, pr_auc_curve, scale_cls_by_sample

__all__ = [
    "leiden_cluster_embeddings",
    "plot_clustering_metrics",
    "predict_gene_r2",
    "calculate_pr_auc",
    "scale_cls_by_sample",
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
    "Wandb_evaluation",
>>>>>>> bfc835b (save config.yaml from wandb sweep)
=======
>>>>>>> 380a98c (remove WandB_evaluation from evaluation)
=======
>>>>>>> 8f175aa459298d5f9d41c0ebfc70f4f7407676fc
]