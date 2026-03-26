"""Evaluation tools."""

from ._gene_loadings import gene_loadings
from ._gene_set_covariance import gene_set_covariance, spatial_covariance_plot
from .clustering import leiden_cluster_embeddings, plot_clustering_metrics
from .gene_rank_analysis import predict_gene_r2, gene_rank_analysis, gene_rank_condition_comparison
from .graph_classification import calculate_pr_auc, pr_auc_curve, scale_cls_by_sample

__all__ = [
    "leiden_cluster_embeddings",
    "plot_clustering_metrics",
    "predict_gene_r2",
    "gene_rank_analysis",
    "gene_rank_condition_comparison",
    "calculate_pr_auc",
    "scale_cls_by_sample",
    "Wandb_evaluation",
]