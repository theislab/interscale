"""Evaluation tools."""

from ._gene_loadings import gene_loadings
from ._gene_rank_analysis import calculate_gene_ranks
from ._gene_set_covariance import gene_set_covariance, spatial_covariance_plot
from ._latent_analysis import get_genes_dim, latent_rank_report, calculate_dim_importance, _get_Z
from .graph_classification import calculate_pr_auc, pr_auc_curve, scale_cls_by_sample
from .net_streams import plot_all_spatial_net_streams, plot_flow_clusters, plot_global_directionality

__all__ = [
    "gene_loadings",
    "gene_set_covariance",
    "pr_auc_curve",
    "spatial_covariance_plot",
    "calculate_gene_ranks",
    "calculate_pr_auc",
    "scale_cls_by_sample",
    "latent_rank_report",
    "get_genes_dim",
    "plot_all_spatial_net_streams",
    "plot_flow_clusters",
    "plot_global_directionality",
    "_get_Z",
    "calculate_dim_importance",
]
