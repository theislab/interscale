"""Evaluation tools."""

from ._gene_loadings import gene_loadings
from ._gene_set_covariance import gene_set_covariance, spatial_covariance_plot
from ._latent_analysis import get_genes_dim, latent_rank_report
from .gene_rank_analysis import gene_rank_analysis, predict_gene_r2
from .graph_classification import calculate_pr_auc, pr_auc_curve, scale_cls_by_sample

__all__ = [
    "gene_loadings",
    "gene_set_covariance",
    "pr_auc_curve",
    "spatial_covariance_plot",
    "predict_gene_r2",
    "calculate_pr_auc",
    "scale_cls_by_sample",
    "latent_rank_report",
    "get_genes_dim",
    "gene_rank_analysis",
]
