import numpy as np
from anndata import AnnData
from graph_transformer_long_range_niches.pl.ligand_receptor import ligand_receptor_interaction

# Create mock AnnData with ligand and receptor genes
n_cells = 100
n_genes = 5

# Create a simple AnnData object with random data
gene_names = ["GeneA", "GeneB", "LigandGene", "ReceptorGene", "GeneE"]
cell_ids = [f"Cell{i}" for i in range(n_cells)]
data = np.random.rand(n_cells, n_genes)  # Random expression values for simplicity
adata = AnnData(X=data, obs={"cell_id": cell_ids}, var={"var_names": gene_names})

# Set expression values for ligand and receptor specifically
adata[:, "LigandGene"].X = np.random.rand(n_cells)  # Random expression values for LigandGene
adata[:, "ReceptorGene"].X = np.random.rand(n_cells)  # Random expression values for ReceptorGene

# Add spatial information to the AnnData object (required by Squidpy)
# Random x, y coordinates for simplicity
adata.obsm["spatial"] = np.random.rand(n_cells, 2)

ligand_receptor_interaction(
    adata=adata,
    ligand="LigandGene",
    receptor="ReceptorGene",
    threshold=0.3,
)