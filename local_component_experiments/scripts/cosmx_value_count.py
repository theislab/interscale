import pandas as pd
import scanpy as sc

RAW = "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/melton25.h5ad"

adata = sc.read_h5ad(RAW, backed="r")
obs = adata.obs
# the same names the prep script builds: slide + "_fov" + FOV number
sample = obs["slide"].astype(str) + "_fov" + obs["fov"].astype(str)
table = pd.DataFrame({"donor": obs["slide_condition"].astype(str), "sample": sample})
print(table.value_counts().sort_index().to_string())
print("Number of FOVs:", sample.nunique())