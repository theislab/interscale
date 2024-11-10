from pathlib import Path

ROOT = Path(__file__).parent.resolve()

nicheformer_database = Path('/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/spatial/preprocessed/human/nanostring_lung_annotated.h5ad')
HE22_HUMAN_LUNG_DATA_PATH = Path('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/he22_cosmx_human_lung.h5ad')

sara_spatial_pancreas = Path('/lustre/groups/ml01/datasets/projects/20230301_Sander_SpatialPancreas_sara.jimenez/spatial/S1_annotated_l0.h5ad')
COSMX_PANCREAS = Path('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/jimenz_spatial_pancreas.h5ad')

CFG_FILES = Path('/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files')
FIG_PATH = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/figures'

RESULTS = Path('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results')