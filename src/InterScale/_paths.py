from pathlib import Path

ROOT = Path(__file__).parent.resolve()

HOME_ICB = Path("/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/")
HOME_LRZ = Path("/dss/dsshome1/05/di93tig/1_projects/GT-long-range-niches/")

PROJECT_ICB = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/")
PROJECT_LRZ = Path("/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/")

DEFAULT_CONFIG_ICB= Path(HOME_ICB, "/src/config_files/")
DEFAULT_CONFIG_LRZ= Path(HOME_LRZ, "/src/config_files/")

def get_default_config(cluster: str):
    if cluster == "icb":
        return Path(HOME_ICB, "/src/config_files/")
    elif cluster == "lrz":
        return Path(HOME_LRZ, "/src/config_files/")
    else:
        raise ValueError(f"Cluster {cluster} not supported")

def get_default_data_path(cluster: str):
    if cluster == "icb":
        return Path(PROJECT_ICB, '/data/')
    elif cluster == "lrz":
        return Path(PROJECT_LRZ, '/data/')
    else:
        raise ValueError(f"Cluster {cluster} not supported")

LEGNINI_CONFIG="Legnini23/legnini23_genes_sample_GlobalModel.yaml"
SCHUERCH_CONFIG="Schuerch20/schuerch20_graph_sample_GlobalModel.yaml"
COSMX_PANCREAS_CONFIG="Cosmx_pancreas/pancreas_genes_sw_gnn.yaml"

nicheformer_database = Path('/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/spatial/preprocessed/human/nanostring_lung_annotated.h5ad')
HE22_HUMAN_LUNG_DATA_PATH = Path('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/he22_cosmx_human_lung.h5ad')

sara_spatial_pancreas = Path('/lustre/groups/ml01/datasets/projects/20230301_Sander_SpatialPancreas_sara.jimenez/spatial/S1_annotated_l0.h5ad')
COSMX_PANCREAS_S1 = Path('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/cosmx_pancreas_s1.h5ad') # 1 slide
COSMX_PANCREAS = Path('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/cosmx_pancreas.h5ad') # 3 slides

LEGNINI23 = "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/legnini23.h5ad"

FIG_PATH = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/figures'

RESULTS = Path('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results')