# src/interscale/pl/config.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import yaml


class Plotting:
    """Class to store plotting configuration and apply matplotlib/scanpy settings"""

    DEFAULT_CONFIG = {
        "plot_configs": {
            "general": {
                "dpi": 75,
                "dpi_save": 300,
                "legend_fontsize": 12,
                "legend_fontweight": "bold",
                "title_fontsize": 14,
                "title_fontweight": "bold",
                "font_family": "DejaVu Sans",
                "cmap": "viridis",
            },
            "embeddings_plots": {
                "alpha": 0.75,
                "frameon": False,
                "add_outline": True,
                "layer": "log_norm",
                "legend_fontsize": 12,
                "outline_color": ["black", "white"],
                "outline_width": [0.2, 0.025],
                "ncols": 1,
                "vmax": "p98",
                "cmap": "viridis",
                "na_color": "white",
                "na_in_legend": False,
            },
            "continuous_plots": {
                "layer": "log_norm",
                "var_group_rotation": 90,
                "cmap": "viridis",
            },
            "rank_genes_plots": {
                "values_to_plot": "logfoldchanges",
                "vmin": -5,
                "vmax": 5,
                "var_group_rotation": 90,
                "min_logfoldchange": 2,
                "cmap": "bwr",
            },
        }
    }

    def __init__(self, config_path=None, output_dir="figures"):
        """
        Initialize plotting configuration.

        Parameters
        ----------
        config_path : str, dict, or None
            Path to YAML config file, dict with config, or None to use defaults.
            Defaults to None (uses DEFAULT_CONFIG).
        output_dir : str
            Output directory for saving figures. Defaults to "figures".
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_plotting_params()

    def _load_config(self, config_path):
        """Load configuration from YAML file, dict, or use defaults"""
        if config_path is None:
            return self.DEFAULT_CONFIG.copy()
        elif isinstance(config_path, str):
            with open(config_path) as f:
                return yaml.safe_load(f)
        elif isinstance(config_path, dict):
            return config_path
        else:
            raise ValueError("Config must be a file path, dictionary, or None")

    def _setup_plotting_params(self):
        """Set up matplotlib and scanpy plotting parameters"""
        cfg = self.config["plot_configs"]["general"]
        plt.rcParams["figure.dpi"] = cfg["dpi"]
        plt.rcParams["savefig.dpi"] = cfg["dpi_save"]
        plt.rcParams["legend.fontsize"] = cfg["legend_fontsize"]
        plt.rcParams["axes.titlesize"] = cfg["title_fontsize"]
        plt.rcParams["font.family"] = cfg["font_family"]
        cmap_cfg = cfg.get("cmap", "viridis")
        if isinstance(cmap_cfg, str):
            cm = plt.get_cmap(cmap_cfg)
            palette = [cm(x) for x in np.linspace(0.1, 0.9, 10)]
        elif isinstance(cmap_cfg, (list, tuple)):
            palette = list(cmap_cfg)
        else:
            raise ValueError("plot_configs.general.cmap must be a colormap name, not a list or tuple")
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=palette)
        sns.set_theme(style="white", font=cfg["font_family"])
        sc.settings.set_figure_params(dpi_save=cfg["dpi_save"], fontsize=cfg["legend_fontsize"])


class _SettingsMeta(type):
    """Metaclass for singleton settings"""

    _instance = None

    def __call__(cls):
        if cls._instance is None:
            cls._instance = super().__call__()
        return cls._instance


class settings(metaclass=_SettingsMeta):
    """Global settings for plotting functions - singleton pattern matching scanpy's approach"""

    def __init__(self):
        self._plotting_config = Plotting()

    def set_plotting_config(self, config_path=None, output_dir="figures"):
        """
        Set global plotting configuration.

        Parameters
        ----------
        config_path : str, dict, or None
            Path to YAML config file, dict with config, or None to use defaults.
        output_dir : str
            Output directory for saving figures.
        """
        self._plotting_config = Plotting(config_path, output_dir)

    @property
    def plotting_config(self) -> Plotting:
        """Get the current plotting configuration"""
        return self._plotting_config

    @property
    def output_dir(self) -> Path:
        """Get the output directory for figures"""
        return self._plotting_config.output_dir

    @property
    def config(self) -> dict:
        """Get the configuration dictionary"""
        return self._plotting_config.config
