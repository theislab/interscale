import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

pancreas_cell_type_coarse_colors = [
    '#edd892', #acinar
    '#219ebc', #alpha
    '#adc178', #beta
    '#f79824', #ductal
    '#718355', #endocrine
    '#f15156', #endothelial
    '#bbd0ff', #Fibroblasts
    '#979dac', #Immune
    '#582f0e'  #mast
]

class CustomColormap:
    def __init__(self, categories, none_color='#D3D3D3', cmap_name='custom_colormap'):
        """
        Initialize the CustomColormap object.

        Parameters:
        - categories (list): List of category names (including 'None').
        - none_color (str, optional): Color code for 'None' category. Default is '#D3D3D3' (light grey).
        - cmap_name (str, optional): Name of the custom colormap. Default is 'custom_colormap'.
        """
        self.categories = categories
        self.none_color = none_color
        self.cmap_name = cmap_name
        self.category_colors = self._generate_category_colors()

        # Create and register the colormap
        self._create_colormap()

    def _generate_category_colors(self):
        """
        Generate category colors including 'None'.
        """
        # Generate a list of evenly spaced colors
        num_categories = len(self.categories)
        cmap = plt.cm.get_cmap('tab10', num_categories)

        # Assign colors to each category
        category_colors = {}
        for i, cat in enumerate(self.categories):
            if cat == 'None':
                category_colors[cat] = self.none_color
            else:
                category_colors[cat] = mcolors.rgb2hex(cmap(i)[:3])

        return category_colors

    def _create_colormap(self):
        """
        Create and register the custom colormap.
        """
        cmap = mcolors.ListedColormap([self.category_colors[cat] for cat in self.categories])
        mpl.cm.unregister_cmap(self.cmap_name)
        plt.register_cmap(name=self.cmap_name, cmap=cmap)

    def get_colormap_name(self):
        """
        Return the name of the custom colormap.
        """
        return self.cmap_name

    def get_category_colors(self):
        """
        Return the dictionary of category colors in #RRGGBB format.
        """
        return self.category_colors

    def save_uns(self, adata, label):
        """
        Saves the correct color assignment for adata given .obs[label]
        """
        classes = np.unique(adata.obs[label])
        uns = [self.category_colors[c] for c in classes]
        color_label = label + '_colors'
        adata.uns[color_label] = uns
        return adata