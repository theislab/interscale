import yaml
import os

def merge_dictionaries_recursively(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be preferred
    '''
    if dict2 is None: return

    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            merge_dictionaries_recursively(dict1[k], v)
        else:
            dict1[k] = v

class Config(object):  
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """
    def __init__(self, config_path, default_path=None):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"The config file path {config_path} does not exist.")
        
        with open(config_path) as cf_file:
            print('Load cfg file...')
            cfg = yaml.safe_load( cf_file.read() )

            if default_path is not None:
                with open(default_path) as def_cf_file:
                    default_cfg = yaml.safe_load( def_cf_file.read() )      
                    cfg = {**default_cfg, **cfg}
                    
            self._data = cfg

        print(self._data)

    def get(self, path=None, default=None):
        """
            args:
                path: reads in a argument in the form of path/to/argument from .yaml file 
            default:
                default argument 
        """
        # deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            print(f'{path} not defined in .yaml file.')
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default
        
    def set(self, path, value):
        """
        Set a value in the nested dictionary using the given path.

        Args:
            path (str): The path in the nested dictionary where the value should be set.
                        Example: "meta/dataset_name"
            value: The value to be set at the given path.
        """
        path_items = path.split("/")
        sub_dict = self._data

        for path_item in path_items[:-1]:
            sub_dict = sub_dict.setdefault(path_item, {})

        sub_dict[path_items[-1]] = value

