import graph_transformer_long_range_niches  # noqa, register custom modules

import argparse
import os
import yaml

def load_cmd_args():
    parser = argparse.ArgumentParser(description='LongRange')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    return parser.parse_args()

def load_cfg(cfg_path):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"The config file path {cfg_path} does not exist.")
    
    return yaml.safe_load(cfg_path)


if __name__ == '__main__':
    # Load cmd line args
    args = load_cmd_args()
    # Load config file
    cfg_dict = load_cfg(args.cfg)
    print(cfg_dict['dataloader']['data_path'])