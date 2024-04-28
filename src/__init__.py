from .utils import _name_to_class as name_to_class
import yaml

def load_configs(path):
    with open(path, 'r') as f:
        configs = yaml.full_load(f)
    return configs

def save_configs(path, mdict):
    with open(path, 'w') as f:
        yaml.dump(mdict, f, default_style=False, sort_keys=False)