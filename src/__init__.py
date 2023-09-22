from .utils import _name_to_class as name_to_class
import yaml

def load_configs(path):
    with open(path, 'r') as f:
        configs = yaml.full_load(f)
    return configs
