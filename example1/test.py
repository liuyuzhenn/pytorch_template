import argparse
import os
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('.')
from torch_temp.utils import test as ttest
from torch_temp.config import DictAction, update_configs

parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, default='./configs/example1.yml')
args = parser.parse_args()

with open(args.configs, 'r') as f:
    configs = yaml.full_load(f)
    update_configs(configs, args.cfg_options)

ttest(configs)
