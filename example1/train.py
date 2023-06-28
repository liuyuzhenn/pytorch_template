import argparse
import numpy as np
import os
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import sys
sys.path.append('.')
from torch_temp.utils import train as ttrain
from torch_temp.config import DictAction, update_configs

parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, default='./configs/example1.yml')
parser.add_argument(
    '-o',
    '--cfg-options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file. If the value to '
    'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    'Note that the quotation marks are necessary and that no white space '
    'is allowed.')
args = parser.parse_args()

with open(args.configs, 'r') as f:
    configs = yaml.full_load(f)
    update_configs(configs, args.cfg_options)

seed = configs.get('seed',0)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

ttrain(configs)
