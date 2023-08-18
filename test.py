import argparse
import os
import yaml

import torch
from torch_temp.utils import test as ttest
from torch_temp.config import DictAction, update_configs

parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, default='./configs/example1.yml')

args = parser.parse_args()

with open(args.configs, 'r') as f:
    configs = yaml.full_load(f)
    update_configs(configs, args.cfg_options)

if int(os.environ.get('LOCAL_RANK', -1)) >= 0:
    torch.distributed.init_process_group('nccl', init_method='env://')

ttest(configs)
