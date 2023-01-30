import argparse
import numpy as np
import os
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import sys
sys.path.append('.')
from torch_temp.utils import train as ttrain

parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, default='./configs/example1.yml')
args = parser.parse_args()

with open(args.configs, 'r') as f:
    configs = yaml.full_load(f)

seed = configs.get('seed',0)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

ttrain(configs)
