import os
t = '6'
os.environ['MKL_NUM_THREADS']=f'{t}'
os.environ['NUMEXPR_NUM_THREADS']=f'{t}'
os.environ['OMP_NUM_THREADS']=f'{t}'

import random
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
import torch.distributed
from src.utils import train as ttrain


@hydra.main(version_base=None, config_path='configs')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    seed = cfg.get('seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if int(os.environ.get('LOCAL_RANK', -1)) >= 0:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', -1)))
        torch.distributed.init_process_group('nccl', init_method='env://')

    ttrain(cfg)

if __name__ == '__main__':
    main()
