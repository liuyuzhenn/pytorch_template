import argparse
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
import yaml
import logging
import torch
from models import get_model
from losses import get_loss
from datasets import get_dataset

def get_logger(logdir):
    logger = logging.getLogger('default')
    handler = logging.FileHandler(os.path.join(logdir, 'info.log'),'w')
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter(fmt='[%(asctime)s] %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    logger.setLevel(level=logging.INFO)
    return logger

def train(configs):

    model_args = configs['model_args']
    train_args = configs['train_args']
    dataset_args = configs['dataset_args']

    out_dir = train_args['log_dir']
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    loss = get_loss(train_args['loss'])(train_args['loss_args'])
    dataset = get_dataset(dataset_args['name'])(dataset_args)

    model = get_model(model_args['name'])(dataset,model_args)

    logger = get_logger(train_args['log_dir'])
    logger.info('Start to train...')
    try:
        model.train(train_args,loss)
    except KeyboardInterrupt:
        model.logger.info(
            'Got Keyboard Interuption, saving model and closing.')
        model.save(train_args['log_dir'],'interrupt_ckpt.pt')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='./src/configs/default.yml')
    args = parser.parse_args()

    with open(args.configs, 'r') as f:
        configs = yaml.full_load(f)
    
    seed = configs.get('seed',0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train(configs)