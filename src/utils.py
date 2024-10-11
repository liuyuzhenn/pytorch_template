import os
import logging
from importlib import import_module
from omegaconf import OmegaConf


# def _name_to_class(name):
#     return ''.join(n.capitalize() for n in name.split('_'))

def build_instance(s, cfg):
    items = s.split('.')
    base_path = items[:-1]
    base_path = '.'.join(base_path)
    cls_name = items[-1]
    instance = getattr(import_module(base_path), cls_name)(cfg)
    return instance

def get_cls(s):
    items = s.split('.')
    base_path = items[:-1]
    base_path = '.'.join(base_path)
    cls_name = items[-1]
    cl = getattr(import_module(base_path), cls_name)
    return cl


def get_logger(logdir):
    logger = logging.getLogger('torch_template')
    handler = logging.FileHandler(os.path.join(logdir, 'info.log'), 'w')
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter(fmt='[%(asctime)s] - %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    logger.setLevel(level=logging.INFO)
    return logger


def train(configs):
    # save the configutation in the log directory
    OmegaConf.resolve(configs)
    train_configs = configs.train
    workspace = train_configs.workspace
    os.makedirs(workspace, exist_ok=True)
    path = os.path.join(workspace, 'configs.yml')
    OmegaConf.save(configs, path)

    runner = build_instance(train_configs.__target__, configs)

    try:
        runner.train()
    except KeyboardInterrupt:
        runner.logger.info(
            'Got Keyboard Interuption, saving model and closing.')
        runner.save(os.path.join(workspace, 'checkpoints'), 'interrupt_ckpt.pth')


def test(configs):
    test_configs = configs.test

    runner = build_instance(test_configs.__target__, configs)

    runner.test()
