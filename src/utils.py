import os
import logging
from importlib import import_module
from omegaconf import OmegaConf


def _name_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))


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

    project = configs.get('project', 'src')

    runner = import_module('.runners.{}'.format(
        train_configs['name']), project)
    runner = getattr(runner, _name_to_class(
        train_configs['name']))(configs)

    try:
        runner.train()
    except KeyboardInterrupt:
        runner.logger.info(
            'Got Keyboard Interuption, saving model and closing.')
        runner.save(train_configs['workspace'], 'interrupt_ckpt.pth')


def test(configs):
    project = configs.get('project', 'src')
    test_configs = configs.test

    runner = import_module('.runners.{}'.format(
        test_configs.name), project)
    runner = getattr(runner, _name_to_class(
        test_configs.name))(configs)

    runner.test()
