import yaml
import os
import logging
from importlib import import_module


def _name_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))


def get_logger(logdir):
    logger = logging.getLogger('torch_temp')
    handler = logging.FileHandler(os.path.join(logdir, 'info.log'), 'w')
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
    # save the configutation in the log directory
    if configs.get('save_configs', True):
        workspace = configs['train_configs']['workspace']
        if not os.path.isdir(workspace):
            os.makedirs(workspace)
        with open(os.path.join(workspace, 'configs.yml'), 'w') as f:
            yaml.dump(configs, f, default_style=False)

    project = configs.get('project', 'torch_temp')
    train_configs = configs['train_configs']

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
    project = configs.get('project', 'torch_temp')
    test_configs = configs['test_configs']

    runner = import_module('.runners.{}'.format(
        test_configs['name']), project)
    runner = getattr(runner, _name_to_class(
        test_configs['name']))(configs)

    runner.test()
