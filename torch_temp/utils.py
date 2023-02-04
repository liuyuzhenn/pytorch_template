import yaml
import os
import logging
from importlib import import_module as get_module

def _name_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))

def get_logger(logdir):
    logger = logging.getLogger('torch_temp')
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
    # save the configutation in the log directory
    if configs.get('save_configs', True):
        log_dir = configs['train_configs']['log_dir']
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(configs['train_configs']['log_dir'], 'configs.yml'), 'w') as f:
            yaml.dump(configs, f, default_style=False)

    project = configs.get('project', 'torch_temp')

    model_configs = configs['model_configs']
    dataset_configs = configs['dataset_configs']
    loss_configs = configs['loss_configs']
    optimizer_configs = configs['optimizer_configs']
    train_configs = configs['train_configs']

    out_dir = train_configs['log_dir']
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model = get_module('{}.models.{}'.format(project, model_configs['name']), '..')
    dataset = get_module('{}.datasets.{}'.format(project, dataset_configs['name']), '..')
    loss = get_module('{}.losses.{}'.format(project, loss_configs['name']), '..')
    trainer = get_module('{}.trainers.{}'.format(project, train_configs['name']), '..')

    model = getattr(model, _name_to_class(model_configs['name']))(model_configs)
    dataset = getattr(dataset, _name_to_class(dataset_configs['name']))(dataset_configs)
    loss = getattr(loss, _name_to_class(loss_configs['name']))(loss_configs)
    trainer = getattr(trainer, _name_to_class(train_configs['name']))(model, dataset, loss)

    try:
        trainer.train(train_configs, optimizer_configs)
    except KeyboardInterrupt:
        trainer.logger.info(
            'Got Keyboard Interuption, saving model and closing.')
        trainer.save(os.path.join(train_configs['log_dir'],'interrupt_ckpt.pt'))
