import importlib
from numpy import copy
import torch
# import torch.optim as optim
import torch.nn as nn
import logging
import time
from abc import ABCMeta, abstractmethod
import yaml
import os
from torch.utils.tensorboard.writer import SummaryWriter

from torch_temp.losses.base_loss import NoGradientError
from torch_temp.utils import get_logger
from .utils import *


class BaseTrainer(metaclass=ABCMeta):
    """Base trainer from training/testing and logging
    """

    def _metrics(self, outputs_model, inputs_data, mode='train') -> dict:
        """Compute metrics that is saved in tensorboard.

        Args:
            outputs_model (dict): returned by `self.model`
            inputs_data (dict): returned by dataset

        Returns:
            A dict containing different metrics.
        """

    def _get_images(self, outputs_model, inputs_data, mode='train'):
        """Visualizing results

        Args:
            outputs_model (dict): model output
            inputs_data (dict): returned by dataset

        Returns:
            None
        """

    def test(self, test_configs):
        """
        Test model after training.

        Args:
            test_configs: arguments for testing

        Returns:
            None
        """
        self.device = test_configs['device']
        checkpoint = torch.load(test_configs['checkpoint'], map_location=test_configs['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        val_data = self.dataset.get_data_loader('val')

        self.model.eval()
        # Validation
        avg_meter = DictAverageMeter()
        with torch.no_grad():
            self.model.eval()
            for data in val_data:
                data = to_device(data, self.device)
                model_outputs = self.model.forward(data)

                try:
                    loss = self.loss_term.compute(model_outputs, data)
                except NoGradientError:
                    continue

                if isinstance(loss, tuple):
                    loss, items = loss
                else:
                    items = None

                if items is not None:
                    items.update({'loss': float(loss)})
                else:
                    items = {'loss': float(loss)}

                metrics = self._metrics(model_outputs, data)
                if metrics is not None:
                    items.update(metrics)

                avg_meter.update(tensor2float(items))
        metrics = avg_meter.mean()
        with open(test_configs['file_path'], 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)


    def _init_weights(self):
        """Initialize model weight at the beggining of training"""

    def __init__(self, model, dataset, loss):
        self.model = model
        self.dataset = dataset
        self.loss_term = loss

    def train(self, train_configs, optimizer_configs):
        # device
        self.logger = get_logger(train_configs['log_dir'])
        self.device = train_configs['device']
        if train_configs['data_parallel']:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        ##################
        # train/val data #
        ##################
        train_data = self.dataset.get_data_loader('train')
        val_data = self.dataset.get_data_loader('val')

        ####################
        # set up optimizer #
        ####################
        optim = importlib.import_module("torch.optim")
        name = optimizer_configs['name']
        params = optimizer_configs.copy()
        params.pop('name')
        self.optimizer = getattr(optim, name)(
            self.model.parameters(), **params)

        ################################
        # continue from the checkpoint #
        ################################
        checkpoint = train_configs.get('checkpoint', '')
        if checkpoint == '':
            self.logger.info('Initializing new weights...')
            self._init_weights()
            self.epoch = 0
        else:
            self.logger.info(
                'Loading weights from {}...'.format(checkpoint))
            self.load(checkpoint)

        ##########################
        # initialize tensorboard #
        ##########################
        if train_configs.get('enable_tensorboard', True):
            writer = SummaryWriter(train_configs['log_dir'])
        else:
            writer = None

        while self.epoch < train_configs['num_epochs']:
            # Train
            avg_meter = DictAverageMeter()
            self.model.train()
            for i, data in enumerate(train_data):
                t1 = time.time()
                data = to_device(data, self.device)
                self.optimizer.zero_grad()
                model_outputs = self.model.forward(data)
                try:
                    loss = self.loss_term.compute(model_outputs, data)
                except NoGradientError:
                    self.logger.info('[Train] [Epoch {}/{}] [Iteration {}/{}] {}'
                                     .format(self.epoch+1, train_configs['num_epochs'], i+1, len(train_data), 'No Gradient!'))
                    continue
                if isinstance(loss, tuple):
                    loss, items = loss
                else:
                    items = None
                loss.backward()
                self.optimizer.step()
                t2 = time.time()

                global_step = len(train_data)*self.epoch+i

                self.logger.info('[Train] [Epoch {}/{}] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s'
                                 .format(self.epoch+1, train_configs['num_epochs'], i+1, len(train_data), float(loss), t2-t1))

                if items is not None:
                    items.update({'loss': float(loss)})
                else:
                    items = {'loss': float(loss)}

                metrics = self._metrics(model_outputs, data)
                if metrics is not None:
                    items.update(metrics)
                # save in average meter
                avg_meter.update(tensor2float(items))

                if global_step % train_configs['summary_freq'] == 0:
                    images = self._get_images(model_outputs, data)
                    save_scalars(writer, 'train', items, global_step)
                    if images is not None:
                        save_images(writer, 'train', images, global_step)

            if writer is not None and avg_meter.count != 0:
                save_scalars(writer, 'train_avg', avg_meter.mean(), self.epoch)
            self.save(train_configs['log_dir'])

            # Validation
            avg_meter = DictAverageMeter()
            with torch.no_grad():
                self.model.eval()
                for i, data in enumerate(val_data):
                    data = to_device(data, self.device)
                    model_outputs = self.model.forward(data)

                    t1 = time.time()
                    try:
                        loss = self.loss_term.compute(model_outputs, data)
                    except NoGradientError:
                        self.logger.info('[Val] [Epoch {}/{}] [Iteration {}/{}] {}'
                                         .format(self.epoch+1, train_configs['num_epochs'], i+1, len(val_data), 'No Gradient!'))
                        continue
                    t2 = time.time()
                    if isinstance(loss, tuple):
                        loss, items = loss
                    else:
                        items = None

                    self.logger.info('[Val] [Epoch {}/{}] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s'
                                     .format(self.epoch+1, train_configs['num_epochs'], i+1, len(val_data), float(loss), t2-t1))
                    global_step = len(val_data)*self.epoch+i
                    if items is not None:
                        items.update({'loss': float(loss)})
                    else:
                        items = {'loss': float(loss)}

                    metrics = self._metrics(model_outputs, data)
                    if metrics is not None:
                        items.update(metrics)

                    avg_meter.update(tensor2float(items))

                if writer is not None and avg_meter.count != 0:
                    save_scalars(writer, 'val', avg_meter.mean(), self.epoch)
                    self.logger.info('[Val] [Epoch {}/{}] {}'.format(self.epoch+1,
                                                                     train_configs['num_epochs'], dict_to_str(avg_meter.mean())))

            self.epoch += 1

    def save(self, out_dir, ckpt_name=None):
        ckpt_name = 'ckpt_{:0>4}.pth'.format(self.epoch+1) if ckpt_name is None else ckpt_name
        save_path = os.path.join(out_dir, ckpt_name)
        torch.save({
            'epoch': self.epoch+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
