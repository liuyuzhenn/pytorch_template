import importlib
import torch
import glob
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import yaml
import os
from torch.utils.tensorboard.writer import SummaryWriter
from abc import ABCMeta

from ..losses.base_loss import NoGradientError
from ..utils import get_logger
from .utils import *


class BaseRunner(metaclass=ABCMeta):
    """Base runner from training/testing and logging
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
        if self.distributed:
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device(test_configs['device'])

        checkpoint = torch.load(
            test_configs['checkpoint'], map_location=test_configs['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        val_data = self.dataset.get_data_loader(test_configs['split'])

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

                metrics = self._metrics(model_outputs, data, mode='test')
                if metrics is not None:
                    items.update(metrics)

                avg_meter.update(tensor2float(items))
        metrics = avg_meter.mean()
        with open(test_configs['file_path'], 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)

    def init_weights(self):
        """Initialize model weight at the beggining of training"""

    def info(self, logger, message):
        if self.local_rank <= 0:
            logger.info(message)

    def __init__(self, model, dataset, loss):
        self.model = model
        self.dataset = dataset
        self.loss_term = loss
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.distributed = self.local_rank >= 0

    def train(self, train_configs, optimizer_configs):
        # device
        workspace = train_configs['workspace']
        self.logger = get_logger(workspace)
        if self.distributed:
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device(train_configs['device'])

        self.info(self.logger, "Parameter count: {}".format(
            sum(p.numel() for p in self.model.parameters())))

        #######################
        # setup data parallel #
        #######################
        self.model.to(self.device)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            world_size = dist.get_world_size()

        ####################
        # set up optimizer #
        ####################
        optim = importlib.import_module("torch.optim")
        name = optimizer_configs['name']
        params = optimizer_configs.copy()
        params.pop('name')
        scheduler_configs = params.pop('lr_scheduler', None)
        self.optimizer = getattr(optim, name)(
            self.model.parameters(), **params)

        ####################
        # set up scheduler #
        ####################
        if scheduler_configs is None:
            # seudo scheduler
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda epoch: 1)
        else:
            lib_schedulers = importlib.import_module(
                'torch.optim.lr_scheduler')
            name_scheduler = scheduler_configs['name']
            self.params = scheduler_configs.copy()
            self.params.pop('name')
            self.params = {k: (v if k != 'lr_lambda' else eval(v))
                           for k, v in self.params.items()}
            self.lr_scheduler = getattr(lib_schedulers, name_scheduler)(
                self.optimizer, **self.params)

        ################################
        # continue from the checkpoint #
        ################################
        self.epoch = 0
        self.loss_val_min = np.inf
        self.init_weights()

        resume = train_configs.get('resume', False)
        if resume:
            checkpoint = train_configs.get('checkpoint', None)
            if checkpoint is None:
                files = glob.glob(os.path.join(workspace, 'ckpt_*.pth'))
                files = [os.path.basename(f) for f in files]
                if len(files) > 0:
                    checkpoint = files[-1]
                else:
                    self.info(self.logger, "No checkpoint found!")

            if checkpoint is not None:
                if isinstance(checkpoint, int):
                    checkpoint = 'ckpt_{:0>4}.pth'.format(checkpoint)
                checkpoint = os.path.join(workspace, checkpoint)
                self.info(self.logger,
                          "Loading checkpoint from {}.".format(checkpoint))
                self.load(checkpoint)

        ##################
        # train/val data #
        ##################
        train_loader = self.dataset.get_data_loader('train')
        val_loader = self.dataset.get_data_loader('val')

        ##########################
        # initialize tensorboard #
        ##########################
        if train_configs.get('enable_tensorboard', True):
            writer = SummaryWriter(workspace)
        else:
            writer = None

        ckpt_best = 'best.pth'
        while self.epoch < train_configs['num_epochs']:
            # Train
            avg_meter = DictAverageMeter()
            if self.distributed:
                train_loader.sampler.set_epoch(self.epoch)
            self.model.train()
            for i, data in enumerate(train_loader):
                t1 = time.time()
                data = to_device(data, self.device)
                self.optimizer.zero_grad()
                model_outputs = self.model.forward(data)
                try:
                    loss = self.loss_term.compute(model_outputs, data)
                except NoGradientError:
                    self.info(self.logger, "[Train] [Epoch {}/{}] [Iteration {}/{}] {}"
                              .format(self.epoch+1, train_configs['num_epochs'], i+1, len(train_loader), 'No Gradient!'))
                    continue
                if isinstance(loss, tuple):
                    loss, items = loss
                    if self.distributed:
                        _ = [dist.all_reduce(x) for x in items.values()]
                        items = {k: v/world_size for k, v in items.items()}
                else:
                    items = None

                loss.backward()
                self.optimizer.step()
                t2 = time.time()


                global_step = len(train_loader)*self.epoch+i

                if self.distributed:
                    dist.all_reduce(loss)
                    loss /= world_size
                self.info(self.logger, "[Train] [Epoch {}/{}] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s"
                          .format(self.epoch+1, train_configs['num_epochs'], i+1, len(train_loader), float(loss), t2-t1))

                if items is not None:
                    items.update({'loss': float(loss)})
                else:
                    items = {'loss': float(loss)}

                metrics = self._metrics(model_outputs, data, mode='train')
                if metrics is not None:
                    if self.distributed:
                        _ = [dist.all_reduce(x) for x in metrics.values()]
                        metrics = {k: v/world_size for k, v in metrics.items()}
                    items.update(metrics)
                # save in average meter
                avg_meter.update(tensor2float(items))

                if (global_step+1) % train_configs['summary_freq'] == 0:
                    if self.local_rank <= 0:
                        images = self._get_images(model_outputs, data)
                        save_scalars(writer, 'train', items, global_step)
                        if images is not None:
                            save_images(writer, 'train', images, global_step)

            if writer is not None and avg_meter.count != 0 and self.local_rank <= 0:
                save_scalars(writer, 'train_avg',
                             avg_meter.mean(), self.epoch+1)

                self.info(self.logger, "[Train] [Epoch {}/{}] {}".format(self.epoch+1,
                                                                         train_configs['num_epochs'], dict_to_str(avg_meter.mean())))
            self.lr_scheduler.step()

            for g in self.optimizer.param_groups:
                self.info(self.logger, "Adjusting learning rate of group 0 to {}.".format(
                    g['lr']))

            if (self.epoch+1) % train_configs['checkpoint_interval'] == 0 and self.local_rank <= 0:
                self.save(workspace)

            # Validation
            if (self.epoch+1) % train_configs.get('val_interval', 1) == 0:
                avg_meter = DictAverageMeter()
                with torch.no_grad():
                    self.model.eval()
                    for i, data in enumerate(val_loader):
                        data = to_device(data, self.device)
                        model_outputs = self.model.forward(data)

                        t1 = time.time()
                        try:
                            loss = self.loss_term.compute(model_outputs, data)
                        except NoGradientError:
                            self.info(self.logger, "[Val] [Epoch {}/{}] [Iteration {}/{}] {}"
                                      .format(self.epoch+1, train_configs['num_epochs'], i+1, len(val_loader), 'No Gradient!'))
                            continue
                        t2 = time.time()

                        if isinstance(loss, tuple):
                            loss, items = loss
                            if self.distributed:
                                _ = [dist.all_reduce(x) for x in items.values()]
                                items = {k: v/world_size for k,
                                         v in items.items()}
                                dist.all_reduce(loss)
                                loss /= world_size
                        else:
                            items = None

                        self.info(self.logger, "[Val] [Epoch {}/{}] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s"
                                  .format(self.epoch+1, train_configs['num_epochs'], i+1, len(val_loader), float(loss), t2-t1))
                        global_step = len(val_loader)*self.epoch+i+1


                        if items is not None:
                            items.update({'loss': float(loss)})
                        else:
                            items = {'loss': float(loss)}

                        metrics = self._metrics(
                            model_outputs, data, mode='val')
                        if metrics is not None:
                            items.update(metrics)

                        avg_meter.update(tensor2float(items))

                    if avg_meter.count != 0:
                        meter_mean = avg_meter.mean()
                        self.info(self.logger, "[Val] [Epoch {}/{}] {}".format(self.epoch+1,
                                                                               train_configs['num_epochs'], dict_to_str(meter_mean)))
                        if writer is not None:
                            save_scalars(
                                writer, 'val', meter_mean, self.epoch+1)

                        loss_current = meter_mean['loss']
                        if loss_current < self.loss_val_min:
                            self.loss_val_min = loss_current
                            self.info(self.logger,
                                      "Update best ckeckpoint, saved as {}".format(ckpt_best))
                            self.save(workspace, ckpt_best)

            self.epoch += 1

    def save(self, out_dir, ckpt_name=None):
        ckpt_name = 'ckpt_{:0>4}.pth'.format(
            self.epoch+1) if ckpt_name is None else ckpt_name
        save_path = os.path.join(out_dir, ckpt_name)
        torch.save({
            'epoch': self.epoch+1,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss_val_min': self.loss_val_min
        }, save_path)

    def load(self, checkpoint_path, model_only=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if not model_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss_val_min = checkpoint['loss_val_min']

            for g in self.optimizer.param_groups:
                self.info(self.logger, "Adjusting learning rate of group 0 to {}.".format(
                    g['lr']))
