import torch
import glob
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import time
import yaml
import os
from torch.utils.tensorboard.writer import SummaryWriter
from omegaconf import OmegaConf
from abc import ABCMeta
from importlib import import_module

from ..losses.exceptions import NoGradientError
from ..utils import get_logger
from .utils import *


def _name_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))


class IterRunner(metaclass=ABCMeta):
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

    def test(self):
        """
        Test model after training.

        Returns:
            None
        """
        test_configs = self.configs.test

        if self.distributed:
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device(test_configs.device)

        workspace = test_configs.workspace
        checkpoint_path = test_configs.get('checkpoint', '')
        ckpt_dir = os.path.join(workspace, 'checkpoints')
        if checkpoint_path != '':
            if isinstance(checkpoint_path, int):
                checkpoint_path = 'ckpt_{:0>10}.pth'.format(checkpoint_path)
            checkpoint_path = os.path.join(ckpt_dir, checkpoint_path)
        else:
            checkpoint_path = os.path.join(ckpt_dir, 'best.pth')

        print(f'Load checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path,
                                map_location=test_configs.device)
        epoch = checkpoint.get('epoch', None)
        metrics = {}
        if epoch is not None:
            metrics['epoch'] = epoch
            print('Epoch: {}'.format(epoch))
        else:
            metrics['step'] = checkpoint['step']
            print('Step: {}'.format(checkpoint['step']))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        ####################
        # setup dataloader #
        ####################
        batch_size = self.dataset_configs.batch_size
        num_workers = self.dataset_configs.num_workers
        test_dataset = self.dataset(self.configs, 'test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=True,
                                 num_workers=num_workers)

        self.model.eval()
        # test
        avg_meter = DictAverageMeter()
        with torch.no_grad():
            self.model.eval()
            for data in test_loader:
                data = to_device(data, self.device)
                model_outputs = self.model(data, mode='test')

                try:
                    loss = self.loss_term(model_outputs, data, mode='test')
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

                m = self._metrics(model_outputs, data, mode='test')
                if m is not None:
                    items.update(m)

                avg_meter.update(tensor2float(items))
        metrics.update(avg_meter.mean())
        metrics['checkpoint'] = checkpoint_path
        with open(test_configs.file_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
        print(dict_to_str(metrics))
        print(f'Results saved to: {test_configs.file_path}')

    def init_weights(self):
        """Initialize model weight at the beggining of training"""

    def info(self, logger, message):
        if self.local_rank <= 0:
            logger.info(message)

    def __init__(self, configs):
        project = configs.get('project', 'src')
        self.configs = configs

        self.confgs = configs
        self.model_configs = configs.model
        self.dataset_configs = configs.dataset
        self.loss_configs = configs.loss

        model = import_module('.models.{}'.format(
            self.model_configs.name), project)
        dataset = import_module('.datasets.{}'.format(
            self.dataset_configs.name), project)
        loss = import_module('.losses.{}'.format(
            self.loss_configs.name), project)

        self.model = getattr(model, _name_to_class(
            self.model_configs.name))(configs)
        # class
        self.dataset = getattr(dataset, _name_to_class(
            self.dataset_configs.name))
        self.loss_term = getattr(loss, _name_to_class(
            self.loss_configs.name))(configs)

        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.distributed = self.local_rank >= 0

    def train(self):
        optimizer_configs = self.configs.optimizer
        train_configs = self.configs.train

        # device
        workspace = train_configs.workspace
        self.logger = get_logger(workspace)
        if self.distributed:
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device(train_configs.device)

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
        optim = import_module("torch.optim")
        name = optimizer_configs.name
        params = OmegaConf.to_container(optimizer_configs)
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
            lib_schedulers = import_module(
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
        self.step = 0
        self.metric_val_min = np.inf
        self.init_weights()
        resume = train_configs.get('resume', False)
        ckpt_dir = os.path.join(workspace, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        if resume:
            checkpoint = train_configs.get('checkpoint', None)
            if checkpoint is None:
                files = glob.glob(os.path.join(ckpt_dir, 'ckpt_*.pth'))
                files = [os.path.basename(f) for f in files]
                files.sort()
                if len(files) > 0:
                    checkpoint = files[-1]
                else:
                    self.info(self.logger, "No checkpoint found!")

            if checkpoint is not None:
                if isinstance(checkpoint, int):
                    checkpoint = 'ckpt_{:0>10}.pth'.format(checkpoint)
                checkpoint = os.path.join(ckpt_dir, checkpoint)
                self.info(self.logger,
                          "Loading checkpoint from {}.".format(checkpoint))
                self.load(checkpoint)

        ####################
        # setup dataloader #
        ####################
        batch_size = self.dataset_configs.batch_size
        num_workers = self.dataset_configs.num_workers
        train_dataset = self.dataset(self.configs, 'train')
        val_dataset = self.dataset(self.configs, 'val')
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size, sampler=val_sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=True,
                                      num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, drop_last=True,
                                    num_workers=num_workers)

        ##########################
        # initialize tensorboard #
        ##########################
        if train_configs.get('enable_tensorboard', True) and self.local_rank <= 0:
            writer = SummaryWriter(workspace)
        else:
            writer = None

        ckpt_best = 'best.pth'
        while self.step < train_configs.num_steps:
            # Train
            avg_meter = DictAverageMeter()
            if self.distributed:
                train_loader.sampler.set_epoch(self.step//len(train_loader))
            self.model.train()
            for i, data in enumerate(train_loader):
                t1 = time.time()
                data = to_device(data, self.device)
                self.optimizer.zero_grad()
                model_outputs = self.model.forward(data)
                try:
                    loss = self.loss_term(model_outputs, data, mode='train')
                except NoGradientError:
                    self.info(self.logger, "[Train] [Iteration {}/{}] {}"
                              .format(self.step+1, train_configs.num_steps, 'No Gradient!'))
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
                self.lr_scheduler.step()
                t2 = time.time()

                if self.distributed:
                    dist.all_reduce(loss)
                    loss /= world_size
                self.info(self.logger, "[Train] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s | LR: {:.8f}"
                          .format(self.step+1, train_configs.num_steps, float(loss), t2-t1, self.optimizer.param_groups[0]['lr']))

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

                if self.step % train_configs.summary_freq == 0 and writer is not None:
                    save_scalars(writer, 'train', items, self.step)
                    images = self._get_images(model_outputs, data)
                    if images is not None:
                        save_images(writer, 'train', images, self.step)

                if (self.step+1) % train_configs.checkpoint_interval == 0 and self.local_rank <= 0:
                    self.save(ckpt_dir)

                # Validation
                if (self.step+1) % train_configs.get('val_interval', 10000) == 0:
                    avg_meter_val = DictAverageMeter()
                    if self.distributed:
                        val_loader.sampler.set_epoch(self.step)
                    with torch.no_grad():
                        self.model.eval()
                        for i, data in enumerate(val_loader):
                            t1 = time.time()
                            data = to_device(data, self.device)
                            model_outputs = self.model(data, mode='val')
                            t2 = time.time()

                            try:
                                loss = self.loss_term(model_outputs, data, mode='val')
                            except NoGradientError:
                                self.info(self.logger, "[Val] [Iteration {}/{}] {}"
                                          .format(i+1, len(val_loader), 'No Gradient!'))
                                continue

                            if isinstance(loss, tuple):
                                loss, items = loss
                                if self.distributed:
                                    _ = [dist.all_reduce(x)
                                         for x in items.values()]
                                    items = {k: v/world_size for k,
                                             v in items.items()}
                                    dist.all_reduce(loss)
                                    loss /= world_size
                            else:
                                items = None

                            self.info(self.logger, "[Val] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s"
                                      .format(i+1, len(val_loader), float(loss), t2-t1))

                            if items is not None:
                                items.update({'loss': float(loss)})
                            else:
                                items = {'loss': float(loss)}

                            metrics = self._metrics(
                                model_outputs, data, mode='val')
                            if metrics is not None:
                                items.update(metrics)

                            avg_meter_val.update(tensor2float(items))

                        if avg_meter_val.count != 0:
                            meter_mean = avg_meter_val.mean()
                            self.info(self.logger, "[Val] [Iteration {}/{}] {}".format(self.step+1,
                                                                                   train_configs.num_steps, dict_to_str(meter_mean)))
                            if writer is not None:
                                save_scalars(
                                    writer, 'val', meter_mean, self.step+1)

                            metric_current = meter_mean[train_configs.get('metric_val', 'loss')]
                            if metric_current < self.metric_val_min:
                                self.metric_val_min = metric_current
                                self.info(self.logger,
                                          "Update best ckeckpoint, saved as {}".format(ckpt_best))
                                self.save(ckpt_dir, ckpt_best)

                self.step += 1
                if self.step==train_configs.num_steps:
                    break

            if writer is not None and avg_meter.count != 0:
                save_scalars(writer, 'train_avg',
                             avg_meter.mean(), self.step//len(train_loader))

            if avg_meter.count != 0:
                self.info(self.logger, "[Train Average] [Epoch {}] {}".format(self.step//len(train_loader),
                                                                         dict_to_str(avg_meter.mean())))


    def save(self, out_dir, ckpt_name=None):
        ckpt_name = 'ckpt_{:0>10}.pth'.format(
            self.step+1) if ckpt_name is None else ckpt_name
        save_path = os.path.join(out_dir, ckpt_name)
        torch.save({
            'step': self.step+1,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'metric_val_min': self.metric_val_min
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
            self.step = checkpoint['step']
            self.metric_val_min = checkpoint['metric_val_min']

            for g in self.optimizer.param_groups:
                self.info(self.logger, "Adjusting learning rate of group 0 to {}.".format(
                    g['lr']))
