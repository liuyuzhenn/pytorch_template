import torch
import glob
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import time
from tqdm import tqdm
import yaml
import os
from torch.utils.tensorboard.writer import SummaryWriter
from omegaconf import OmegaConf
from abc import ABCMeta

from ..losses.exceptions import NoGradientError
from src.utils import get_logger, build_instance, get_cls
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

    def on_train_epoch_start(self):
        return

    def training_step(self, batch, batch_idx):
        model_outputs = self.model.forward(batch)
        try:
            loss = self.loss_term(model_outputs, batch, mode='train')
        except NoGradientError:
            raise NoGradientError
        return loss, model_outputs

    def on_train_epoch_end(self):
        return

    def on_validation_epoch_start(self):
        return

    def on_validation_batch_start(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        '''
        could use it to store validaiton values,
        then use the values to compute metrics, 
        and finally save the overall metrics through `on_validation_end`
        '''
        model_outputs = self.model(batch, mode='val')
        try:
            loss = self.loss_term(model_outputs, batch, mode='val')
        except NoGradientError:
            raise NoGradientError
        metrics = {'loss': float(loss)}
        metrics.update(self._metrics(model_outputs, batch, mode='val'))
        return model_outputs, metrics

    def on_validation_batch_end(self, model_outputs, batch, batch_idx):
        return

    def on_validation_end(self):
        return

    ##################
    # test callbacks #
    ##################
    def on_test_epoch_start(self):
        return

    def on_test_batch_start(self, batch, batch_idx):
        return

    def test_step(self, batch, batch_idx):
        '''
        could use it to store validaiton values,
        then use the values to compute metrics, 
        and finally save the overall metrics through `on_validation_end`
        '''
        return self.validation_step(batch, batch_idx)

    def on_test_batch_end(self, model_outputs, batch, batch_idx):
        return

    def on_test_end(self):
        return

    #####################
    # logging functions #
    #####################
    def print(self, message):
        if self.local_rank <= 0:
            self.logger.info(message)

    def log(self, mode, scalar_dict, step):
        if self.writer is not None:
            save_scalars(self.writer, mode, scalar_dict, step)

    def log_imgs(self, mode, img_dict, step):
        if self.writer is not None:
            save_images(self.writer, mode, img_dict, step)

    def __init__(self, configs):
        self.configs = configs

        self.confgs = configs
        self.model_configs = configs.model
        self.dataset_configs = configs.dataset
        self.loss_configs = configs.loss


        self.model = build_instance(self.model_configs.__target__, configs)
        self.loss_term = build_instance(self.loss_configs.__target__, configs)

        self.dataset = get_cls(self.dataset_configs.__target__)

        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.distributed = self.local_rank >= 0
        self.writer = None

    def train(self):
        raise NotImplementedError


    def save(self, out_dir, ckpt_name=None):
        raise NotImplementedError

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
            self.global_step = checkpoint['step']
            self.metric_val_min = checkpoint['metric_val_min']

            for g in self.optimizer.param_groups:
                self.print("Adjusting learning rate of group 0 to {}.".format(
                    g['lr']))


class IterRunner(BaseRunner):
    """Base runner from training/testing and logging
    """
    def __init__(self, configs):
        super().__init__(configs)


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
            self.on_test_epoch_start()
            for idx, data in tqdm(enumerate(test_loader)):
                data = to_device(data, self.device)
                
                self.on_test_batch_start(data, idx)
                try:
                    model_outputs, m = self.test_step(data, idx)
                except NoGradientError:
                    continue
                self.on_test_batch_end(model_outputs, data, idx)

                avg_meter.update(tensor2float(m))

        metrics.update(avg_meter.mean())
        metrics['checkpoint'] = checkpoint_path
        with open(test_configs.file_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
        print(dict_to_str(metrics))
        print(f'Results saved to: {test_configs.file_path}')


    def train(self):
        optimizer_configs = self.configs.optimizer
        train_configs = self.configs.train

        val_interval = train_configs.get('val_interval', 10000)
        val_batches = train_configs.get('val_batches', np.inf)
        val_mode  = train_configs.get('val_mode', 'min')
        metric_val = train_configs.get('val_metric', 'loss')
        max_steps = train_configs.max_steps

        # device
        workspace = train_configs.workspace
        self.logger = get_logger(workspace)
        if self.distributed:
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device(train_configs.device)

        self.print("Parameter count: {}".format(
            sum(p.numel() for p in self.model.parameters())))

        #######################
        # setup data parallel #
        #######################
        self.model.to(self.device)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        ####################
        # set up optimizer #
        ####################
        optim_cls = get_cls(optimizer_configs.__target__)
        params: dict = OmegaConf.to_container(optimizer_configs)
        params.pop('__target__')
        # scheduler_configs = params.pop('lr_scheduler', None)
        self.optimizer = optim_cls(self.model.parameters(), **params)

        ####################
        # set up scheduler #
        ####################
        scheduler_configs: dict = OmegaConf.to_container(self.configs.lr_scheduler)
        scheduler_cls = get_cls(scheduler_configs.pop('__target__'))
        params = {k: (v if k != 'lr_lambda' else eval(v))
                       for k, v in scheduler_configs.items()}
        self.lr_scheduler = scheduler_cls(self.optimizer, **params)

        ################################
        # continue from the checkpoint #
        ################################
        self.global_step = 0
        if val_mode=='min':
            self.metric_val_record = np.inf
        else:
            self.metric_val_record = -np.inf

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
                    self.print("No checkpoint found!")

            if checkpoint is not None:
                if isinstance(checkpoint, int):
                    checkpoint = 'ckpt_{:0>10}.pth'.format(checkpoint)
                checkpoint = os.path.join(ckpt_dir, checkpoint)
                self.print("Loading checkpoint from {}.".format(checkpoint))
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
            self.train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size, sampler=train_sampler)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=True,
                                      num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=True,
                                num_workers=num_workers)

        ##########################
        # initialize tensorboard #
        ##########################
        if train_configs.get('enable_tensorboard', True) and self.local_rank <= 0:
            self.writer = SummaryWriter(workspace)
        else:
            self.writer = None

        ckpt_best = 'best.pth'
        while self.global_step < max_steps:
            # Train
            avg_meter_train = DictAverageMeter()
            if self.distributed:
                self.train_loader.sampler.set_epoch(self.global_step//len(self.train_loader))
            train_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            self.on_train_epoch_start()
            for batch_idx, batch_data in train_bar:
                batch_data = to_device(batch_data, self.device)
                self.optimizer.zero_grad()

                try:
                    loss, model_outputs  = self.training_step(batch_data, batch_idx)
                except NoGradientError:
                    continue

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if self.distributed:
                    dist.all_reduce(loss)
                    loss /= self.world_size
                train_bar.set_description(f'[Train] [Iteration {self.global_step+1}/{max_steps}] Loss: {float(loss.mean()):.4f}')


                metrics = self._metrics(model_outputs, batch_data, mode='train')
                metrics.update({'loss': loss})
                avg_meter_train.update(tensor2float(metrics))

                if self.global_step % train_configs.summary_freq == 0:
                    self.log('train', metrics, self.global_step)
                    images = self._get_images(model_outputs, batch_data)
                    if images is not None:
                        self.log_imgs('train', images, self.global_step)

                if (self.global_step+1) % train_configs.checkpoint_interval == 0 and self.local_rank <= 0:
                    self.save(ckpt_dir)

                # Validation
                if (self.global_step+1) % val_interval == 0:
                    avg_meter_val = DictAverageMeter()
                    self.model.eval()
                    val_bar = tqdm(enumerate(self.val_loader), total=min(val_batches, len(self.val_loader)))
                    self.on_validation_epoch_start()
                    for batch_idx, batch_data in val_bar:
                        batch_data = to_device(batch_data, self.device)

                        self.on_validation_batch_start(batch_data, batch_idx)
                        batch_data = to_device(batch_data, self.device)
                        try:
                            with torch.no_grad():
                                model_outputs, metrics = self.validation_step(batch_data, batch_idx)
                        except NoGradientError:
                            self.print("[Val] Value errors encountered!")
                            continue

                        self.on_validation_batch_end(model_outputs, batch_data, batch_idx)

                        avg_meter_val.update(tensor2float(metrics))

                        if batch_idx == val_batches:
                            break
                    self.model.train()

                    self.on_validation_end()

                    if avg_meter_val.count != 0:
                        p = os.path.join(ckpt_dir, ckpt_best)
                        meter_mean = avg_meter_val.mean()
                        self.print("[Val] [Iteration {}/{}] {}".format(self.global_step+1, max_steps, dict_to_str(meter_mean)))
                        self.log('val', meter_mean, self.global_step)

                        metric_current = meter_mean[metric_val]
                        if val_mode=='min' and (metric_current < self.metric_val_record):
                            self.metric_val_record = metric_current
                            self.save(ckpt_dir, ckpt_best)
                            self.print("Update best ckeckpoint, saved to {}".format(p))
                        elif val_mode=='max' and (metric_current > self.metric_val_record):
                            self.metric_val_record = metric_current
                            self.save(ckpt_dir, ckpt_best)
                            self.print("Update best ckeckpoint, saved to {}".format(p))
                        self.print(f"[Val] Best {metric_val}: {self.metric_val_record:.4f}")

                self.global_step += 1
                if self.global_step==max_steps:
                    break

            if avg_meter_train.count != 0:
                meter_mean_train = avg_meter_train.mean()
                self.log('train_avg', meter_mean_train, self.global_step//len(self.train_loader))
                self.print("[Train] [Iteration {}/{}] {}".format(self.global_step, max_steps, dict_to_str(meter_mean_train)))

            for idx,g in enumerate(self.optimizer.param_groups):
                self.print("Adjusting learning rate of group {} to {}.".format(idx, g['lr']))


    def save(self, out_dir, ckpt_name=None):
        ckpt_name = 'ckpt_{:0>10}.pth'.format(
            self.global_step+1) if ckpt_name is None else ckpt_name
        save_path = os.path.join(out_dir, ckpt_name)
        torch.save({
            'step': self.global_step+1,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'metric_val_record': self.metric_val_record
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
            self.global_step = checkpoint['step']
            self.metric_val_record = checkpoint['metric_val_record']

            for g in self.optimizer.param_groups:
                self.print("Adjusting learning rate of group 0 to {}.".format(
                    g['lr']))


class EpochRunner(BaseRunner):
    """Base runner from training/testing and logging
    """
    def __init__(self, configs):
        super().__init__(configs)


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
                checkpoint_path = 'ckpt_{:0>4}.pth'.format(checkpoint_path)
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
            self.on_test_epoch_start()
            for idx, data in tqdm(enumerate(test_loader)):
                data = to_device(data, self.device)
                
                self.on_test_batch_start(data, idx)
                try:
                    model_outputs, m = self.test_step(data, idx)
                except NoGradientError:
                    continue
                self.on_test_batch_end(model_outputs, data, idx)

                avg_meter.update(tensor2float(m))

        self.on_test_end()
        metrics.update(avg_meter.mean())
        metrics['checkpoint'] = checkpoint_path
        with open(test_configs.file_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
        print(dict_to_str(metrics))
        print(f'Results saved to: {test_configs.file_path}')


    def train(self):
        optimizer_configs = self.configs.optimizer
        train_configs = self.configs.train

        val_interval = train_configs.get('val_interval', 1)
        val_batches = train_configs.get('val_batches', np.inf)
        val_mode  = train_configs.get('val_mode', 'min')
        metric_val = train_configs.get('val_metric', 'loss')
        max_epochs = train_configs.max_epochs

        # device
        workspace = train_configs.workspace
        self.logger = get_logger(workspace)
        if self.distributed:
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device(train_configs.device)

        self.print("Parameter count: {}".format(
            sum(p.numel() for p in self.model.parameters())))

        #######################
        # setup data parallel #
        #######################
        self.model.to(self.device)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        ####################
        # set up optimizer #
        ####################
        optim_cls = get_cls(optimizer_configs.__target__)
        params: dict = OmegaConf.to_container(optimizer_configs)
        params.pop('__target__')
        scheduler_configs = params.pop('lr_scheduler', None)
        self.optimizer = optim_cls(self.model.parameters(), **params)

        ####################
        # set up scheduler #
        ####################
        scheduler_configs: dict = OmegaConf.to_container(self.configs.lr_scheduler)
        scheduler_cls = get_cls(scheduler_configs.pop('__target__'))
        params = {k: (v if k != 'lr_lambda' else eval(v))
                       for k, v in scheduler_configs.items()}
        self.lr_scheduler = scheduler_cls(self.optimizer, **params)

        ################################
        # continue from the checkpoint #
        ################################
        self.epoch = 0
        if val_mode=='min':
            self.metric_val_record = np.inf
        else:
            self.metric_val_record = -np.inf
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
                    self.print("No checkpoint found!")

            if checkpoint is not None:
                if isinstance(checkpoint, int):
                    checkpoint = 'ckpt_{:0>4}.pth'.format(checkpoint)
                checkpoint = os.path.join(ckpt_dir, checkpoint)
                self.print("Loading checkpoint from {}.".format(checkpoint))
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
            self.train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size, sampler=train_sampler)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=True,
                                      num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=True,
                                num_workers=num_workers)

        ##########################
        # initialize tensorboard #
        ##########################
        if train_configs.get('enable_tensorboard', True) and self.local_rank <= 0:
            self.writer = SummaryWriter(workspace)
        else:
            self.writer = None

        ckpt_best = 'best.pth'
        while self.epoch < max_epochs:
            # Train
            avg_meter = DictAverageMeter()
            if self.distributed:
                self.train_loader.sampler.set_epoch(self.epoch)
            self.model.train()
            bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            self.on_train_epoch_start()
            for batch_idx, batch_data in bar:
                batch_data = to_device(batch_data, self.device)
                self.optimizer.zero_grad()

                try:
                    loss, model_outputs  = self.training_step(batch_data, batch_idx)
                except NoGradientError:
                    continue

                loss.backward()
                self.optimizer.step()

                self.global_step = len(self.train_loader)*self.epoch+batch_idx

                if self.distributed:
                    dist.all_reduce(loss)
                    loss /= self.world_size
                # bar.set_description(f'[Train] [Epoch {self.epoch+1}/{max_epochs}] Loss: {float(loss.mean()):.6f}')
                bar.set_description(f'[Train] [Epoch {self.epoch+1}/{max_epochs}] Loss: {float(loss.mean()):.4f}')

                metrics = self._metrics(model_outputs, batch_data, mode='train')
                metrics.update({'loss': loss})
                # if self.distributed and (metrics is not None):
                #     _ = [dist.all_reduce(x) for x in metrics.values()]
                #     metrics = {k: v/self.world_size for k, v in metrics.items()}
                # save in average meter
                avg_meter.update(tensor2float(metrics))

                if self.global_step % train_configs.summary_freq == 0 and self.local_rank <= 0:
                    self.log('train', metrics, self.global_step)
                    images = self._get_images(model_outputs, batch_data)
                    if images is not None:
                        self.log_imgs('train', images, self.global_step)
            self.on_train_epoch_end()

            if avg_meter.count != 0:
                meter_mean = avg_meter.mean()
                self.log('train_avg', meter_mean, self.epoch)
                bar.set_description("[Train] [Epoch {}/{}] {}".format(self.epoch+1, max_epochs, dict_to_str(meter_mean)))
                self.print("[Train] [Epoch {}/{}] {}".format(self.epoch+1, max_epochs, dict_to_str(meter_mean)))

            self.lr_scheduler.step()

            for g in self.optimizer.param_groups:
                self.print("Adjusting learning rate of group 0 to {}.".format(
                    g['lr']))

            if (self.epoch+1) % train_configs.checkpoint_interval == 0 and self.local_rank <= 0:
                self.save(ckpt_dir)

            # Validation
            if (self.epoch+1) % val_interval == 0:
                self.on_validation_epoch_start()
                avg_meter = DictAverageMeter()
                self.model.eval()
                bar = tqdm(enumerate(self.val_loader), total=min(val_batches, len(self.val_loader)))
                for batch_idx, batch_data in bar:
                    self.on_validation_batch_start(batch_data, batch_idx)

                    batch_data = to_device(batch_data, self.device)
                    try:
                        with torch.no_grad():
                            model_outputs, metrics = self.validation_step(batch_data, batch_idx)
                    except NoGradientError:
                        self.print("[Val] [Epoch {}/{}] [Iteration {}/{}] Value errors encountered!"
                                  .format(self.epoch+1, max_epochs, batch_idx+1, len(self.val_loader)))
                        continue

                    self.on_validation_batch_end(model_outputs, batch_data, batch_idx)

                    avg_meter.update(tensor2float(metrics))
                    if batch_idx == val_batches:
                        break
                self.on_validation_end()

            if avg_meter.count != 0:
                meter_mean = avg_meter.mean()
                p = os.path.join(ckpt_dir, ckpt_best)
                self.print("[Val] [Epoch {}/{}] {}".format(self.epoch+1, max_epochs, dict_to_str(meter_mean)))
                self.log('val', meter_mean, self.epoch)

                metric_current = meter_mean[metric_val]
                if val_mode=='min' and (metric_current < self.metric_val_record):
                    self.metric_val_record = metric_current
                    self.save(ckpt_dir, ckpt_best)
                    self.print("Update best ckeckpoint, saved as {}".format(p))
                elif val_mode=='max' and (metric_current > self.metric_val_record):
                    self.metric_val_record = metric_current
                    self.save(ckpt_dir, ckpt_best)
                    self.print("Update best ckeckpoint, saved as {}".format(p))
                self.print(f"[Val] Best {metric_val}: {self.metric_val_record:.4f}")
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
            'metric_val_record': self.metric_val_record
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
            self.metric_val_record = checkpoint['metric_val_min']

            for g in self.optimizer.param_groups:
                self.print("Adjusting learning rate of group 0 to {}.".format(g['lr']))
