import torch
import torch.optim as optim
import torch.nn as nn
import logging
import time
from abc import ABCMeta, abstractmethod
import os
from torch.utils.tensorboard import SummaryWriter
from .utils import *


class BaseModel(metaclass=ABCMeta):
    """Base model from training/testing and logging
    """
    @abstractmethod
    def _model(self):
        """ Implements the model.
        Arguments
        ---
            config: A configuration dictionary.
        Returns
        ---
            A torch.nn.Module that implements the model.
        """
        pass


    @abstractmethod
    def _forward(self, inputs, mode):
        """ Calls the model on some input.
        This method is called three times: for training, testing and
        prediction (see the `mode` argument) and can return different tensors
        depending on the mode.
        Arguments
        ---
            inputs: A dictionary of input features, where the keys are their
                names (e.g. `"image"`) and the values of type `torch.Tensor`.
            mode: An attribute of the `Mode` class, either `Mode.TRAIN`,
                  `Mode.TEST`
            config: A configuration dictionary.
        Returns
        ---
            A dictionary of outputs, where the keys are their names
            (e.g. `"logits"`) and the values are the corresponding Tensor.
        """
        pass

    # @abstractmethod
    # def _loss(self):
    #     """"""
        # """ Implements the training loss.
        # This method is called on the outputs of the `_model` method
        # in training mode.
        # Arguments
        # ---
        #     outputs: A dictionary, as returned by `_model` called with
        #              `mode=Mode.TRAIN`.
        #     inputs: A dictionary of input features (same as for `_model`).
        #     config: A configuration dictionary.
        # Returns
        # ---
        #     loss: A Tensor corresponding to the loss to minimize during training.
        #     dict: A dict containing data that will be saved.
        # """

    def _metrics(self, outputs, inputs):
        """Compute metrics that is saved in tensorboard.

        Args:
            outputs (dict): returned by `_model`
            inputs (dict): returned by dataset
        """

    def _get_images(self, outputs, inputs):
        """Visualizing results

        Args:
            outputs (dict): returned by `_model`
            inputs (dict): returned by dataset
        """

    def _init_weights(self):
        pass

    def __init__(self, dataset, configs, logger_name='default'):
        self.logger = logging.getLogger(logger_name)
        self.dataset = dataset
        self.configs = configs
        self.net = self._model()

    def train(self, train_args, loss_term):
        self._init_weights()
        # device
        self.device = train_args['device']
        if train_args['data_parallel']:
            self.net = nn.DataParallel(self.net)
        self.net.to(self.device)

        # train/val data
        train_data = self.dataset.get_data_loader('train')
        val_data = self.dataset.get_data_loader('test')

        # set optimzer/scheduler 
        wd = train_args['wd']
        lr = train_args['lr']
        self.optimizer = optim.Adam(self.net.parameters(),lr=lr,weight_decay=wd)
        if 'lrepochs' in train_args.keys():
            do_schedule = True
            milestones = [int(epoch_idx) for epoch_idx in train_args['lrepochs'].split(':')[0].split(',')]
            lr_gamma = float(train_args['lrepochs'].split(':')[1])
            lr = train_args['lr']
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones,
                                                        gamma=lr_gamma,
                                                        last_epoch=-1)
        else:
            do_schedule = False
            self.lr_scheduler = None

        # continue from the checkpoint
        resume_training = train_args.get('resume_training','')
        if resume_training == '':
            self.logger.info('Initializing new weights...')
            self.epoch = 0
        else:
            self.logger.info(
                'Loading weights from {}...'.format(resume_training))
            self.load(resume_training)
        
        if train_args.get('enable_tensorboard',True):
            writer = SummaryWriter(train_args['log_dir'])
        else:
            writer = None

        while self.epoch < train_args['num_epochs']:
            # Train
            self.net.train()
            for i, data in enumerate(train_data):
                t1 = time.time()
                data = to_device(data,self.device)
                self.optimizer.zero_grad()
                model_outputs = self._forward(data, mode='train')
                # loss, items = loss_term.compute(model_outputs, data)
                loss = loss_term.compute(model_outputs, data)
                if isinstance(loss,tuple):
                    loss, items = loss
                else:
                    items = None
                loss.backward()
                self.optimizer.step()
                t2 = time.time()

                global_step = len(train_data)*self.epoch+i

                self.logger.info('[Train] [Epoch {}/{}] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s'
                                 .format(self.epoch+1, train_args['num_epochs'], i+1, len(train_data), float(loss), t2-t1))

                if items is not None:
                    items.update({'loss':float(loss)})
                else:
                    items = {'loss':float(loss)}

                metrics = self._metrics(model_outputs, data)
                if metrics is not None:
                    items.update(metrics)
                # save in average meter
                avg_meter.update(tensor2float(items))

                if global_step % train_args['summary_freq'] == 0:
                    images = self._get_images(model_outputs, data)
                    save_scalars(writer, 'train', items, global_step)
                    if images is not None:
                        save_images(writer, 'train', images, global_step)

            if writer is not None and avg_meter.count != 0:
                save_scalars(writer, 'train_avg', avg_meter.mean(), self.epoch)
            self.save(train_args['log_dir'])

            # Validation
            avg_meter = DictAverageMeter()
            with torch.no_grad():
                self.net.eval()
                for i, data in enumerate(val_data):
                    data = to_device(data, self.device)
                    model_outputs = self._forward(data, mode='val')
                    t1 = time.time()
                    # loss, items = loss_term.compute(model_outputs, data, mode='val')
                    loss = loss_term.compute(model_outputs, data)
                    if isinstance(loss,tuple):
                        loss, items = loss
                    else:
                        items = None
                    t2 = time.time()
                    self.logger.info('[Val] [Epoch {}/{}] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s'
                                     .format(self.epoch+1, train_args['num_epochs'], i+1, len(val_data), float(loss), t2-t1))
                    global_step = len(val_data)*self.epoch+i
                    if items is not None:
                        items.update({'loss':float(loss)})
                    else:
                        items = {'loss':float(loss)}

                    metrics = self._metrics(model_outputs, data)
                    if metrics is not None:
                        items.update(metrics)

                    avg_meter.update(tensor2float(items))
                    # save_scalars(writer, 'val', items, global_step)
                    
                if writer is not None and avg_meter.count != 0:
                    save_scalars(writer, 'val', avg_meter.mean(), self.epoch)
                    self.logger.info('[Val] [Epoch {}/{}] {}'.format(self.epoch,
                                train_args['num_epochs'], dict_to_str(avg_meter.mean())))

            self.epoch += 1
            if do_schedule:
                self.lr_scheduler.step()

    def save(self, out_dir, ckpt_name=None):
        save_path = os.path.join(out_dir, 'ckpt{:0>4}'.format(self.epoch))
        if self.lr_scheduler is not None:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.net.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
            }, save_path)
        else:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.net.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        if 'scheduler_state_dict' in checkpoint.keys():
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
