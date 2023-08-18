import os
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch


class BaseDataset(metaclass=ABCMeta):
    @abstractmethod
    def get_dataset(self, split):
        """To be implemented by the child class

        Args:
            split: "train"|"val"|"test".

        Returns:
            A child class inherited from torch Dataset.
        """

    def __init__(self, configs):
        self.configs = configs
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.distributed = self.local_rank >= 0

    def get_data_loader(self, split):
        """Return a data loader with a given split."""
        assert split in ['train', 'val', 'test']
        batch_size = self.configs['batch_size']
        ds = self.get_dataset(split)
        if self.distributed: 
            train_sampler = DistributedSampler(ds, shuffle=True)
            return DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
        else:
            return DataLoader(ds, batch_size=batch_size,
                              shuffle=(split == 'train'), drop_last=True,
                              num_workers=self.configs['num_workers'])
