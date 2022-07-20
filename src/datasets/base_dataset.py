from abc import ABCMeta,abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader

class BaseDataset(metaclass=ABCMeta):
    @abstractmethod
    def get_dataset(self, split):
        """To be implemented by the child class."""
        raise NotImplementedError

    def __init__(self,configs):
        self.configs = configs
    
    def get_data_loader(self, split):
        """Return a data loader for a given split."""
        assert split in ['train', 'val', 'test']
        batch_size = self.configs['batch_size']
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=(split=='train'),drop_last=True)