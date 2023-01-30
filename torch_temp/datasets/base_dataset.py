from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader


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

    def get_data_loader(self, split):
        """Return a data loader with a given split."""
        assert split in ['train', 'val', 'test']
        batch_size = self.configs['batch_size']
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=(split == 'train'), drop_last=True)
