from .base_dataset import BaseDataset
from torch.utils.data import Dataset
import torch


class MyDataset(BaseDataset):
    def __init__(self, configs):
        super().__init__(configs)

    def get_dataset(self, split):
        return _Dataset(split, self.configs)


class _Dataset(Dataset):
    def __init__(self, split, configs) -> None:
        super().__init__()
        self.configs = configs
        self.size = configs['{}_size'.format(split)]
        self.min = configs['min']
        self.max = configs['max']
        self.data_x = []
        self.data_y = []
        for _ in range(self.size):
            x = torch.rand(3)*(self.max-self.min) + self.min
            y = x[0]**2 + torch.exp(x[0]+x[1]) - 2*x[2]
            y = y.reshape(1)
            self.data_x.append(x)
            self.data_y.append(y)
        self.data_x = torch.stack(self.data_x)
        self.data_y = torch.stack(self.data_y)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            'x': self.data_x[index],
            'y': self.data_y[index]
        }
