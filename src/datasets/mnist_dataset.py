from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, configs, split) -> None:
        super().__init__()
        self.configs = configs.dataset

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            'x': self.data_x[index],
            'y': self.data_y[index]
        }
