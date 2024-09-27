from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, configs, split) -> None:
        super().__init__()
        self.configs = configs.dataset
        self.size = self.configs['{}_size'.format(split)]
        self.min = self.configs.min
        self.max = self.configs.max
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
