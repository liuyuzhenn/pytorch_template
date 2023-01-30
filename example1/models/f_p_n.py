import torch.nn as nn


class FPN(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.model = nn.Sequential(
            nn.Linear(configs['dim_in'], configs['dim_hidden']),
            nn.ReLU(),
            nn.Linear(configs['dim_hidden'], configs['dim_out'])
        )

    def forward(self, x):
        x = x['x']
        y = self.model(x)
        return {'y': y}
