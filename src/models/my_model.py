import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.model_configs = configs['model_configs']
        self.model = nn.Sequential(
            nn.Linear(self.model_configs['dim_in'], self.model_configs['dim_hidden']),
            nn.ReLU(),
            nn.Linear(self.model_configs['dim_hidden'], self.model_configs['dim_out'])
        )

    def forward(self, inputs_data, mode='train'):
        inputs_data = inputs_data['x']
        y = self.model(inputs_data)
        return {'y': y}
