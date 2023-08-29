import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        configs = configs['model_configs']
        self.model = nn.Sequential(
            nn.Linear(configs['dim_in'], configs['dim_hidden']),
            nn.ReLU(),
            nn.Linear(configs['dim_hidden'], configs['dim_out'])
        )

    def forward(self, inputs_data):
        inputs_data = inputs_data['x']
        y = self.model(inputs_data)
        return {'y': y}
