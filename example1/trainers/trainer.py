from torch_temp.trainers.base_trainer import BaseTrainer
import numpy as np
import torch


class Trainer(BaseTrainer):
    def __init__(self, model, dataset, loss):
        super().__init__(model, dataset, loss)

    def _metrics(self, outputs_model, inputs_data, mode='train') -> dict:
        with torch.no_grad():
            y_pred = outputs_model['y']
            y_gt = inputs_data['y']
            residual = torch.norm(y_pred-y_gt, p=2, dim=-1).cpu().numpy()
            metrics = {
                'residual_0.1': np.mean(residual <= 0.1),
                'residual_0.2': np.mean(residual <= 0.2),
                'residual_0.5': np.mean(residual <= 0.5),
                'residual_1': np.mean(residual <= 1),
            }
        return metrics
