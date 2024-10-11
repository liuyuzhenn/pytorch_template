from src.runners.base_runner import IterRunner, EpochRunner
import numpy as np
import torch


class FitIterRunner(IterRunner):
    def __init__(self, configs):
        super().__init__(configs)

    def _metrics(self, outputs_model, inputs_data, mode='train') -> dict:
        with torch.no_grad():
            y_pred = outputs_model['y']
            y_gt = inputs_data['y']
            residual = torch.norm(y_pred-y_gt, p=2, dim=-1)
            metrics = {
                'residual': residual.mean(),
                'residual_0.1': torch.mean((residual <= 0.1).float()),
                'residual_0.2': torch.mean((residual <= 0.2).float()),
                'residual_0.5': torch.mean((residual <= 0.5).float()),
                'residual_1': torch.mean((residual <= 1).float()),
            }
        return metrics

class FitEpochRunner(EpochRunner):
    def __init__(self, configs):
        super().__init__(configs)

    def _metrics(self, outputs_model, inputs_data, mode='train') -> dict:
        with torch.no_grad():
            y_pred = outputs_model['y']
            y_gt = inputs_data['y']
            residual = torch.norm(y_pred-y_gt, p=2, dim=-1)
            metrics = {
                'residual': torch.mean(residual),
                'residual_0.1': torch.mean((residual <= 0.1).float()),
                'residual_0.2': torch.mean((residual <= 0.2).float()),
                'residual_0.5': torch.mean((residual <= 0.5).float()),
                'residual_1': torch.mean((residual <= 1).float()),
            }
        return metrics
