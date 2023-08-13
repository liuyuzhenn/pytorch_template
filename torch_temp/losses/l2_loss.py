from .base_loss import BaseLoss
import torch.nn.functional as F
import torch


class L2Loss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)

    def compute(self, outputs_model, inputs_data):
        y_pred = outputs_model['y']
        y_gt = inputs_data['y']
        loss = torch.norm(y_pred-y_gt, p=2, dim=-1).mean()
        # optionally return some statistics stored in a dict
        return loss, {
            'residual': loss}
        
