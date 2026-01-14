
import torch
import torch.nn as nn

def Neg_Sharpe(portfolio):
    return -torch.mean(portfolio) / torch.std(portfolio)

class SharpeLoss(nn.Module):
    def __init__(self):
        super(SharpeLoss, self).__init__()
    def forward(self, outputs, future_rets):
        portfolio = outputs * future_rets
        loss = Neg_Sharpe(portfolio)
        return loss