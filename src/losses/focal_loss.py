import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduce = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce

    def forward(self, input, target):
        p = F.softmax(input, dim=1)
        p_t = p * target + (1 - p) * (1 - target)
        ce_loss =  nn.CrossEntropyLoss()(input, target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        # loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-6)

        if self.reduce == 'mean': return loss.mean()
        return loss.sum()
