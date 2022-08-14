from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma=1, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        cross_entropy = super().forward(input_, target)

        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = (target * (target != self.ignore_index)).to(dtype=torch.int64)

        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        # print(input_)
        # print(F.softmax(input_, 1))
        # print(target)
        # print(input_prob)

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss