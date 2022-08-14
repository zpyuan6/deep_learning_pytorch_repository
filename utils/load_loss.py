import torch.nn as nn
import torch
from loss.focal_loss import FocalLoss

def load_loss_function(weight=None,loss_name="CrossEntropyLoss"):
    # CrossEntropyLoss()内部将input做了softmax后再与label进行交叉熵 BCEloss()内部啥也没干直接将input与label做了交叉熵 BCEWithLogitsLoss()内部将input做了sigmoid后再与label进行交叉熵
    loss_function_mapping = {
        "CrossEntropyLoss":nn.CrossEntropyLoss(weight=weight),
        "FocalLoss":FocalLoss(gamma=1,alpha=weight)
    }

    return loss_function_mapping[loss_name]