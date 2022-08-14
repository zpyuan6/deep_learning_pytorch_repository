import torchvision.models as models
import torch.nn as nn
import torch
from model.MRINet import MRINet
from model.VGGNet import VGG3DNet
from model.ResNet import Res3DNet18,Res3DNet34

from  torchsummary import summary


def load_model(model_name, input_shape, classes=['CN','MCI','AD']):
    model_mapping = {"DenseNet":models.DenseNet(block_config = (6, 12, 24, 16),num_classes=len(classes)),
    "DenseNet166Feature":models.DenseNet(block_config = (6, 12, 24, 16),num_classes=len(classes)),
    "DenseNet121":models.densenet121(num_classes=len(classes)),
    "MRINet":MRINet(num_classes=len(classes)),
    "VGG3DNet":VGG3DNet(num_classes=len(classes)),
    "Res3DNet18":Res3DNet18(num_classes=len(classes),input_shape=input_shape),
    "Res3DNet34":Res3DNet34(num_classes=len(classes),input_shape=input_shape)}

    model = model_mapping[model_name]

    if model_name=="DenseNet166Feature":
        model.features.conv0 = nn.Conv2d(228,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    summary(model,(1,128,128,128))

    return model

if __name__ == "__main__":
    model = models.densenet121(pretrained=True)
    torch.save(model.state_dict(), 'saved_model/pretrained_densenet121.pth')