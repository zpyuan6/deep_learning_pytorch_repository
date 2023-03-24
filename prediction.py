import torch

import cv2
from cv2 import cvtColor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from utils.load_deep_learning_model import load_model

model_name = "DenseNet121"
model_path = "saved_model\\2D\DenseNet\DenseNet121-4_classes-[214, 214]_input_shape-ep020-loss0.000-val_loss0.032.pth"
input_shape = [214, 214]
classes = ["mild_demented", "moderate_demented", "non_demented", "very_mild"]
Cuda = True


def load_prediction_model(device):
    model = load_model(model_name, input_shape, classes=classes)

    print('Load weights {}.'.format(model_path))
    # state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
    model_dict = model.state_dict()
    # Loads an object saved with torch.save() from a file.
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items(
    ) if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    return model


def prediction(model, device, image_path=""):
    if image_path == "":
        img = input("Input image filename:")
    else:
        img = image_path

    try:
        print(f"Try to open image {img}")
        image = Image.open(img).convert('RGB')
    except:
        print('Open Error! Try again!')
    else:
        # image_shape = np.array(np.shape(image)[0:2])
        image = image.resize((input_shape[0], input_shape[1]), Image.BICUBIC)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_data = transform(image)
        # plt.imshow(np.transpose(image_data,(1,2,0)))
        # plt.show()
        image_data = np.expand_dims(image_data, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images.to(device)
            output = model(images)
            preds = torch.softmax(output, 1)
            result = torch.argmax(preds[0])
            print(f"Classify result: {classes[result]}")

    if image_path != "":
        return result, classes[result], output, image_data




def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() and Cuda else 'cpu')
    model = load_prediction_model(device)

    while True:
        prediction(model, device)


if __name__ == "__main__":
    main()
