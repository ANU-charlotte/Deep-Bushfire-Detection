import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from model import create_model,generate_prediction_with_updated_features
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)
import time
import torch
import torch.nn as nn
from torchvision import models
from GNN import GNNNet
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output[0][0].detach()
    return hook

image = cv2.imread('data/testing/image_0.jpg')
image = image.astype(np.float32)
image /= 255.0
image = np.transpose(image, (2, 0, 1)).astype(np.float32)
image = torch.tensor(image, dtype=torch.float)
image = torch.unsqueeze(image, 0)

image2 = cv2.imread('data/testing/image_1.jpg')
image2 = image2.astype(np.float32)
image2 /= 255.0
image2 = np.transpose(image2, (2, 0, 1)).astype(np.float32)
image2 = torch.tensor(image2, dtype=torch.float)
image2 = torch.unsqueeze(image2, 0)

image3 = cv2.imread('data/testing/image_2.jpg')
image3 = image3.astype(np.float32)
image3 /= 255.0
image3 = np.transpose(image3, (2, 0, 1)).astype(np.float32)
image3 = torch.tensor(image3, dtype=torch.float)
image3 = torch.unsqueeze(image3, 0)

in_features, model, num_classes = create_model(num_classes=2)
model = generate_prediction_with_updated_features(model, in_features, num_classes)
checkpoint = torch.load('outputs/best_model.pth', map_location=torch.device('cpu'))
model.backbone.fpn.extra_blocks.register_forward_hook(get_activation('backbone.fpn.extra_blocks'))
#model.roi_heads.box_head.fc6.register_forward_hook(get_activation('roi_heads.box_head.fc6'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
output = model(image)
f1 = activation['backbone.fpn.extra_blocks']

output = model(image2)
f2 = activation['backbone.fpn.extra_blocks']

output = model(image2)
f3 = activation['backbone.fpn.extra_blocks']
#print(aa.size()) # torch.Size([1, 256, 192, 336])

GNNmodel = GNNNet(2)

GNNmodel(f1,f2,f3)
#print(GNNmodel)
