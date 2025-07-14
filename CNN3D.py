import glob
import os
import torch
from torch.utils.data import Dataset
import cv2
import torchvision
import torchvision.transforms.functional as rgb_to_grayscale
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from early_stopping_pytorch import EarlyStopping
from torch import optim
from torchinfo import summary
from matplotlib import pyplot as plt
from classification_layer import ClassificationLayer

class CNNClassifayer(nn.Module):
    def __init__(self,num_c = 22):
        super(CNNClassifayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(3, 128, kernel_size=3, padding='same'),
                                   nn.BatchNorm3d(128),
                                   nn.ReLU(),
                                   nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))) #C,H,W
                                   
        self.conv2 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=3, padding='same'),
                                   nn.BatchNorm3d(256),
                                   nn.ReLU(),
                                   nn.Dropout3d(0.3),
                                   nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))
                                   
        self.conv3 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=3,  padding='same'),
                                   nn.BatchNorm3d(256),
                                   nn.ReLU(),
                                   nn.Dropout3d(0.3),
                                   nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))) 
                 
        self.classifier = ClassificationLayer(num_feat=256,n_classes=num_c)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_avg_pool3d(x, (1,1,1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    
if __name__ == "__main__":
    model = CNNClassifayer(22)
    model.eval()

    # Create a single random input with the desired shape: [batch, 1, 75, 224, 224]
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 16, 180, 180)

    # Optional: Move to GPU if available and model is on CUDA
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # input_tensor = input_tensor.to(device)


    output = model(input_tensor)