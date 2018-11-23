import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from tqdm import tqdm

import os
import sys
import time
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        ## InceptionV3 features extractor        
        self.inc = models.inception_v3(pretrained=True)
        self.inc.aux_logits = False
        # Freezing first layers
        for child in list(self.inc.children())[:-2]:
            for param in child.parameters():
                param.requires_grad = False
        # Removing the softmax layer
        self.inc.fc = nn.Sequential()
        
        
        ## ResNet152 features extractor        
        self.res152 = models.resnet152(pretrained=True)
        # Freezing first layers
        for child in list(self.res152.children())[:-3]:
            for param in child.parameters():
                param.requires_grad = False
        # Removing the softmax layer
        self.res152 = nn.Sequential(*list(self.res152.children())[:-1])
       
        self.Avg = nn.AvgPool2d(4)
        self.ReLU = nn.ReLU()
        
        self.linear = nn.Linear(4096, nclasses)


    def forward(self, x):
        x1 = self.Avg(self.ReLU(self.res152(x)))
        x1 = x1.view(-1, 2048)
        
        x2 = self.inc(x).view(-1, 2048)
        x = torch.cat([x1,x2],1)
        
        return self.linear(x)