import warnings
warnings.filterwarnings("ignore")

from subprocess import check_output
import os
import sys
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# Settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
# Detecting setting 
parser.add_argument('--image_folder', type=str, default='bird_dataset', help='path to dataset')
parser.add_argument('--config_path', type=str, default='PyTorch_YOLO/config/yolov3.cfg', help='path to Yolo model config file')
parser.add_argument('--weights_path', type=str, default='PyTorch_YOLO/weights/yolov3.weights', help='path to Yolo weights file')
parser.add_argument('--class_path', type=str, default='PyTorch_YOLO/data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--detector_batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--output', type=str, default='bird_dataset_output', help='path to output detections')
parser.add_argument('--padding', action='store_true', dest='padding', help='Enable image padding to conserve ratio')
parser.add_argument('--no-padding', action='store_false', dest='padding',  help='Disable image padding to conserve ratio')
parser.add_argument('--crop', action='store_true', dest='crop', help='Enable image cropping')
parser.add_argument('--no-crop', action='store_false', dest='crop',  help='Disable image cropping')
parser.add_argument('--pad_size', type=int, default=331, help='size of padded image')
parser.set_defaults(padding=False)
parser.set_defaults(crop=True)

# Training settings
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=47, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--merge', action='store_true', dest='merge', help='Enable train on both val and train sets')
parser.add_argument('--no-merge', action='store_false', dest='merge',  help='Disable train on both val and train sets')
parser.set_defaults(merge=False)

args = parser.parse_args()


# Detect birds and generate new images
if args.crop:
    # Create output folder
    if not os.path.isdir(args.output):
        os.makedirs(args.output)    
  
    # Download pretrained Yolo
    if not os.path.isdir('PyTorch_YOLO'):
        check_output('git clone https://github.com/eriklindernoren/PyTorch-YOLOv3.git PyTorch_YOLO', shell=True)
        if 'yolov3.weights' not in list(os.listdir("PyTorch_YOLO/weights")):
            check_output('wget https://pjreddie.com/media/files/yolov3.weights -O '+args.weights_path, shell=True) 
    
    # Copying the original images to cropped folder
    for folder in list(os.listdir("bird_dataset")):
        check_output('cp -fr bird_dataset/'+folder+' bird_dataset_output', shell=True) 
    args.data=args.output
    from YOLO_model import Yolo_Bird_Detector
    Bird_Detector=Yolo_Bird_Detector(args)
    Bird_Detector.detect_crop_birds()
    
    
# Merging val and train dataset for Kaggle submission
if args.merge:
    # Create merged folder
    if not os.path.isdir("bird_dataset_merged"):
        os.makedirs("bird_dataset_merged")
    check_output('cp -fr bird_dataset_output/. bird_dataset_merged', shell=True)
    check_output('cp -fr bird_dataset_merged/val_images/. bird_dataset_merged/train_images/', shell=True)
    args.data='bird_dataset_merged'
    
    
# Cuda use and seed set
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_train_transforms, data_val_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_train_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_val_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = nn.DataParallel(Net())
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

multi_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15,25,40,60], gamma=0.3)

def train(epoch):
    multi_lr_scheduler.step()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        if isinstance(output, tuple):
            loss = sum((criterion(o,target) for o in output))
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
   
    
for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')
