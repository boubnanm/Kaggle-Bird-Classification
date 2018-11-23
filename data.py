import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 331 x 331 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set


# Training transformations
data_train_transforms = transforms.Compose([
    transforms.Resize((331, 331)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])


# Validation transformations
data_val_transforms = transforms.Compose([
    transforms.Resize((331, 331)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])