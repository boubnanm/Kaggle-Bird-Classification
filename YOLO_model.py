import warnings
warnings.filterwarnings("ignore")

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

if os.path.isdir("PyTorch_YOLO"): 
    sys.path.insert(0, 'PyTorch_YOLO')

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class Yolo_Bird_Detector():
    def __init__(self,args):
        
        self.args = args
        self.input_directory = args.image_folder
        self.output_folder = args.output
        self.padding = args.padding
        self.pad_size = args.pad_size
        
    def pad_resize(self,image):
        desired_size = self.pad_size
        x,y,z= image.shape
        ratio_x, ratio_y = desired_size/x, desired_size/y
        ratio = np.minimum(ratio_x,ratio_y)

        image=Image.fromarray(image)
        image=image.resize((int(np.ceil(y*ratio)),int(np.ceil(x*ratio))),Image.ANTIALIAS)
        xs, ys = image.size

        new_im = Image.new("RGB", (desired_size,desired_size))

        new_size = np.maximum(xs,ys)/2
        top = int(np.ceil(new_size - xs/2))
        left = int(new_size - ys/2)

        new_im.paste(image, (top,left))
        del image
        return new_im
    
    def detect_crop_birds(self):
        
        cuda = torch.cuda.is_available()

        # Set up model
        model = Darknet(self.args.config_path, img_size=self.args.img_size)
        model.load_weights(self.args.weights_path)

        if cuda:
            model.cuda()
            model = nn.DataParallel(model)

        model.eval() # Set in evaluation mode


        for data_folder in list(os.listdir(self.input_directory)):
            non_cropped = 0
            print("\nDetecting birds on :",data_folder)
            directory=self.input_directory+'/'+data_folder
            num_imgs = 0
            for folder in tqdm(list(os.listdir(directory))):
                num_imgs+=len(list(os.listdir(directory+'/'+folder)))
                
                os.makedirs(self.output_folder, exist_ok=True)
                os.makedirs(self.output_folder+'/'+data_folder+'/'+folder, exist_ok=True)

                dataloader = DataLoader(ImageFolder(directory+'/'+folder, img_size=self.args.img_size),
                                        batch_size=self.args.detector_batch_size, shuffle=False, num_workers=self.args.n_cpu)

                classes = load_classes(self.args.class_path) # Extracts class labels from file

                Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

                imgs = []           # Stores image paths
                img_detections = [] # Stores detections for each image index

                #print ('\n\tPerforming object detection..')
                prev_time = time.time()
                try: list(dataloader)[0]
                except Exception as e:
                    exception = e
                    for file in os.listdir(directory+'/'+folder):
                        i=plt.imread(directory+'/'+folder+'/'+file)
                        if len(i.shape)==2 or i.shape[2]!=3:
                            i=Image.fromarray(i)
                            i=i.convert('RGB')
                            i.save(directory+'/'+folder+'/'+file)
                    del i
                for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
                    # Configure input
                    input_imgs = Variable(input_imgs.type(Tensor))

                    # Get detections
                    with torch.no_grad():
                        detections = model(input_imgs)
                        detections = non_max_suppression(detections, 80, self.args.conf_thres, self.args.nms_thres)

                    # Log progress
                    #current_time = time.time()
                    #inference_time = datetime.timedelta(seconds=current_time - prev_time)
                    #prev_time = current_time
                    #print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

                    # Save image and detections
                    imgs.extend(img_paths)
                    img_detections.extend(detections)


                # Iterate through images and save cropped images
                for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

                    # Load img
                    img = np.array(Image.open(path))

                    # The amount of padding that was added
                    pad_x = max(img.shape[0] - img.shape[1], 0) * (self.args.img_size / max(img.shape))
                    pad_y = max(img.shape[1] - img.shape[0], 0) * (self.args.img_size / max(img.shape))
                    # Image height and width after padding is removed
                    unpad_h = self.args.img_size - pad_y
                    unpad_w = self.args.img_size - pad_x

                    # Bounding boxes and labels of detections
                    if detections is not None:
                        count=0
                        unique_labels = detections[:, -1].cpu().unique()
                        n_cls_preds = len(unique_labels)
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                            if cls_pred == classes.index("bird"):
                                count=1
                                #print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                                # Rescale coordinates to original dimensions
                                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                                x1, y1 = np.maximum(0,int(x1)-20), np.maximum(0,int(y1)-20)
                                x2, y2 = np.minimum(x1+box_w+40,img.shape[1]), np.minimum(y1+box_h+40,img.shape[0])
                                img = img[int(np.ceil(y1)):int(y2), int(np.ceil(x1)):int(x2), :]

                                # Save generated image with detections
                                path=path.split("/")[-1]
                                if self.padding : 
                                    img = np.array(self.pad_resize(img))

                                if "test" in data_folder:
                                    plt.imsave(self.args.output+'/'+data_folder+'/'+folder+'/'+path, img)
                                else:
                                    plt.imsave(self.args.output+'/'+data_folder+'/'+folder+'/'+path[:-4]+"_cropped.jpg", img)
                                plt.close()
                        if count==0:
                            non_cropped+=1
                    else:
                        non_cropped+=1
            print("\t{}% of {} images non cropped".format(np.round(100*non_cropped/num_imgs,2),data_folder))



