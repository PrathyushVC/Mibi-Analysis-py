
import torch
from torch import nn
from torch.utils.data import DataLoader
import albumentations as alb
from albumentations.pytorch import ToTensorV2


from unet import UNet #code borrowed from https://github.com/jvanvugt/pytorch-unet

import PIL
import matplotlib.pyplot as plt
import cv2

import numpy as np
import sys, glob

from tensorboardX import SummaryWriter

import scipy.ndimage 
import time
import math
import tables
import random

from sklearn.metrics import confusion_matrix

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


if __name__=="__main__":
    dataname="Mibi_trial"
    gpuid=0
    #Unet Params
    n_classes=3
    in_channels=3
    padding=True
    depth=5
    wf=2
    up_mode='upconv'
    batch_norm=True
    #Training params
    batch_size=3
    patch_size=256
    num_epochs=100
    edge_weight=1.1
    phases=['train','val']
    validation_phases=["val"]

    transforms=alb.Compose([alb.VerticalFlip(p=0.5),alb.HorizontalFlip(p=0.5), 
    alb.Rotate(p=0.75, border_mode=cv2.BORDER_CONSTANT,value=0), 
    alb.RandomSizedCrop((patch_size,patch_size), patch_size,patch_size),
    alb.ToTensor()   
       ])
    
    dataset={}
    dataLoader={}
    for phase in phases:
        dataset[phase]=Dataset(r"..\..\Scratch\{dataname}_{phase}.pytable", transforms= transforms ,edge_weight=edge_weight)

    (img,patch_mask,patch_mask_weight)=dataset["train"][6]
    fig,ax=plt.subplots(1,4,figsize(10,4))

    ax[0].imshow(np.moveaxis(img.numpy(),0,-1))
    ax[1].imshow(patch_mask==1)
    ax[2].imshow(patch_mask_weight)
    ax[3].imshow(patch_mask)

    optim=torch.optim.SparseAdam(model.parameters())
    print(model.parameters())

    nclasses = dataset["train"].numpixels.shape[1]
    class_weight=dataset["train"].numpixels[1,0:2] #don't take ignored class into account here
    class_weight = torch.from_numpy(1-class_weight/class_weight.sum()).type('torch.FloatTensor').to(device)

    print(class_weight) #show final used weights, make sure that they're reasonable before continouing
    criterion = nn.CrossEntropyLoss(weight = class_weight, ignore_index = ignore_index ,reduce=False) #reduce = False makes sure we get a 2D output instead of a 1D "summary" value


