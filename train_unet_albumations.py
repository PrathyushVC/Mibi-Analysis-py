




import torch
from torch import nn
from torch.utils.data import DataLoader

from albumentations import *
from albumentations.pytorch import ToTensor


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
    n_classes=28
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

