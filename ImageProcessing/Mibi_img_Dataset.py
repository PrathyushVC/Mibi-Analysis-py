import os
import numpy
import numpy as np
import torch 
from torch import nn 
from torch.utils.data import Dataset
from albumentations import *
import tables
import scipy.ndimage 

class Mibi_img_Dataset(Dataset):
    def __init__(self,fname,transform=None,edge_weight=False):
        self.fname = fname
        self.transform = transform
        self.edge_weight = edge_weight
        self.data = tables.open_file(fname)
        self.numpixels=self.tables.root.numpixels[:]
        self.nitems=self.tables.root.images.shape[0]
        self.tabels.close()
        self.img=None
        self.mask=None

    def __getitem__(self,index):
        with tables.open_file(filename=self.name,mode='r')as db:
            self.img=db.root.img
            self.mask=db.root.mask

            img=self.img[index,:,:,:]
            mask=self.mask[index,:,:,:]
        if self.edge_weight is not None:
            weight=scipy.ndimage.morphology.binary_dilation(mask=1,iterations=2)&~mask
        else:
            weight=np.ones(mask.shape,dtype=mask.dtype)
        img_new =img
        mask_new=mask
        weight_new=weight

        if self.transforms:
            augmented=self.transforms(image=img,masks=[mask,weight])
            img_new=augmented['image']
            mask_new,weight_new=augmented['masks']

        return img_new,mask_new,weight_new
    
    def __len__(self):
        return self.nitems

  
