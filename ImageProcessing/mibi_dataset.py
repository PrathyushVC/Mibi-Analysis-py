#TODO replace the print handling in this class with proper logging. pvc5
#TODO clean up the binarization to handle variable groups
import os
import tifffile as tiff
import numpy as np
import torch 
from torch import nn 
from torch.utils.data import Dataset
import polars as pl
import logging

class MibiDataset(Dataset):
    """
    MibiDataset is a PyTorch Dataset class for handling TIFF images and their corresponding labels.

    This dataset is designed to load images from a specified directory, patch them into smaller squares of configurable sizes,
    and provide the necessary transformations. It also supports binarization of group labels based on a provided DataFrame.

    Attributes:
        root_dir (str): The root directory containing the FOV directories with TIFF images.
        patch_size_x (int): The height of the patches to be extracted from the images.
        patch_size_y (int): The width of the patches to be extracted from the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        image_paths (list): List of paths to the TIFF images.
        labels (list): List of corresponding labels for the images.
        expressions: Approved lists of protein expressions as .tif files that are shared across desired samples
    """
# Dataset to handle FOVs and TIFF images, patch them into squares of configurable sizes
    def __init__(self, root_dir,df, patch_size_x=128, patch_size_y=128,prefix='FOV',fov_col='FOV',label_col='Group', transform=None,expressions=None):
        self.root_dir = root_dir
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.transform = transform
        self.expressions=expressions
        self.image_paths,self.labels = self._load_image_paths(prefix,df,fov_col,label_col)#Needs to be last as it needs everything else to exist
        
        
    def _load_image_paths(self,prefix,df,fov_col,label_col):
        
        """
        Loads the image paths and corresponding labels from the specified DataFrame.

        This method constructs a list of image file paths by iterating through the directories in the root directory.
        It checks if the directory names match the specified prefix and if they exist in the provided DataFrame's 
        FOV column. If a match is found, it retrieves the corresponding label from the label column, binarizes it, 
        and appends the image paths and labels to their respective lists.

        Parameters:
            prefix (str): The prefix to filter the FOV directories.
            df (DataFrame): The DataFrame containing FOV and label information.
            fov_col (str): The column name in the DataFrame that contains FOV identifiers.
            label_col (str): The column name in the DataFrame that contains the labels.

        Returns:
            tuple: A tuple containing two lists - the first list contains the image paths, and the second list contains the corresponding labels.
        """
        image_paths = []
        labels=[]
        
        #Iterate over each FOV directory
        #five if checks to achieve this is hard to parse need to rethink approac
        for fov_dir in os.listdir(self.root_dir):
            if fov_dir.startswith(prefix):#replace with Pathlib matching later
                fov_path = os.path.join(self.root_dir, fov_dir)
                
                if os.path.isdir(fov_path):
                    #print(fov_dir)
                    # check if the folder name exists in the "FOV" column of the DataFrame
                    group = df.filter(pl.col(fov_col) == fov_dir)  #why oh why did I decide to decide to use polars instead of pandas for this test?
                    if group.height>0:
                        # pull the matching data from the "group" column
                        
                        group_data = group[label_col].to_list()[0]  
                        binarized_data = self._binarize_group(group_data)  # Binarize the group data
                        #print(group_data,binarized_data)
                        tif_path = os.path.join(fov_path, 'TIFs')
                        if os.path.exists(tif_path):
                            # Load all .tiff files ignoring those with "segmentation in the name"
                            sublist=[]
                            labels.append(binarized_data)
                            for image_file in os.listdir(tif_path):
                                if self.expressions:# this is a list of expression names we want to make sure are in the dataset. 
                                    if image_file in self.expressions:
                                        sublist.append(os.path.join(tif_path, image_file))
                                else:
                                    if (image_file.endswith('.tif') or image_file.endswith('.tiff')) and ('segmentation' not in image_file) and not (image_file[0].isdigit()):
                                        sublist.append(os.path.join(tif_path, image_file))
                            

                            sublist.sort()#Should normalize the data assuming the data format is maintained true for this dataset 
                            #should create a perminant mapping
                            image_paths.append(sublist)
                        num_sublists = len(sublist)
                        logging.info(f"FOV: {fov_dir}, Number of sublists: {num_sublists}")

                        #else:
                            #print(f"Skipped {fov_dir} because TIF does not exist")
                    #else:
                       #print(f"Skipped {fov_dir} because its not in the table of labels")
                #else:
                    #print(f"Skipped {fov_dir} because its not in the root_dir")

        return image_paths,labels
    
    def _binarize_group(self,group_data):
        #Assumes a desired class structure 
        if str(group_data) in ['G1', 'G4']:
            return 1 
        elif str(group_data) in ['G2', 'G3']:
            return 0  
        else:
            raise Exception("Group Data does not align with Expected Groups")  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and label data for a specific index in the dataset.

        Parameters:
            idx (int): The index of the data item to retrieve.

        Returns:
            tuple: A tuple containing:
                - patches (list): A list of image patches extracted from the stacked image.
                - label (int): The label corresponding to the data item.
                - patches_loc (list): A list of tuples indicating the top-left coordinates of each patch.
        
        Raises:
            IndexError: If the index is out of bounds for the dataset.
        """
        fov_paths = self.image_paths[idx]
        image = self._image_stacker(fov_paths=fov_paths)  # Load 2048x2048 image
        label=self.labels[idx]
        
        patches = []
        patches_loc=[]
        for i in range(0, image.shape[1], self.patch_size_x):
            for j in range(0, image.shape[2], self.patch_size_y):
                patch = image[:,i:i+self.patch_size_x, j:j+self.patch_size_y]
                
                if self.transform:
                    patch = self.transform(patch)
                #print(patch.shape)
                patches.append(patch)
                patches_loc.append((i,j))
        
        return patches,label,patches_loc,fov_paths
    
    def _image_stacker(self,fov_paths):

        stacked_image = None
        for path in fov_paths:
            image = tiff.imread(path) 
            if stacked_image is None:
                stacked_image = np.expand_dims(image,axis=0)  
            else:
                stacked_image = np.concatenate((stacked_image, np.expand_dims(image,axis=0)), axis=0)  
    
        return stacked_image

