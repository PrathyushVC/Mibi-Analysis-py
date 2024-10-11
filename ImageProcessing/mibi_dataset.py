import os
import tifffile as tiff
import numpy as np
import torch 
from torch import nn 
from torch.utils.data import Dataset

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
    """
# Dataset to handle FOVs and TIFF images, patch them into squares of configurable sizes
    def __init__(self, root_dir,df, patch_size_x=128, patch_size_y=128,prefix='FOV',fov_col='FOV',label_col='Group', transform=None):
        self.root_dir = root_dir
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.transform = transform
        self.image_paths,self.labels = self._load_image_paths(prefix,df,fov_col,label_col)

    def _load_image_paths(self,prefix,df,fov_col,label_col):
        """
        Loads the image paths and corresponding labels from the specified DataFrame.

        This method constructs a list of image file paths by iterating through the directories in the root directory.
        It checks if the directory names match the specified prefix and if they exist in the provided DataFrame's 
        FOV column. If a match is found, it retrieves the corresponding label from the label column, binarizes it, 
        and appends the image paths and labels to their respective lists.

        Args:
            prefix (str): The prefix to filter the FOV directories.
            df (DataFrame): The DataFrame containing FOV and label information.
            fov_col (str): The column name in the DataFrame that contains FOV identifiers.
            label_col (str): The column name in the DataFrame that contains the labels.

        Returns:
            tuple: A tuple containing two lists - the first list contains the image paths, and the second list contains the corresponding labels.
        """
        image_paths = []
        labels=[]
        
        # Iterate over each FOV directory
        for fov_dir in os.listdir(self.root_dir):
            if fov_dir.startswith(prefix):
                fov_path = os.path.join(self.root_dir, fov_dir)
                
                if os.path.isdir(fov_path):
                    # check if the folder name exists in the "FOV" column of the DataFrame
                    group = df[df[fov_col] == fov_dir]  # Compare with the "FOV" column
                    if not group.empty:
                        # pull the matching data from the "group" column
                        group_data = group[label_col].values[0]  
                        binarized_data = self.binarize_group(group_data)  # Binarize the group data
                        
                        tif_path = os.path.join(fov_path, 'TIF')
                        if os.path.exists(tif_path):
                            # Load all .tiff files ignoring those with "segmentation in the name"
                            for image_file in os.listdir(tif_path):
                                if image_file.endswith('.tiff') and 'segmentation' not in image_file:
                                    image_paths.append(os.path.join(tif_path, image_file))
                                    labels.append(binarized_data)
            
        return image_paths
    def _binarize_group(group_data,mapping=None):
        if mapping is not None:
            #Need to implement this later
            raise Exception("Only Default mapping currently implemented")
        else:
            # Default mapping behavior
            if str(group_data) in ['G1', 'G4']:
                return 1  # Map groups 1 and 4 to label 1
            elif str(group_data) in ['G2', 'G3']:
                return 0  # Map groups 2 and 3 to label 0
            else:
                raise Exception("Group Data does not align with Expected Groups")  # Return original value if it doesn't match any group


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = tiff.imread(image_path)  # Load 2048x2048 image
        label=self.labels[idx]
        
        patches = []
        for i in range(0, image.shape[0], self.patch_size_x):
            for j in range(0, image.shape[1], self.patch_size_y):
                patch = image[i:i+self.patch_size_x, j:j+self.patch_size_y]
                
                if self.transform:
                    patch = self.transform(patch)
                
                patches.append(patch)
        
        # Convert list of patches into a tensor
        patches = torch.stack(patches)  # Shape: (num_patches, patch_size_x, patch_size_y)
        return patches,label