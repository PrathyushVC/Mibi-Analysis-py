
import numpy as np
import h5py
import json
import warnings

import torch
from torch_geometric.data import Dataset
from torchvision import transforms
from PIL import Image
from graph_maker import graph_maker

class ImagePatchDataset(Dataset):
    def __init__(self, image_paths, patch_size, transform=None, randomize=0, overlap=False):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.transform = transform
        self.randomize = randomize
        self.overlap = overlap

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_dict = np.load(image_path)
        

        patches = self.segment_image(image)
        if self.randomize > 0:
            patches = self.randomize_patches(patches)

        if self.transform:
            patches = [self.transform(patch) for patch in patches]

        data = self.process_patches_to_graph(patches)
        return data

    def segment_image(self, image):
        width, height = image.size
        patches = []

        step_size = self.patch_size // 2 if self.overlap else self.patch_size

        for i in range(0, height - self.patch_size + 1, step_size):
            for j in range(0, width - self.patch_size + 1, step_size):
                patch = image.crop((j, i, j + self.patch_size, i + self.patch_size))
                patches.append(patch)

        return patches

    def randomize_patches(self, patches):
        num_patches_to_select = int(len(patches) * self.randomize)
        selected_indices = torch.randperm(len(patches))[:num_patches_to_select]
        patches = [patches[i] for i in selected_indices]
        return patches

    def process_patches_to_graph(self, patches):
        

        
        pass

# Example usage:
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
patch_size = 128
transform = transforms.Compose([transforms.ToTensor()])
randomize = 0.2
overlap = False

dataset = ImagePatchDataset(image_paths, patch_size, transform, randomize, overlap)
data = dataset[0]  # Accessing the first graph data in the dataset


if __name__ == "__main__":
    input_image_path = r"D:\MIBI-TOFF\Data_Full\PN1\FOV2_PN1_CD4.npz"
    image_dict=np.load(input_image_path)
    

    patch_size = 256 
    stride = 64  
    randomize = True  
    overlap = True  

    output_hdf5_path = "output_patches.h5"  # Output HDF5 file name

    patches = segment_image(image_dict, patch_size, stride, randomize, overlap)
    save_patches_to_hdf5(patches, output_hdf5_path)

    print(f"Patches saved to {output_hdf5_path}")