
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import h5py
import json
import warnings
from PIL import Image
from graph_helper import graph_maker

import tables
import glob, cv2, random
import PIL
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection


def extract_patches_with_stride(image, patch_size, stride):
    """
    Extract patches from an image with a specified stride size.

    Parameters:
    - image: NumPy array representing the image.
    - patch_size: Tuple (height, width) specifying the size of each patch.
    - stride: integer specifying the stride size.

    Returns:
    - patches: List of patches extracted from the image.
    """

    # Get the dimensions of the image and patches
    image_height, image_width = image.shape[:2]
    patch_height, patch_width = patch_size[:2]
    vertical_stride = stride
    horizontal_stride=stride
    # Calculate the number of patches in each dimension
    num_patches_vertical = ((image_height - patch_height) // vertical_stride) + 1
    num_patches_horizontal = ((image_width - patch_width) // horizontal_stride) + 1

    # Initialize a list to store the extracted patches
    patches = np.empty((num_patches_vertical, num_patches_horizontal, patch_height, patch_width, image.shape[2]), dtype=image.dtype)

    # Extract patches using nested loops
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            start_y = i * vertical_stride
            end_y = start_y + patch_height
            start_x = j * horizontal_stride
            end_x = start_x + patch_width

            patch = image[start_y:end_y, start_x:end_x]
            patches[i, j] = patch

    return patches




if __name__ == "__main__":

    dataname="Mibi_TOFF"
    patch_size=64 #size of the tiles to extract and save in the database, must be >= to training size
    stride_size=64 #distance to skip between patches, 1 indicated pixel wise extraction, patch_size would result in non-overlapping tiles
    mirror_pad_size=64 # number of pixels to pad *after* resize to image with by mirroring (edge's of patches tend not to be analyzed well, so padding allows them to appear more centered in the patch)
    #test_set_size=.1 what percentage of the dataset should be used as a held out validation/testing set
    #resize=1 resize input images
    class_names=["G1","G2","G3","G4"] #what classes we expect to have in the data, here we have only 2 classes but we could add additional classes

    random.seed(42)
    img_dtype=tables.UInt8Atom()
    filenameAtom=tables.StringAtom(itemsize=255)
    file=glob.glob(r"D:\MIBI-TOFF\Data_Full\PN1\FOV2_PN1_CD4.npz") # create a list of the files, in this case we're only interested in files which have masks so we can use supervised learning

    storage={} #holder for future pytables

    block_shape=np.array((patch_size,patch_size,3)) #block shape specifies what we'll be saving into the pytable array, here we assume that masks are 1d and images are 3d
    filters=tables.Filters(complevel=6, complib='zlib') #we can also specify filters, such as compression, to improve storage speed
    output_hdf5_path = "output_patches.h5"  # Output HDF5 file name


    print(f"Patches saved to {output_hdf5_path}")