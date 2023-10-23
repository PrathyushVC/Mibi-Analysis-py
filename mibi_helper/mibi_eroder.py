'''
This script Takes as an input the npz files generated by MibiLoader and applies a systamatic erosion on them to create addition space then resaves that data.


inputs:
    root: The data dir
    expressiontypes: the expression types of interest as a list if strings. Note that by default this is built to assume you want a subset of 5 of them
    grp: list of grps currently assumes two
    T_path: the excel spread sheet of cell segmentations needed to run the code
    save_path: where to write the resulting npz files. if you dont provide it it will use your cwd
outputs:
    will generate a npz file with the expression image, the segmentation, and the clustered segmentation.
    (we may need to change this so that the output is a 3d volume which would save space)
    
    '''

import os,sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time


def mibi_eroder(segmentation):
    if segmentation is None:
        raise ValueError("segmentation is empty . It should have a valid value.")


    unique_regions = np.unique(segmentation)

    merged_mask = np.zeros_like(segmentation, dtype=np.int32)

    for region_value in unique_regions:
        #print(region_value)
    # Create a binary mask for the current region
        binary_mask = (segmentation == region_value).astype(np.uint8)
    
    # Perform a 1-pixel erosion operation using OpenCV
        kernel = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.uint8)
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    
        # Assign the original region value to the eroded pixels
        merged_mask[eroded_mask == 1] = region_value
    return merged_mask
