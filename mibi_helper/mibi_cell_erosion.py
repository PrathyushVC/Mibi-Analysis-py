

import numpy as np
import cv2
from multiprocessing import Pool


def mibi_eroder(segmentation,kernel=np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=np.uint8)):
    """
    Performs erosion on the input segmentation using a specified or default kernel.

    This function iterates through each unique region in the segmentation array, creating a binary mask for each region. 
    It then applies a morphological erosion operation to the binary mask using the provided kernel. 
    The result is a merged mask where each pixel is assigned the original region value if it was part of the eroded region.

    Args:
    - segmentation (np.ndarray): A Numpy array representing the segmentation data.
    - kernel (np.ndarray): A 2D Numpy array representing the structuring element for erosion. 
                           Default is a 3x3 cross-shaped kernel.

    Returns:
    - np.ndarray: A Numpy array with the result of the erosion operation applied to each region, 
                  saved as a 32-bit integer array.
    """

    if segmentation is None:
        raise ValueError("segmentation is empty . It should have a valid value.")


    unique_regions = np.unique(segmentation)

    merged_mask = np.zeros_like(segmentation, dtype=np.int32)

    for region_value in unique_regions:
        #print(region_value)
    # Create a binary mask for the current region
        binary_mask = (segmentation == region_value).astype(np.uint8)
    
    # Perform a 1-pixel erosion 
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    
        # Assign the original region value to 
        merged_mask[eroded_mask == 1] = region_value
    return merged_mask

def mibi_eroder_parallel(segmentation,kernel=np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=np.uint8)):
    """
    mibi_eroder_parallel(segmentation, kernel): 
    Performs erosion on the input segmentation using a specified or default kernel in parallel.
    
    Args:
    - segmentation: Numpy array representing the segmentation data.
    - kernel: Structuring element for erosion. Default is a 3x3 cross-shaped kernel.
              You can provide a custom kernel as a 2D Numpy array.
    
    Returns:
    - Merged mask: Numpy array with the result of the erosion operation applied to each region.
                   The result is saved as a 32-bit integer array.
    
    """
    if segmentation is None:
        raise ValueError("segmentation is empty . It should have a valid value.")

    unique_regions = np.unique(segmentation)
    merged_mask = np.zeros_like(segmentation, dtype=np.int32)

    def process_region(region_value):
        # Create a binary mask for the current region
        binary_mask = (segmentation == region_value).astype(np.uint8)

        # Perform a 1-pixel erosion 
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

        # Assign the original region value to 
        merged_mask[eroded_mask == 1] = region_value

    with Pool() as pool:
        pool.map(process_region, unique_regions)

    return merged_mask
