import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from skimage.measure import label, regionprops
import ray
def centroid_compute(segmentation=None,region_mask=None):
    ''' 
    inputs
        segmentation: a numpy array with unique labels for each cell that you want to compute centroids from
        region_mask: optional binary mask which is used to retain only cells of interest(specific cell type or group) 
    outputs
        centroids: list of centroids
        regions: output of regionprops if needed
        Note these will be returned as a tuple and will need to handled correctly
    '''
    if (segmentation is None) or  not (isinstance(segmentation, np.ndarray)):
        raise Exception("Segmentation mask was not provided or is not a numpy")
    if  region_mask is not None:
        if not (isinstance(region_mask, np.ndarray)) or not (region_mask.shape==segmentation.shape) :
            raise Exception("Region Mask mask is not a numpy or is a different shape then the segmentation")
        else:   
            segmentation=segmentation*region_mask
    regions=regionprops(segmentation)
    centroids = np.array([region.centroid for region in regions])
    return centroids, regions

#In progress and needs to be reviewed with mohsen or toms
@ray.remote
def pairwise_distances(chunk, centroids,num_chunks):
    start_idx = chunk * len(centroids) // num_chunks
    end_idx = (chunk + 1) * len(centroids) // num_chunks
    return np.linalg.norm(centroids[start_idx:end_idx, None] - centroids, axis=2)

def dist_comp(centroids, num_chunks=1):
    '''
    Compute pairwise distances between centroids using Ray for parallelization.

    inputs
        centroids: Numpy array of centroids
        num_chunks: Number of chunks for parallelization
    outputs
        distances: Pairwise distances between centroids
    '''
    if num_chunks <= 1:
        # If num_chunks is 1 or less, compute distances without parallelization
        distances = np.linalg.norm(centroids[:, None] - centroids, axis=2)
        return distances
    ray.init()
    ray.available_resources()
    # Split centroids into chunks for parallel processing
    chunk_ids = [pairwise_distances.remote(chunk, centroids,num_chunks) for chunk in range(num_chunks)]
    distances = np.concatenate(ray.get(chunk_ids), axis=0)

    return distances




