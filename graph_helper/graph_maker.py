import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from skimage.measure import regionprops

import ray

def seg_to_graph(segmentation=None,max_distance=None,FOV_table=None,num_chunks=1):
    """
    Convert a segmentation image and related data into a networkx graph.

    Parameters:
        - segmentation (numpy.ndarray): A segmentation image representing regions.
        - max_distance (float): The maximum distance threshold for creating edges between nodes.
        - FOV_table (pandas.DataFrame): A DataFrame containing data associated with regions.
        - num_chunks (int): The number of chunks for computing distances between regions.

    Returns:
        - G_test (networkx.Graph): A networkx graph representing regions and their relationships.

    Raises:
        - ValueError: If any of the required inputs (segmentation, FOV_table) are missing or empty.

    This function takes a segmentation image, a maximum distance threshold, a DataFrame with data related to the regions, and an optional number of chunks for distance computation. It constructs a networkx graph that represents regions as nodes and their relationships based on the provided distance threshold.

    The nodes in the graph are associated with various attributes, including cell type, CD4, CD45, CD8, area, and centroid information extracted from the FOV_table DataFrame.

    The graph is returned as output.
    """
    if segmentation is None:
        raise ValueError("Missing input is empty. It should have a valid segmentation,and FOV_table, and distance")
    # Create a mapping from node indices to regions

    # Extract centroids into an array as well as compute the regions of the segmentations
    centroids,regions=centroid_compute(segmentation=segmentation)
    distances=dist_comp(centroids, num_chunks=num_chunks)
    G_test = nx.Graph()

# Add nodes to the graph
    for i, region in enumerate(regions):
        Fmatching_row = FOV_table[FOV_table['label'] == segmentation[int(region.centroid[0]), int(region.centroid[1])]].head(1)
        #print(Fmatching_row)
        if str(Fmatching_row['class'].values[0]).lower()=='immune':
            cell_type=0
        else:
            cell_type=1
            #Add all the needed protein info
        G_test.add_node(i,cell_type=cell_type,CD4=Fmatching_row['CD4'].values[0],CD45=Fmatching_row['CD45'].values[0], CD8=Fmatching_row['CD8'].values[0],area=region.area, centroid=region.centroid)
    
    added_edges = set()
#add edges to graph
    row_indices, col_indices = np.where((distances <= max_distance) & (distances>0)) # Find
    index_pairs = list(zip(row_indices, col_indices))

    #Code to avoid duplicates until I find a way to compute the distance graph so that it only fills the top triangle.
    for source, target in index_pairs:
        if source == target:
            continue  

        weight = distances[source, target]

        # Check if the edge (source, target) or (target, source) already exists
        if (source, target) in added_edges or (target, source) in added_edges:
            continue  # Skip adding duplicate edges
        else:
            G_test.add_edge(source, target, weight=weight)
            added_edges.add((source, target))

    return G_test

def centroid_compute(segmentation=None):
    ''' 
    Parameters:
        -segmentation: a numpy array with unique labels for each cell that you want to compute centroids from
        -region_mask: optional binary mask which is used to retain only cells of interest(specific cell type or group) 
    Returns:
        -centroids: list of centroids
        -regions: output of regionprops if needed
        -Note these will be returned as a tuple and will need to handled correctly
    '''
    if (segmentation is None) or  not (isinstance(segmentation, np.ndarray)):
        raise Exception("Segmentation mask was not provided or is not a numpy")
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

    Parameters:
        -centroids: Numpy array of centroids
        -num_chunks: Number of chunks for parallelization
    Returns
        -distances: Pairwise distances between centroids
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
    distances = np.triu(distances, k=1)  # k=1 to exclude the diagonal


    return distances




