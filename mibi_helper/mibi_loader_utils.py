

import os
import numpy as np
import pandas as pd
from tifffile import TiffFile
import json
import warnings
import concurrent.futures 


def mibi_loader(root=None, expressiontypes=None, T_path=None,save_directory = None,updated_cluster_map=None):

    """
    Loads Mibi data from the specified root directory and prepares it for analysis.

    Args:
        root (str): The root directory containing the data files.
        expressiontypes (list): A list of expression types to be considered. If None, defaults to a predefined list.
        T_path (str): The path to the CSV file containing the data. If None, defaults to a specific file in the root directory.
        save_directory (str): The directory where output files will be saved. If None, defaults to the current working directory.
        updated_cluster_map (dict): An optional dictionary to update the cluster mapping.

    Returns:
        str: The path to the CSV file that was read.
    """
    # Check if the inputs are not provided and provide default values It is done this way as several of the inputs are long paths and made the function def really hard to read
    if root is None:
        raise ValueError("root cannot be None. Please provide a valid input.")
    
    if expressiontypes is None:
        expressiontypes = ['Alexa Fluor 488', 'Bax', 'CD4', 'CD8', 'CD20', 'CD14', 'CD68', 'CD206', 'CD11c', \
            'CD21', 'CD3', 'DC-SIGN', 'CD56', 'Granzyme B', 'CD163', 'Foxp3', 'S100A9-Calprotectin', \
            'CD45RA', 'CD45RO', 'CCR7', 'CD31', 'CD45', 'CD69', 'COL1A1', 'HLA-DR-DP-DQ', 'HLA-class-1-A-B-C', \
            'IDO-1', 'Ki67', 'LAG-3', 'MECA-79', 'MelanA', 'PD-1', 'SMA', 'SOX10', 'TCF1TCF7', 'TIM-3', \
            'Tox-Tox2', 'anti-Biotin', 'dsDNA' ]
        expressiontypes = [expressiontypes[i] for i in [2,3,4,21,24,25,38]]
    
    if T_path is None:
        T_path=os.path.join(root,'cleaned_expression_with_both_classification_prob_spatial_30_08_24.csv')
        print(T_path)
    
    if save_directory is None:
        save_directory = os.getcwd
    print("saving Files to :",save_directory)

    T=pd.read_csv(T_path)
    #print(T.head(5))
    if os.path.isfile('clusters.npy') and os.path.isfile('cluster_map.json'):
        clusters=np.load('clusters.npy', allow_pickle=True)
        with open("cluster_map.json", "r") as json_file:
            cluster_map = json.load(json_file)
    else:
        warning_message = "The default map files were not found creating new files based on the spreadsheet provided: This may result in a varation in cell type numbering"
        warnings.warn(warning_message)
        clusters = T['pred'].unique() #Due to varying tables these might change and as such use the established json files to map things
        cluster_map = {cluster: i + 1 for i, cluster in enumerate(clusters)}

        with open("cluster_map.json", "w") as json_file:
            json.dump(cluster_map, json_file)

    
    #Create a dictionary of all the FOV and patient combinations. This can be returned if needed 
    #As we will need to do this anyway  seems easier to do at the start
    fov_to_patient_map = dict(zip(T['fov'], T['patient number']))


    dirlist = os.listdir(root)
   # filtered_FOV = [file for file in os.listdir(root) if file.startswith('FOV')]
    #full_FOV=list(fov_to_patient_map.keys())

    #diff=full_FOV-filtered_FOV

    #This part of the loop or the expression loop could be parrallized
    percentage_of_cores = 0.8  # Set your desired percentage (e.g., 80%)
    max_workers = int(os.cpu_count() * percentage_of_cores)
    missed_directories = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        for dirname in dirlist:
            if os.path.isdir(os.path.join(root, dirname)):
                future = executor.submit(process_directory, root, dirname, expressiontypes, clusters, cluster_map, T, fov_to_patient_map, save_directory)
                futures.append(future)
            else:
                futures.append(None)
                missed_directories.append(dirname)
        
        for future in concurrent.futures.as_completed(futures):
            if future is not None:
                try:
                    pass            
                except Exception as e:
                    print(f"An error occurred: {e}")
                    
    for missed_directory in missed_directories:
        print(f"Error: Directory {missed_directory} does not exist.") 

            
def process_directory(root, dirname, expressiontypes, clusters, cluster_map,T, fov_to_patient_map,save_directory):
    """
    Processes a directory containing image data and associated metadata.

    Args:
        root (str): The root directory containing the data.
        dirname (str): The name of the directory to process.
        expressiontypes (list): A list of expression types to analyze.
        clusters (array-like): An array of unique cluster identifiers.
        cluster_map (dict): A mapping of cluster identifiers to their corresponding values.
        T (DataFrame): A DataFrame containing metadata about the images.
        fov_to_patient_map (dict): A mapping of field of view (FOV) identifiers to patient numbers.
        save_directory (str): The directory where processed data will be saved.

    Returns:
        None: This function does not return a value. It saves processed data to the specified directory.
    """
    segmentation_path = os.path.join(root,dirname,'TIFs', 'segmentation_labels.tiff')
    FOV_table = T[T['fov'] == dirname]
    # Replace with the FOV you want to look up
    patient_number = fov_to_patient_map.get(dirname, 'control')  # Use 'Unknown' or a default value if FOV is not found
    print(patient_number)

    save_patient_dir=os.path.join(save_directory, f'PN{patient_number}')
    if not os.path.exists(save_patient_dir):
        # If it doesn't exist, create the directory
        os.makedirs(save_patient_dir)

    with TiffFile(segmentation_path) as tif:
        segmentation = tif.asarray()
    clustered_seg = segmentation_grouper(segmentation, FOV_table, clusters, cluster_map)
    
    for expression_type in expressiontypes:
        save_path = f"{dirname}_{expression_type}.npz"
        
        tiff_path = os.path.join(root,dirname,'TIFs', f'{expression_type}.tif')
        with TiffFile(tiff_path) as tif:
            image_data = tif.asarray()
        unique_labels = np.unique(image_data)
        save_path = f"{dirname}_PN{patient_number}_{expression_type}.npz"
        save_path=os.path.join(save_patient_dir, save_path)
        print(save_path)
        np.savez(save_path, imageData=image_data, FOV_table=FOV_table.to_records(index=False), clustered_seg=clustered_seg, segmentation=segmentation,)
     

def segmentation_grouper(segmentation, T, clusters, cluster_map):
    """
    Groups the segmentation labels based on the provided clusters and their corresponding mapping.

    This function takes a segmentation array and a DataFrame containing metadata about the clusters. 
    It assigns each label in the segmentation to a cluster based on the mapping provided in the cluster_map.

    Args:
        segmentation (np.ndarray): A 2D array representing the segmentation labels.
        T (DataFrame): A DataFrame containing metadata about the clusters and their labels.
        clusters (array-like): An array of unique cluster identifiers.
        cluster_map (dict): A mapping of cluster identifiers to their corresponding values.

    Returns:
        np.ndarray: A 2D array where each pixel is assigned a cluster identifier based on the segmentation labels.
    """
    clustered_seg = np.zeros(segmentation.shape, dtype=int)
    for cluster in clusters:
        cluster_table = T[T['pred'] == cluster]
        labels = cluster_table['label'].values
        for i, label in enumerate(labels):
            clustered_seg[segmentation == label] = cluster_map[cluster]
    return clustered_seg

def find_files_ending(directory,subscript_search='.npz'):
    """
    Finds all files in the specified directory that end with a given substring.

    Args:
        directory (str): The path to the directory where the search will be performed.
        subscript_search (str): The substring to search for at the end of the file names. 
                                Default is '.npz'.

    Returns:
        list: A list of file paths that match the search criteria.
    """
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(subscript_search):
                file_path = os.path.join(root, file)
                # Load the .npz file
                npz_files.append((file_path))
    return npz_files