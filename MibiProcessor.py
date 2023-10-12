import os
import numpy as np
import pandas as pd
from tifffile import TiffFile
from MibiHelper import MibiLoader
from MibiHelper import MibiEroder
import matplotlib.pyplot as plt
import networkx as nx
from skimage.measure import label, regionprops

#If you dont have the expression wise numpys generate them
save_directory=r'D:\MIBI-TOFF\Data'
#MibiLoader(root=None, expressiontypes=None, grps=None, T_path=None,save_directory=r'D:\MIBI-TOFF\Data')


'''
#data_catch=np.load(r'D:\MIBI-TOFF\Data\FOV1_G3_CD4.npz')
#print(data_catch.files)
plt.figure()
im = plt.imshow(data_catch['imageData'], 'gray')
plt.title('ImageData')


plt.figure()
segshow = plt.imshow(data_catch['segmentation'], 'gray')
plt.title('Segmentation')

plt.figure()
clusterdseg=plt.imshow(data_catch['clustered_seg'], 'gray')
plt.title('Clustered_Seg')
plt.show()

'''
directory = save_directory  

# List all files in the directory
file_list = os.listdir(directory)

# Filter files that end with "_CD4.npz"
#This is for testing in the future I think these individual files wont have the these redundent info in them and instead a single seg file with be made per patient


'''
filtered_files = [file for file in file_list if file.endswith("_CD4.npz")]


for file in filtered_files:
    print(file)

    load_path=os.path.join(save_directory, file)
    data_catch=np.load(load_path,allow_pickle=True)
    
    segmentation=data_catch['segmentation']

    erroded_mask=MibiEroder(segmentation=segmentation)
    save_directory=r'D:\MIBI-TOFF\Data'

    #This assumes the filed are named FOVX_GX and will get mad if its anything else
    parts = file.split("_")
    desired_part = parts[0] + "_" + parts[1]

    save_path = f"{desired_part}_{'segmentations'}.npz"
    save_path=os.path.join(save_directory, save_path)
    check=data_catch['FOV_table']
    print(check)
    np.savez(save_path,
             erroded_mask=erroded_mask,
             FOV_table=data_catch['FOV_table'],
             clustered_seg=data_catch['clustered_seg'],
             segmentation=segmentation)

'''


# Errosion Loops

'''
directory = save_directory  
file_list = os.listdir(directory)

filtered_files = [file for file in file_list if file.endswith("_segmentations.npz")]
for file in filtered_files:
    print(file)

    load_path=os.path.join(save_directory, file)
    data_catch=np.load(load_path,allow_pickle=True)
    erroded_mask=data_catch['erroded_mask']

    parts = file.split("_")
    desired_part = parts[0] + "_" + parts[1]

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Create the first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(data_catch['erroded_mask'], 'gray')
    plt.title(desired_part+': erroded_mask')

    # Create the second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(data_catch['segmentation'], 'gray')
    plt.title(desired_part+': segmentation')
    plt.tight_layout()

    plt.savefig(desired_part+'erroded_fig.png')  # Change the file format as needed (e.g., .jpg, .pdf, .svg)

'''
#This is a test for conversion to a graph
data_catch=np.load('D:\MIBI-TOFF\Data\FOV1_G4_segmentations.npz',allow_pickle=True)
segmentation=data_catch['segmentation']

segmentation = np.array([[1, 1, 0, 2, 2],
                         [0, 1, 0, 2, 0],
                         [3, 0, 0, 2, 2],
                         [3, 0, 4, 4, 4],
                         [1, 1, 0, 2, 2],
                         [1, 1, 0, 2, 2]])

labeled_segments = label(segmentation)

# Define the maximum distance for nodes to be connected (100 pixels in this example)
max_distance = 3

# Create a graph
G = nx.Graph()

# Iterate through the labeled segments to create nodes
for region in regionprops(labeled_segments):
    segment_label = region.label
    
    # Encode additional information into the node attributes
    node_attributes = {
        'cell_type': region.label,  # Replace with the actual cell type
        'area': region.area,     # Area of the cell
        'centroid': region.centroid  # Centroid coordinates
    }
    
    G.add_node(segment_label, **node_attributes)

# Connect nodes if their corresponding regions are within the maximum distance
centroids = {}
for region in regionprops(labeled_segments):
    segment_label = region.label
    centroids[segment_label] = region.centroid

    # Store the original segment value as an attribute
    node_attributes = {
        'cell_type': 'Type A',  # Replace with the actual cell type
        'area': region.area,     # Area of the cell
        'centroid': region.centroid,  # Centroid coordinates
        'original_segment_value': segment_label  # Store the original segment value
    }

    G.add_node(segment_label, **node_attributes)

# Connect nodes if their corresponding regions are within the maximum distance
for region in regionprops(labeled_segments):
    segment_label = region.label
    for neighbor_label, neighbor_centroid in centroids.items():
        if neighbor_label != segment_label:
            distance = np.sqrt((region.centroid[0] - neighbor_centroid[0])**2 + (region.centroid[1] - neighbor_centroid[1])**2)
            if distance <= max_distance:
                G.add_edge(segment_label, neighbor_label)


# Visualize the graph
pos = nx.spring_layout(G)  # Layout for visualization
nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10)
plt.show()

node_label = 5
node_attributes = G.nodes[node_label]

# Print or display the attributes
print("Attributes of Node {}: {}".format(node_label, node_attributes))













