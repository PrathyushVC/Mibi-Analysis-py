import os,cv2
import numpy as np
import pandas as pd
from tifffile import TiffFile
from MibiHelper import MibiLoader
from MibiHelper import MibiEroder
from GraphGenerator import ProximityFinder as proxF
import matplotlib.pyplot as plt
import networkx as nx
from skimage.measure import label, regionprops
# Initialize Ray and connect to the cluster



#If you dont have the expression wise numpys generate them
save_directory=r'D:\MIBI-TOFF\Data'
MibiLoader(root=r'D:\MIBI-TOFF\Data_For_Amos', expressiontypes=None, T_path=None,save_directory=r'D:\MIBI-TOFF\Data_Full')


'''
#data_catch=np.load(r'D:\MIBI-TOFF\Data\FOV1_G3_CD4.npz')
#print(data_catch.files)
plt.figure()dir\
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
'''
directory = save_directory  

# List all files in the directory
file_list = os.listdir(directory)

# Filter files that end with "_CD4.npz"
#This is for testing in the future I think these individual files wont have the these redundent info in them and instead a single seg file with be made per patient
'''

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
data_catch=np.load("D:\MIBI-TOFF\Data_Full\PN1\FOV2_PN1_CD4.npz",allow_pickle=True)
segmentation=data_catch['segmentation']
remove_non_cells=data_catch['clustered_seg']

segmentation=segmentation[0:64,0:64]
remove_non_cells=remove_non_cells[0:64,0:64]

print(np.shape(segmentation))
#This is to test how we can easily remove types that are not useful or to reduce the space down to a single cell type
values_to_set_to_zero=[1,7,8,9]
# Create a boolean mask for the values to be set to zero
mask = np.isin(remove_non_cells, values_to_set_to_zero)

# Set the values to zero using the mask
remove_non_cells[mask] = 0
remove_non_cells[remove_non_cells>0]=1





#The bellow line would have relabled the image but not needed due to the nature of our image
#labeled_segments = label(segmentation)
print(list(data_catch.keys()))
print(data_catch['FOV_table'])
FOV_table =pd.DataFrame.from_records( data_catch['FOV_table'])
print(FOV_table.head())


centroids,regions=proxF.centroid_compute(segmentation=segmentation,region_mask=remove_non_cells)

segmentation=segmentation*remove_non_cells


# Define the maximum distance for nodes to be connected (100 pixels in this example)
max_distance = 25
# Create a graph
G = nx.Graph()
# Extract centroids into a NumPy array
distances=proxF.dist_comp(centroids, num_chunks=1)
'''


# Compute pairwise Euclidean distances between centroids
distances = np.linalg.norm(centroids[:, None] - centroids, axis=2)
'''
# Set the maximum distance threshold
max_distance = 25

# Create a mapping from node indices to regions
node_to_region = {}

G_test = nx.Graph()


node_to_region_test = {}

# Add nodes to the graph
for i, region in enumerate(regionprops(segmentation)):
    node_to_region_test[i] = region

    G_test.add_node(i, cell_type='Type A', area=region.area, centroid=region.centroid, segment_label = segmentation[int(region.centroid[0]), int(region.centroid[1])])


row_indices, col_indices = np.where(distances >= max_distance)
index_pairs = list(zip(row_indices, col_indices))
G_test.add_edges_from(index_pairs)

plt.figure()
pos = nx.spring_layout(G)  # Layout for visualization
nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10)
#plt.show()
#Verify that the distances are being computed correctly
#DF = pd.DataFrame(distances)
#DF.to_csv("distance.csv")

#DF = pd.DataFrame(centroids)
#DF.to_csv("centroids.csv")
#print(centroids)








