import os,cv2
import numpy as np
import pandas as pd
from tifffile import TiffFile
from mibi_helper import mibi_loader, mibi_eroder_parallel, mibi_eroder, find_files_ending
from graph_helper import graph_maker
import matplotlib.pyplot as plt
import networkx as nx
from skimage.measure import label, regionprops

# Initialize Ray and connect to the cluster
#r'C:\Users\chirr\OneDrive - Case Western Reserve University\Research\MIBI-TOFF\Data\weizmann\lymph_node_metastasis\Original_Data'
#T_old=r'C:\Users\chirr\Downloads\weizmann\lymph_node_metastasis\Original_Data\Cell_Table_Marker_150622.csv

#If you dont have the expression wise numpys generate them
save_directory=r'D:\MIBI-TOFF\Data_Full'
mibi_loader(root=r'D:\MIBI-TOFF\Data_For_Amos', expressiontypes=None, T_path=None,save_directory=r'D:\MIBI-TOFF\Data_Full')




directory = r'D:\MIBI-TOFF\Data_Full\PN1'

# List all files in the directory
file_list = os.listdir(directory)
filtered_files = [file for file in file_list if file.endswith("_CD4.npz")]


for file in filtered_files:
    print(file)

    load_path=os.path.join(save_directory, file)
    data_catch=np.load(load_path,allow_pickle=True)
    
    segmentation=data_catch['segmentation']

    erroded_mask=mibi_eroder_parallel(segmentation=segmentation)
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
''''''
directory = save_directory  

filterd_files=find_files_ending(directory,subscript_search='_CD4.npz')
for file in filterd_files:
    data_catch=np.load(file,allow_pickle=True)
    segmentation=data_catch['segmentation']
    remove_non_cells=data_catch['clustered_seg']
    segmentation=segmentation
    remove_non_cells=remove_non_cells
    print(np.shape(segmentation))
#This is to test how we can easily remove types that are not useful or to reduce the space down to a single cell type
    values_to_set_to_zero=[2,3,4,12]#These values are depended on the the cluster_map json always remove unidentified
# Create a boolean mask for the values to be set to zero
    mask = np.isin(remove_non_cells, values_to_set_to_zero)
# Set the values to zero using the mask
    remove_non_cells[mask] = 0
    remove_non_cells[remove_non_cells>0]=1
    segmentation=segmentation*remove_non_cells

#The bellow line would have relabled the image but not needed due to the nature of our image
#labeled_segments = label(segmentation)
    print(list(data_catch.keys()))
    FOV_table =pd.DataFrame.from_records( data_catch['FOV_table'])
# Define the maximum distance for nodes to be connected (50 used for images with 800 microns as the default as this is approximatly 20um)
    max_distance = 50
    G_test=graph_maker.seg_to_graph(segmentation=segmentation,max_distance=max_distance,FOV_table=FOV_table,num_chunks=1)
# Add nodes to the graph

    print(G_test)
    G_adj=nx.to_numpy_array(G_test)
    print(G_adj)
#plt.figure()
#plt.imshow(G_adj)
#plt.show()
    
'''
Generate test figure
plt.figure()
pos = nx.spring_layout(G_test)  # Layout for visualization
nx.draw(G_test, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10)
plt.show()
'''
#Verify that the distances are being computed correctly
#DF = pd.DataFrame(distances)
#DF.to_csv("distance.csv")

#DF = pd.DataFrame(centroids)
#DF.to_csv("centroids.csv")
#print(centroids)








