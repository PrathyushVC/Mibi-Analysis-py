import os
import numpy as np
import pandas as pd
from tifffile import TiffFile
from MibiHelper import MibiLoader
from MibiHelper import MibiEroder
import matplotlib.pyplot as plt

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
directory = save_directory  

# List all files in the directory
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








