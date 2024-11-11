
import tifffile as tiff
import h5py
from pathlib import Path
import numpy as np
import polars as pl

#TODO switch to logging after initial tests

def dataset_overlap(df1,df2,col):
  #set up for polars dataframes   
    set1 = set(df1[col].to_list())
    set2 = set(df2[col].to_list())

    if set1 & set2:
        print("There is an overlap in patient numbers.")
    else:
        print("No overlap in patient numbers.")

def fovs_to_hdf5(root_dir,df,hdf5_path, patch_size_x=128, patch_size_y=128,prefix='FOV',fov_col='FOV',label_col='Group', expressions=None):
    image_paths = []
    labels=[]
    root_path = Path(root_dir)
    
    #Iterate over each FOV directory
    for fov_path in root_path.glob(f"{prefix}*"):
        if not fov_path.is_dir():
            print(f"{fov_path} is not dir")
            continue  # Skip if not a directory

        group = df.filter(pl.col(fov_col) == fov_path.name)
        if group.height == 0:
            print(f"{fov_path} is not in provided Dataframe")
            continue  # Skip if FOV is not in the DataFrame

        group_data = group[label_col].to_list()[0]
        binarized_data = _binarize_group(group_data)
        tif_path = fov_path / 'TIFs'
        
        if not tif_path.exists():
            print(f"{fov_path} is missing a TIF folder")
            continue  # Skip if 'TIFs' directory doesn't exist

        sublist = []
        for image_file in tif_path.iterdir():
            if expressions and image_file.name not in expressions:
                #print(f"{fov_path}\{image_file} is not in approved expressions")
                continue  # skip if file is not in the expressions list
            
            if image_file.suffix in ['.tif', '.tiff'] and \
               'segmentation' not in image_file.name and \
               not image_file.name[0].isdigit():
                sublist.append(str(image_file))

        if sublist:#If all files share the same set of paths this will have a normalized order. 
            sublist.sort()  # normalize the order of images
            labels.append(binarized_data)
            image_paths.append(sublist)

    _gen_patches_hdf5(image_paths,labels,hdf5_path=hdf5_path,patch_size_x=patch_size_x,patch_size_y=patch_size_y)


    return image_paths,labels

def _gen_patches_hdf5(image_paths_list,labels,hdf5_path,patch_size_x,patch_size_y,max_patches=None):
    
    
    if max_patches is None:
        patch_count=[]
        max_patches=0
        for sublist in image_paths_list:
            image = tiff.imread(sublist[0])
            tx,ty= image.shape
            num_patches_x = tx // patch_size_x
            num_patches_y = ty // patch_size_y
            total_patches = num_patches_x * num_patches_y
            patch_count.append(num_patches_x * num_patches_y)# for debugging
            max_patches+=total_patches
            #This assumes clean ratios of the images and the patch size

    print(max_patches)



    with h5py.File(hdf5_path,'w') as hdf5_file:
        patch_dataset = hdf5_file.create_dataset('patches', 
                                                 shape=(max_patches, len(image_paths_list[0]), patch_size_x, patch_size_y),
                                                 dtype=np.float32)
        
        label_dataset = hdf5_file.create_dataset('labels', shape=(max_patches,), dtype=int)
        
        # FOV and patch locations
        dt = np.dtype([('image_id', h5py.string_dtype(encoding='utf-8')),  # String field
                       ('coords', np.int32, (2,))])  # Tuple of 2 integers
        loc_dataset = hdf5_file.create_dataset('locations', shape=(max_patches,), dtype=dt)
        print("Keys in the HDF5 file:")
        print(list(hdf5_file.keys()))
    start_index=0
    print(hdf5_path)

    for image_idx, image_paths in enumerate(image_paths_list):
        image = _image_stacker(image_paths=image_paths)  # Stack images to get the final image
        label = labels[image_idx]  # Label for this image
        start_index=_store_patch_in_hdf5(hdf5_path=hdf5_path, image_name=image_paths,
                                       image=image,label=label,patch_size_x=patch_size_x,
                                       patch_size_y=patch_size_y, start_index=start_index)
        print(start_index)

    return hdf5_path

def _store_patch_in_hdf5(hdf5_path, image_name,image,label,patch_size_x,patch_size_y, start_index):
    with h5py.File(hdf5_path,'a') as hdf5:
        print("Keys in the HDF5 file:")
        print(list(hdf5.keys()))
        for i in range(0, image.shape[1], patch_size_x):
            for j in range(0, image.shape[2], patch_size_y):
                hdf5['patches'][start_index] = image[:, i:i+patch_size_x, j:j+patch_size_y]
                hdf5['labels'][start_index] =label
                hdf5['locations'][start_index]=(f"{image_name[0]}", (i, j)) 
                start_index+=1
                
    return start_index


def _binarize_group(group_data):
    #Assumes a desired class structure 
    if str(group_data) in ['G1', 'G4']:
        return 1 
    elif str(group_data) in ['G2', 'G3']:
        return 0  
    else:
        raise Exception("Group Data does not align with Expected Groups")  
    
def _image_stacker(image_paths):

    stacked_image = None
    for path in image_paths:
        image = tiff.imread(path) 
        if stacked_image is None:
            stacked_image = np.expand_dims(image,axis=0)  
        else:
            stacked_image = np.concatenate((stacked_image, np.expand_dims(image,axis=0)), axis=0)  

    return stacked_image
