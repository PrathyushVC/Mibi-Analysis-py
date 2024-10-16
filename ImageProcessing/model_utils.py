import torch
import tifffile as tiff
import h5py
import os
import polars as pl
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5,check_val_freq=5):
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels,_,fov_paths in train_loader:
            labels=labels.to(device)
            print(labels)
            print(fov_paths[0])
            
            for patches in inputs:
               
                patch = patches.to(device).float()
                optimizer.zero_grad()
                outputs = model(patch)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Compute accuracy for training set
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)*len(inputs)#Lables is replicated
                correct += (predicted == labels).sum().item()
            print(f"Correct:{100*correct/total}|| last output:{outputs}|| running_loss:{running_loss}")
        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)#Adjust for length of loader
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        val_loss, val_accuracy = eval_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


        # Evaluate on validation set
        if (epoch%check_val_freq==0) and epoch>0:
            print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}")

        print(f'Epoch {epoch}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():

        for inputs, labels,_,fov_paths in dataloader:
            labels=labels.to(device)
            print(labels)
            print(fov_paths[0])
            for patches in inputs:
                
                patch = patches.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)#Needs to be adjusted for the number of patches
        accuracy = 100 * correct / total
        print(f"avg_loss{avg_loss}|| Loss:{loss}")
        return avg_loss, accuracy

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
    
    #Iterate over each FOV directory
    #five if checks to achieve this is hard to parse need to rethink approac
    for fov_dir in os.listdir(root_dir):
        if fov_dir.startswith(prefix):#replace with Pathlib matching later
            fov_path = os.path.join(root_dir, fov_dir)
            
            if os.path.isdir(fov_path):
                #print(fov_dir)
                # check if the folder name exists in the "FOV" column of the DataFrame
                group = df.filter(pl.col(fov_col) == fov_dir)  #why oh why did I decide to decide to use polars instead of pandas for this test?
                if group.height>0:
                    # pull the matching data from the "group" column
                    
                    group_data = group[label_col].to_list()[0]  
                    binarized_data = _binarize_group(group_data)  # Binarize the group data
                    #print(group_data,binarized_data)
                    tif_path = os.path.join(fov_path, 'TIFs')
                    if os.path.exists(tif_path):
                        # Load all .tiff files ignoring those with "segmentation in the name"
                        sublist=[]
                        labels.append(binarized_data)
                        for image_file in os.listdir(tif_path):
                            if expressions:# this is a list of expression names we want to make sure are in the dataset. 
                                if image_file in expressions:
                                    sublist.append(os.path.join(tif_path, image_file))
                            else:
                                if (image_file.endswith('.tif') or image_file.endswith('.tiff')) and ('segmentation' not in image_file) and not (image_file[0].isdigit()):
                                    sublist.append(os.path.join(tif_path, image_file))
                        

                        sublist.sort()#Should normalize the data assuming the data format is maintained true for this dataset 
                        #should create a perminant mapping
                        image_paths.append(sublist)


                    #else:
                        #print(f"Skipped {fov_dir} because TIF does not exist")
                #else:
                    #print(f"Skipped {fov_dir} because its not in the table of labels")
            #else:
                #print(f"Skipped {fov_dir} because its not in the root_dir")

    _gen_patches_hdf5(image_paths,labels,hdf5_path=hdf5_path,patch_size_x=patch_size_x,patch_size_y=patch_size_y)


    return image_paths,labels

def _gen_patches_hdf5(image_paths_list,labels,hdf5_path,patch_size_x,patch_size_y):

    with h5py.File(hdf5_path,'w') as hdf5_file:
        patch_dataset = hdf5_file.create_dataset('patches', 
                                                 shape=(0, len(image_paths_list[0]), patch_size_x, patch_size_y),
                                                 maxshape=(None, len(image_paths_list[0]), patch_size_x, patch_size_y),
                                                 dtype=np.float32)
        
        label_dataset = hdf5_file.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=int)
        
        # FOV and patch locations
        dt = np.dtype([('image_id', h5py.string_dtype(encoding='utf-8')),  # String field
                       ('coords', np.int32, (2,))])  # Tuple of 2 integers
        loc_dataset = hdf5_file.create_dataset('locations', shape=(0,), maxshape=(None,), dtype=dt)
        patch_idx=0
        for image_idx, image_paths in enumerate(image_paths_list):
            image = _image_stacker(image_paths=image_paths)  # Stack images to get the final image
            label = labels[image_idx]  # Label for this image
            
            num_patches_x = image.shape[1] // patch_size_x
            num_patches_y = image.shape[2] // patch_size_y
            num_new_patches = num_patches_x * num_patches_y
            
            # Resize the datasets 
            patch_dataset.resize(patch_idx + num_new_patches, axis=0)
            label_dataset.resize(patch_idx + num_new_patches, axis=0)
            loc_dataset.resize(patch_idx + num_new_patches, axis=0)
            
            #make this its own function
            for i in range(0, image.shape[1], patch_size_x):
                for j in range(0, image.shape[2], patch_size_y):
                    patch = image[:, i:i+patch_size_x, j:j+patch_size_y]
                    patch_dataset[patch_idx] = patch  
                    label_dataset[patch_idx] = label  
                    
                    # compound datatype (string, tuple of integers)
                    loc_info = (f"{image_paths[0]}", (i, j)) 
                    loc_dataset[patch_idx] = loc_info
                    
                    patch_idx += 1

    return hdf5_path

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
