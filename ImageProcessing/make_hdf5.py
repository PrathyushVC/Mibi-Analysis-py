import os,sys
import numpy as np
import h5py
import logging
import tables
import glob
import numbers
import cv2
from numpy.lib.stride_tricks import as_strided  # as_strided allows for the creation of a view into an array with a different shape and strides configuration without copying the data.
from sklearn import model_selection
import random
import PIL
import matplotlib.pyplot as plt


def extract_patches(arr,patch_shape=32,stride=32):
    if isinstance(patch_shape,numbers.Number):
        patch_shape=(patch_shape,) * arr.ndim
    if isinstance(stride,numbers.Number):
        extraction_step = (stride,) * arr.ndim  # Changed to create a tuple with 'stride' repeated 'arr.ndim' times
    
    patch_strides=arr.strides

    slices=tuple(slice(None,None,step) for step in extraction_step)
    indexing_strides=arr[slices].strides

    patch_indices_shape=( (np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)) + 1
    shape=tuple(list(patch_indices_shape) + list(patch_shape))
    strides=tuple(list(indexing_strides) + list(patch_strides))
    patches=as_strided(arr,shape=shape,strides=strides)
    return patches




def make_hdf5(dataname,patch_size=64,stride_size=64,mirror_pad_size=16,test_set_size=0.1,resize=1,classes=np.arange(1),data_full_path=None,tables_base_path=None):
    if data_full_path is None:
        raise ValueError("data_full_path must be provided")
    if tables_base_path is None:
        raise ValueError("table_full_path must be provided")
    
    logging.info(f"Patch size: {patch_size}, Stride size: {stride_size}, Mirror pad size: {mirror_pad_size}, Test set size: {test_set_size}, Resize: {resize}, Classes: {classes}")
    seed=1995
    random.seed(1995)
    img_dtype=tables.Float32Atom()
    filenameAtom= tables.StringAtom(itemsize=255)
    
    logging.info(f"Constructed file path: {data_full_path}")
    files = glob.glob(data_full_path, recursive=True)
    
    logging.info(f"Number of files: {len(files)}")
    patients=[os.path.basename(file).split('_')[1] for file in files]
    gss=model_selection.GroupShuffleSplit(n_splits=1,test_size=test_set_size)
    train_idx,val_idx=next(gss.split(files,groups=patients))
    phases = {}
    phases["train"] = [files[i] for i in train_idx]
    phases["val"] = [files[i] for i in val_idx]
    logging.info(f"Training set size: {len(phases['train'])}, Validation set size: {len(phases['val'])}")
    logging.debug(f"Training set: {phases['train']}")
    logging.debug(f"Validation set: {phases['val']}")


    imgtypes=['img','mask']

    storage={}
    block_shape={}
    block_shape["img"]=np.array((patch_size,patch_size,3))
    block_shape["mask"]=np.array((patch_size,patch_size))
    filters=tables.Filters(complevel=6,complib='zlib')
    
    for phase in phases.keys():
        logging.info(f"Processing phase: {phase}")
        totals=np.zeros((len(phases[phase]),len(classes)))
        totals[0,:]=classes

        with tables.open_file(os.path.join(tables_base_path,f"{dataname}_{phase}.pytable"), mode='w') as hdf5_file: #open the respective pytable
            storage["filename"] = hdf5_file.create_earray(hdf5_file.root, 'filename', filenameAtom, (0,)) #create the array for storage
        
            for imgtype in imgtypes: #for each of the image types, in this case mask and image, we need to create the associated earray
                storage[imgtype]= hdf5_file.create_earray(hdf5_file.root, imgtype, img_dtype,  
                                                        shape=np.append([0],block_shape[imgtype]), 
                                                        chunkshape=np.append([1],block_shape[imgtype]),
                                                        filters=filters)
            
            for filei in phases[phase]: #now for each of the files
                fname=filei 
                
                logging.debug(f"Processing file: {fname}")
                try:
                    Img_data = np.load(fname) 
                    logging.info(f"Loaded file: {fname} with headers: {Img_data.keys()}")

                except FileNotFoundError:
                    logging.error(f"File not found: {fname}")
                    break
                except Exception as e:
                    logging.error(f"Error processing file: {fname}")
                    logging.error(f"Error: {e}")
                    break#Just breaking to make it easy to debug
                for imgtype in imgtypes:#This is basically a switch statement to handle the two image types.
                    
                    if(imgtype=="img"): #if we're looking at an img, it must be 3 channel, but cv2 won't load it in the correct channel order, so we need to fix that
                        io=Img_data['imageData']
                        io = np.repeat(io[:, :, np.newaxis], 3, axis=2)
                        interp_method=PIL.Image.BICUBIC
                    else:    
                        io=Img_data['clustered_seg']  
                        io = np.repeat(io[:, :, np.newaxis], 3, axis=2)#This is only to handle the fact that the template code assumes the image has been read in by cv2.read resulting in 3d.
                        #expanding and squeezing achieves the same dimensions without needing custom cases
                        interp_method=PIL.Image.NEAREST

                        for i,key in enumerate(classes): #sum the number of pixels, this is done pre-resize, the but proportions don't change which is really what we're after
                            totals[1,i]+=sum(sum(io[:,:,0]==key))


                    io = cv2.resize(io,(0,0),fx=resize,fy=resize, interpolation=interp_method) #resize it as specified above
                    io = np.pad(io, [(mirror_pad_size, mirror_pad_size), (mirror_pad_size, mirror_pad_size),(0,0)], mode="reflect")
                    
                        
                    #convert input image into overlapping tiles, size is ntiler x ntilec x 1 x patch_size x patch_size x3
                    
                    io_arr_out=extract_patches(io,(patch_size,patch_size,3),stride_size)
                    
                    io_arr_out=io_arr_out.reshape(-1,patch_size,patch_size,3)



                    if(imgtype=="img"):
                        storage[imgtype].append(io_arr_out)
                    else:
                        storage[imgtype].append(io_arr_out[:,:,:,0].squeeze()) #only need 1 channel for mask data

                storage["filename"].append([fname for x in range(io_arr_out.shape[0])]) #add the filename to the storage array
                
            npixels=hdf5_file.create_carray(hdf5_file.root, 'numpixels', tables.Atom.from_dtype(totals.dtype), totals.shape)
            npixels[:]=totals
            hdf5_file.close()  

def binarize_segmentation(mask,class_to_keep=[]):
    if class_to_keep is None:
        raise Exception('class to keep must have at least one real integer value')
    


def show_image(image_data,title='Image'):
    plt.imshow(image_data, cmap='gray', vmin=0, vmax=25)
    print(np.unique(image_data))
    plt.title(title)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def print_hdf5_file_data(file_path):
    with tables.open_file(file_path, mode='r') as hdf:
        # Print the structure of the HDF5 file
        def print_structure(node, prefix=''):
            print(f"{prefix}{node._v_name} ({type(node).__name__})")
            if isinstance(node, tables.Group):
                for child in node._v_children.values():
                    print_structure(child, prefix + '  ')
        print_structure(hdf.root)

def lazy_load_imgs(hdf):
    for img in hdf.root.img:
        yield img

def lazy_load_mask(hdf):
    for mask in hdf.root.mask:
        yield mask

def test_samp_imgs(file_path,print_ximages=1000):
    with tables.open_file(file_path, mode='r') as hdf:
        images_generator = lazy_load_imgs(hdf)
        try:
            for i,image in enumerate(images_generator):
                if i%print_ximages==0:
                    first_image = next(images_generator)
                    show_image(first_image, title='f{i}')
        except StopIteration:
            logging.info("no more images")
        mask_gen = lazy_load_mask(hdf)
        try:
            for i,mask in enumerate(mask_gen):
                if i%print_ximages==0:
                    first_image = next(mask_gen)
                    show_image(first_image, title='f{i}')
        except StopIteration:
            logging.info("no more images")


        
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    current_dir = os.path.dirname(__file__)
    print_ximages=1000
    #os.path.join(current_dir, r'..\..\..\Data_Full\**\*_CD4.npz')
    #os.path.join(current_dir, r'..\..\..\Scratch')
    #6,7,13 we are actually only interested in these 3 classes and everything else can be labeled background for this test
    #Setting it up like this makes it easy to sanitize later
    make_hdf5(dataname='Mibi_trial',data_full_path=os.path.join(current_dir, r'..\..\..\Data_Full\**\*_CD4.npz'),tables_base_path=os.path.join(current_dir, r'..\..\..\Scratch'))              
    #file_path='D:\MIBI-TOFF\Scratch\Mibi_trial_train.pytable'
    #print_hdf5_file_data(file_path)
    #test_samp_imgs(file_path,print_ximages=1000)

    



