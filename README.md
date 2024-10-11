# Mibi-Analysis-py
This code is designed to enable the processing of mibi-tof data and expression information.

# Project README

## Overview
This project is a Python-based implementation of a UNet model for image segmentation of Mibi-ToF imaging using PyTorch and Albumentations. The code includes training parameters, data loading, model definition, and training loop.

## Setup
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Ensure you have the necessary dataset stored in the specified directory.
3. Modify the parameters in `train_unet_albumations.py` according to your dataset and requirements.

## Training
To train the UNet model, run the following command:
```bash
python train_unet_albumations.py
This script can be modified to add additional protein images but by default it looks for CD4 expression alone.

```

## Root Folder
- `train_unet_albumations.py`: Main script for training the UNet model.
- `MibiProcessor.py`: Description of MibiProcessor module goes here.
- default location that cluster maps are stored to syncronize cell labels

### Modules
1. **ImageProcessing Folder**:
- `make_hdf5.py`: Script for building the pytables for the segmentation model.
Here is a sample running code snippet for the [make_hdf5.py](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/ImageProcessing/make_hdf5.py#1%2C1-1%2C1) script:

```python
from ImageProcessing.make_hdf5 import make_hdf5

# Define the parameters for creating the HDF5 file
dataname = "image_data"
patch_size = 64
stride_size = 64
mirror_pad_size = 16
test_set_size = 0.1
resize = 1
classes = np.arange(1)
data_full_path = "path/to/data_directory"
tables_base_path = "path/to/HDF5_files"
class_to_keep = [1, 2, 3]  # Define the classes to keep

# Call the make_hdf5 function to create the HDF5 file
make_hdf5(dataname=dataname, patch_size=patch_size, stride_size=stride_size, mirror_pad_size=mirror_pad_size,
          test_set_size=test_set_size, resize=resize, classes=classes, data_full_path=data_full_path,
          tables_base_path=tables_base_path, class_to_keep=class_to_keep)
```

This code snippet sets up the necessary parameters and calls the `make_hdf5` function to create an HDF5 file for storing image patches and their corresponding masks for training and validation sets. Make sure to replace the placeholder paths and parameters with the actual values specific to your project.
- `Mibi_img_Dataset.py`: Modified Dataset class for the Segmentation. In future revisions will handle conversion to graph form as a transform stage.

- 

The [mibi_helper](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/MibiProcessor.py#5%2C6-5%2C6) Folder provides essential functionality for loading and processing MIBI data in the project. Here is an overview of the key components within the package:


2. **mibi_loader.py**:
   - This module contains functions for loading MIBI data, processing it, and generating npz files for further analysis.
   - It includes functions like [mibi_loader](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/MibiProcessor.py#5%2C25-5%2C25), [segmentation_grouper](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/mibi_helper/__init__.py#6%2C39-6%2C39), [find_files_ending](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/MibiProcessor.py#5%2C73-5%2C73), and [process_directory](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/mibi_helper/mibi_loader.py#100%2C5-100%2C5).
   - The [mibi_loader](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/MibiProcessor.py#5%2C25-5%2C25) function loads MIBI data, processes it based on specified parameters, and generates npz files with expression images, segmentations, and clustered segmentations.

3. **mibi_eroder.py**:
   - This module provides functions for eroding segmentations in parallel or sequentially.
   - Functions like [mibi_eroder](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/MibiProcessor.py#5%2C60-5%2C60) and [mibi_eroder_parallel](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/MibiProcessor.py#5%2C38-5%2C38) are included for erosion operations on segmentations.

### How to Run:
To utilize the functionalities provided by the [mibi_helper](file:///d%3A/MIBI-TOFF/scripts/Mibi-Analysis-py/MibiProcessor.py#5%2C6-5%2C6) package, follow these steps:
1. Import the necessary functions from the package in your script:
   ```python
   from mibi_helper import mibi_loader, mibi_eroder_parallel, mibi_eroder, find_files_ending
   ```
2. Use the functions as needed in your script. For example, to load MIBI data and generate npz files:
   ```python
   mibi_loader(root='path_to_data_directory', expressiontypes=None, T_path=None, save_directory='output_directory')
   ```

### Example Usage:
Here is an example of how you can use the `mibi_loader` function in your script:
```python
from mibi_helper import mibi_loader

# Load MIBI data and generate npz files
mibi_loader(root='path_to_data_directory', expressiontypes=None, T_path=None, save_directory='output_directory')
```

By following these steps and utilizing the functions provided in the `mibi_helper` package, you can efficiently load and process MIBI data for further analysis in your project.

## Acknowledgements
- The UNet architecture is borrowed from [jvanvugt's GitHub repository](https://github.com/jvanvugt/pytorch-unet).
- Various libraries and tools used in the project are credited in the respective license files.

## Contact
For any questions or issues, feel free to reach out to the project maintainers.

---
For more details, refer to the specific code blocks provided in the project files.
