import torch
import h5py
from torch.utils.data import Dataset
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt

class MibiDataset(Dataset):
    """
    MibiDataset is a PyTorch Dataset class for handling TIFF images and their corresponding labels.

    This dataset is designed to load images from a specified directory, patch them into smaller squares of configurable sizes,
    and provide the necessary transformations. It also supports binarization of group labels based on a provided DataFrame.

    Args:
        root_dir (str): The root directory containing the FOV directories with TIFF images.
        patch_size_x (int): The height of the patches to be extracted from the images.
        patch_size_y (int): The width of the patches to be extracted from the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        image_paths (list): List of paths to the TIFF images.
        labels (list): List of corresponding labels for the images.
        expressions: Approved lists of protein expressions as .tif files that are shared across desired samples
    """
    def __init__(self, hdf5_path, transform=None,expressions=None):
        self.hdf5_path=hdf5_path
        self.transform=transform
        self.hdf5_expressions=expressions
        with h5py.File(hdf5_path, 'r') as f:
            self.num_samples = f['patches'].shape[0]  

        self.labels = []  
        with h5py.File(hdf5_path, 'r') as f:
            self.labels = f['labels'][:] 
            self.class_counts = {label: 0 for label in set(self.labels)}  
            for label in self.labels:
                self.class_counts[label] += 1 


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, 'r') as f:
            patch = f['patches'][index]  
            label = f['labels'][index]  
            #location = f['locations'][index]  # Load the location (if needed)
        patch= torch.tensor(patch, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            patch = self.transform(patch)
        
        return patch,label
    
class AugmentedGraphDataset(tgd.Dataset):
    def __init__(self, graphs, augments=[]):
        """
        Custom graph dataset that applies a list of augmentations on a list of input graphs.
        
        Args:
            graphs (list): List of PyG Data objects representing individual graphs.
            augments (list): List of augmentation functions to apply.
        """
        super(AugmentedGraphDataset).__init__()
        self.graphs = graphs  # Store the list of graphs
        self.augments = augments

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        # Retrieve the graph at the specified index
        data = self.graphs[idx].clone()  # Clone to prevent modifying the original graph

        # Apply each augmentation in the list
        for augment in self.augments:
            data = augment(data)
        
        return data