# handles reading  HDF5/NIfTI files, stacking the 4 MRI modalities, and converting them into PyTorch Tensors.

import torch
from torch.utils.data import Dataset, DataLoader
import h5py # Assuming you are using HDF5 slices as mentioned in your proposal
import numpy as np

class BraTSDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths (list): List of paths to your 2D slice files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        # TODO: Return the total number of slices
        pass

    def __getitem__(self, idx):
        # TODO: 1. Open the HDF5/NIfTI file at self.file_paths[idx]
        # TODO: 2. Extract the 4 MRI modalities (T1, T1CE, T2, FLAIR)
        # TODO: 3. Stack them into a numpy array of shape (4, H, W)
        # TODO: 4. Extract the ground truth segmentation mask of shape (1, H, W) or (Classes, H, W)
        
        # Mock data (Replace this with your actual loading logic)
        image = np.zeros((4, 240, 240), dtype=np.float32) 
        mask = np.zeros((1, 240, 240), dtype=np.float32)

        if self.transform:
            # TODO: Apply augmentations (random flip, rotation, etc.)
            pass

        # Convert to PyTorch tensors
        image_tensor = torch.tensor(image)
        mask_tensor = torch.tensor(mask)

        return image_tensor, mask_tensor

# TODO: Create a function `get_dataloaders(train_paths, val_paths, batch_size)` 
# that returns train and validation DataLoaders.