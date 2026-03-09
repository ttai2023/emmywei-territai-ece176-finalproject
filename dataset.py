# handles reading  HDF5 slide files, stacking the 4 MRI modalities, and converting them into PyTorch Tensors

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
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
        return len(self.file_paths)

    def __getitem__(self, idx):
        # TODO: 1. Open the HDF5/NIfTI file at self.file_paths[idx]
        file_path = self.file_paths[idx]
        
        # TODO: 2. Extract the 4 MRI modalities (T1, T1CE, T2, FLAIR)
        with h5py.File(file_path, "r") as f:
            image = f["image"][:].astype(np.float32)
            mask = f["mask"][:]
            
        # TODO: 3. Stack them into a numpy array of shape (4, H, W)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 3 and image.shape[0] == 4:
            pass

        image = image.astype(np.float32)
            
        # TODO: 4. Extract the ground truth segmentation mask of shape (1, H, W) or (Classes, H, W)
        if mask.ndim == 2:
            whole_tumor = (mask > 0).astype(np.float32)
            tumor_core = np.isin(mask, [1, 4]).astype(np.float32)
            enhancing_tumor = (mask == 4).astype(np.float32)
            mask = np.stack([whole_tumor, tumor_core, enhancing_tumor], axis=0)

        elif mask.ndim == 3 and mask.shape[-1] == 3:
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        elif mask.ndim == 3 and mask.shape[0] == 3:
            mask = mask.astype(np.float32)

        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
            whole_tumor = (mask > 0).astype(np.float32)
            tumor_core = np.isin(mask, [1, 4]).astype(np.float32)
            enhancing_tumor = (mask == 4).astype(np.float32)
            mask = np.stack([whole_tumor, tumor_core, enhancing_tumor], axis=0)

        elif mask.ndim == 3 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)
            whole_tumor = (mask > 0).astype(np.float32)
            tumor_core = np.isin(mask, [1, 4]).astype(np.float32)
            enhancing_tumor = (mask == 4).astype(np.float32)
            mask = np.stack([whole_tumor, tumor_core, enhancing_tumor], axis=0)

        if self.transform:
            image_hwc = np.transpose(image, (1, 2, 0))
            mask_hwc = np.transpose(mask, (1, 2, 0))

            augmented = self.transform(image=image_hwc, mask=mask_hwc)

            image = np.transpose(augmented["image"], (2, 0, 1))
            mask = np.transpose(augmented["mask"], (2, 0, 1))

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()

        return image_tensor, mask_tensor

# TODO: Create a function `get_dataloaders(train_paths, val_paths, batch_size)` 
# that returns train and validation DataLoaders.
def get_dataloaders(train_paths, val_paths, batch_size, train_transform=None, val_transform=None, num_workers=0):
    train_dataset = BraTSDataset(train_paths, transform=train_transform)
    val_dataset = BraTSDataset(val_paths, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
