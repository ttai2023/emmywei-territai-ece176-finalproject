# run training loop

import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    
    # tqdm gives you a nice progress bar in your terminal/notebook
    for images, masks in tqdm(dataloader, desc="Training"):
        # TODO: 1. Move images and masks to `device` (GPU)
        # TODO: 2. Zero the parameter gradients (optimizer.zero_grad())
        # TODO: 3. Forward pass: compute predictions
        # TODO: 4. Calculate loss using `criterion`
        # TODO: 5. Backward pass: compute gradients (loss.backward())
        # TODO: 6. Update weights (optimizer.step())
        # TODO: 7. Accumulate loss for logging
        pass
        
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad(): # No gradients needed for validation!
        for images, masks in tqdm(dataloader, desc="Validating"):
            # TODO: Move to device, forward pass, calculate loss
            # TODO: Calculate Dice Score for evaluation metric
            pass
            
    return val_loss / len(dataloader), val_dice / len(dataloader)