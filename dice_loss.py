# dice loss implementation
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # TODO: 1. Apply Sigmoid or Softmax to `inputs` to get probabilities
        # TODO: 2. Flatten both inputs and targets to 1D arrays
        # TODO: 3. Calculate the intersection: sum(inputs * targets)
        # TODO: 4. Calculate the denominator: sum(inputs) + sum(targets)
        # TODO: 5. Calculate Dice: (2. * intersection + smooth) / (denominator + smooth)
        # TODO: 6. Return 1 - Dice
        
        loss = 0.0 # Replace with actual calculation
        return loss