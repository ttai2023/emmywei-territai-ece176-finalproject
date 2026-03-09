# dice loss implementation
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # TODO: 1. Apply Sigmoid or Softmax to `inputs` to get probabilities
        inputs = torch.sigmoid(inputs)
        # TODO: 2. Flatten both inputs and targets to 1D arrays
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # TODO: 3. Calculate the intersection: sum(inputs * targets)
        intersection = (inputs * targets).sum()
        # TODO: 4. Calculate the denominator: sum(inputs) + sum(targets)
        denominator = inputs.sum() + targets.sum()
        # TODO: 5. Calculate Dice: (2. * intersection + smooth) / (denominator + smooth)
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        # TODO: 6. Return 1 - Dice
        loss = 1 - dice

        return loss