import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)