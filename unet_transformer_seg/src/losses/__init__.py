"""Loss functions for segmentation tasks."""

from .dice_loss import DiceLoss
from .bce_dice_loss import BCEDiceLoss
from .seg_losses import get_loss_function

__all__ = [
    'DiceLoss',
    'BCEDiceLoss', 
    'get_loss_function'
]