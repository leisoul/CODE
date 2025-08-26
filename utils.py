import torch
import torch.nn.functional as F

def compute_psnr_mean(preds, targets, max_val=1.0):
    """
    Compute the average PSNR (Peak Signal-to-Noise Ratio) between predictions and targets.

    Args:
        preds (torch.Tensor): Model output images, shape (B, C, H, W), values in [0, max_val].
        targets (torch.Tensor): Ground truth images, same shape as preds.
        max_val (float): Maximum possible pixel value (1.0 if normalized, 255 if raw images).

    Returns:
        float: Mean PSNR value across the batch.
    """
    # Compute per-pixel MSE without reduction
    mse = F.mse_loss(preds, targets, reduction='none')

    # Flatten each image to compute mean MSE per image
    mse = mse.view(mse.size(0), -1).mean(dim=1)

    # PSNR formula: 10 * log10(MAX^2 / MSE)
    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-8))

    # Return the batch mean as a Python float
    return psnr.mean().item()


def L1Loss(x, y):
    """
    Smoothed L1 loss (similar to Charbonnier Loss).

    Args:
        x (torch.Tensor): Prediction.
        y (torch.Tensor): Target.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    diff = x - y
    # Charbonnier loss: sqrt(diff^2 + epsilon^2)
    epsilon = 1e-3
    loss = torch.mean(torch.sqrt((diff * diff) + (epsilon * epsilon)))
    return loss
