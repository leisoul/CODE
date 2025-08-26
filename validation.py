import torch
from torch.utils.data import DataLoader
from model import model
from utils import compute_psnr_mean
from dataset_loader import Datasets

if __name__ == '__main__':
    # Select device: use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and move it to the selected device
    model = model().to(device)

    # Load checkpoint
    checkpoint = torch.load('0525_GOPRO_model_64/model_best.pth', map_location=device)  # safer: map to correct device
    model.load_state_dict(checkpoint['state_dict'])

    # Load validation dataset
    # gt: path to ground truth images
    # deg: path to degraded (noisy/corrupted) images
    val_dataset = Datasets(gt='dataset/DPDD/target', deg='dataset/DPDD/inputC')

    # Create validation DataLoader
    # batch_size=8: process 8 images per batch
    # shuffle=False: no need to shuffle validation set
    # num_workers=8: number of worker processes for data loading
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Set model to evaluation mode (disables dropout, uses running stats in BatchNorm, etc.)
    model.eval()

    # Iterate through the validation dataset
    for idx, data_val in enumerate(val_loader):
        # ⚠️ Important: dataset.__getitem__() is expected to return (GT, DEG)
        # data_val[0] → Ground Truth image
        # data_val[1] → Degraded image
        target = data_val[0].to(device)   # Move Ground Truth image to device
        input_ = data_val[1].to(device)   # Move Degraded image to device
        
        # Forward pass: feed degraded image into the model to get restored output
        output = model(input_)

        # Compute PSNR score between model output and ground truth
        score = compute_psnr_mean(output, target)

    # Print validation PSNR score (⚠️ only the last batch is shown here)
    print(f'Validation PSNR: {score:.4f}')
