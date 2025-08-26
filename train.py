import torch
from torch.utils.data import DataLoader
from model import model
from utils import compute_psnr_mean, L1Loss
from dataset_loader import Datasets

if __name__ == '__main__':
    # Select device: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and move it to the device
    model = model().to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Load datasets (Ground Truth + Degraded images)
    train_dataset = Datasets(gt='dataset/GT', deg='dataset/DEG')
    val_dataset = Datasets(gt='dataset/GT', deg='dataset/DEG')

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader   = DataLoader(dataset=val_dataset,   batch_size=8, shuffle=False, num_workers=8)

    # Training loop (100 epochs)
    for epoch in range(1, 100):
        # --------------------
        # Training phase
        # --------------------
        model.train()
        for batch_idx, train_data in enumerate(train_loader):
            optimizer.zero_grad()

            # Unpack training data (GT, degraded image)
            target = train_data[0].to(device)
            input_ = train_data[1].to(device)

            # Forward pass
            output = model(input_)

            # Compute L1 loss
            loss = L1Loss(output, target)

            # Backpropagation
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Training Loss: {loss:.4f}')

        # --------------------
        # Validation phase
        # --------------------
        model.eval()
        psnr_scores = []  # collect scores for averaging
        with torch.no_grad():  # no gradients needed in validation
            for idx, data_val in enumerate(val_loader):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)
                
                # Forward pass
                output = model(input_)

                # Compute PSNR
                score = compute_psnr_mean(output, target)
                psnr_scores.append(score)

        # Compute average PSNR across validation set
        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        print(f'Validation PSNR: {avg_psnr:.4f}')
