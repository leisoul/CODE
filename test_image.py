import torch
import os
from torchvision import transforms
import cv2
from torchvision.utils import save_image
from model import model

if __name__ == '__main__':
    # -----------------------------
    # 1. Load model and checkpoint
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model().to(device)
    model.eval()  # set model to inference mode

    # Load checkpoint
    checkpoint = torch.load('weight.pth', map_location=device)  # safer: map to correct device
    model.load_state_dict(checkpoint['state_dict'])

    # -----------------------------
    # 2. Load and preprocess image
    # -----------------------------
    img_path = 'dataset/DEG/0001_SRGB_010_0.png'

    # OpenCV loads in BGR, so convert to RGB
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # Convert numpy → torch tensor, scale to [0,1], add batch dim
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # -----------------------------
    # 3. Forward pass (inference)
    # -----------------------------
    with torch.no_grad():  # no gradients needed
        output = model(img)

    # -----------------------------
    # 4. Save restored image
    # -----------------------------
    os.makedirs('results', exist_ok=True)
    save_image(output, f"results/{os.path.basename(img_path)}")

    print(f"Restored image saved at: results/{os.path.basename(img_path)}")
