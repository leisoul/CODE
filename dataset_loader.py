import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class Datasets(Dataset):
    def __init__(self, gt, deg):
        """
        Args:
            gt (str): Path to folder with ground truth images.
            deg (str): Path to folder with degraded images.
        """
        super().__init__()

        self.ground_truth = []
        self.degradation = []

        # Center crop to 256x256 after converting to tensor
        self.transform = transforms.CenterCrop((256, 256))
        
        # Collect file paths
        # ⚠️ Assumes that GT and DEG have identical filenames and same number of images
        for f in os.listdir(gt):
            self.ground_truth.append(os.path.join(gt, f))
            self.degradation.append(os.path.join(deg, f))


    def _load_and_process_image(self, path):
        """Load an image using OpenCV and convert BGR → RGB"""
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        """Return a paired (clean, degraded) image with augmentation + preprocessing"""
        clean_img = self._load_and_process_image(self.ground_truth[idx])
        degrade_img = self._load_and_process_image(self.degradation[idx])

        # Apply the same random augmentation to both GT and degraded images
        mode = random.randint(0, 7)
        clean_img = data_augmentation(clean_img, mode).copy()
        degrade_img = data_augmentation(degrade_img, mode).copy()
        
        # Convert numpy → torch tensor, normalize [0, 255] → [0, 1]
        clean_img = transforms.ToTensor()(clean_img)
        degrade_img = transforms.ToTensor()(degrade_img)

        # Apply cropping
        clean_img = self.transform(clean_img)
        degrade_img = self.transform(degrade_img)

        return clean_img, degrade_img

    def __len__(self):
        """Return dataset size"""
        return len(self.ground_truth)
    


def data_augmentation(image, mode):
    """
    Perform simple flip/rotation augmentations.
    mode values:
        0: original
        1: flip up/down
        2: rotate 90° counterclockwise
        3: rotate 90° + flip up/down
        4: rotate 180°
        5: rotate 180° + flip up/down
        6: rotate 270°
        7: rotate 270° + flip up/down
    """
    augmentations = {
        0: lambda img: img,
        1: lambda img: np.flipud(img),
        2: lambda img: np.rot90(img),
        3: lambda img: np.flipud(np.rot90(img)),
        4: lambda img: np.rot90(img, k=2),
        5: lambda img: np.flipud(np.rot90(img, k=2)),
        6: lambda img: np.rot90(img, k=3),
        7: lambda img: np.flipud(np.rot90(img, k=3)),
    }
    return augmentations[mode](image)
