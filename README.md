
# Image Restoration Framework

This repository contains the implementation of an image restoration framework that supports multiple tasks, including denoising, deblurring, low-light enhancement, and raindrop removal.  
The framework is based on a U-shaped encoder-decoder architecture with novel attention modules to balance efficiency and performance.


## 📂 Project Structure
```

project_root/
│── models.py             # Network architecture definitions
│── utils.py              # Utility functions
│── dataloader.py         # Dataset loader and preprocessing
│── train.py              # Training entry point
│── validation.py         # Validation and evaluation scripts
│── README.md             # Project description and usage

````

---

## 🚀 Features
- **Enhanced U-shaped architecture** with improved head and tail design.
- **Multi-Branch Directional Convolution Mechanism (MBDM)** for structure recovery.
- **Shallow Feature Channel Attention Module (SF-CAM)** for detail refinement.
- **Lightweight attention mechanism** that captures both global and local features.
- Supports **multiple restoration tasks** with competitive efficiency.

---

## ⚙️ Requirements
- Python 3.8+
- PyTorch >= 1.9
- torchvision
- numpy, scipy, opencv-python
- tqdm, matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## 🏋️ Training

To start training, run:

```bash
python train.py 
```

---

## 📊 Validation

To evaluate the model:

```bash
python validation.py
```

---

## 📂 Dataset

Prepare datasets according to the task (e.g., SIDD for denoising, GoPro for deblurring).
Modify `dataloader.py` to set the dataset paths.

---

## 🔥 Results

The proposed method achieves state-of-the-art performance on multiple benchmarks:

* **Denoising:** SIDD, DND
* **Deblurring:** GoPro
* **Low-light enhancement:** LOL dataset
* **Raindrop removal:** RainDrop dataset

---

## 📄 Citation

If you find this work useful, please cite:

```

```

