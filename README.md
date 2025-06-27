# Otsu's Thresholding & Region Growing Implementation

A comprehensive implementation of two fundamental image processing algorithms: **Otsu's automatic thresholding** and **Region Growing segmentation**. This project demonstrates these techniques on grayscale images with practical applications in medical image analysis.

## 🔍 Overview

This repository contains Python implementations of:

1. **Otsu's Algorithm**: An automatic threshold selection method for image binarization
2. **Region Growing**: A pixel-based image segmentation technique

Both algorithms are implemented from scratch using NumPy and OpenCV for educational and research purposes.

## 🚀 Features

### Otsu's Thresholding (`otsuAlgorithm.py`)
- ✅ **From-scratch implementation** of Otsu's algorithm
- ✅ **Noise addition** simulation using Gaussian noise
- ✅ **Automatic threshold detection** based on between-class variance maximization
- ✅ **Statistical analysis** of original, noisy, and thresholded images
- ✅ **Comprehensive visualization** with histograms and results

### Region Growing (`regionGrow.py`)
- ✅ **8-connectivity region growing** algorithm
- ✅ **Seed-based segmentation** with customizable threshold
- ✅ **Medical image processing** (brain tumor segmentation example)
- ✅ **Visual comparison** of original and segmented results

## 📋 Requirements

```bash
pip install numpy opencv-python matplotlib
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChandulaJ/otsu-regiongrow-implementation.git
   cd otsu-regiongrow-implementation
   ```

## 💻 Usage

### Otsu's Thresholding

```bash
python otsuAlgorithm.py
```

**What it does:**
- Loads `image.jpg` and adds Gaussian noise
- Implements Otsu's algorithm to find optimal threshold
- Creates binary segmentation
- Displays comparison plots and saves results

**Output:**
- `gn_img.jpg`: Image with added Gaussian noise
- `otsu_result.jpg`: Binary image after Otsu thresholding
- Console output with threshold value and statistics

### Region Growing Segmentation

```bash
python regionGrow.py
```

**What it does:**
- Loads `brain.jpg` for image segmentation
- Applies region growing from a seed point (500, 800)
- Uses 8-connectivity with configurable threshold
- Visualizes segmentation results

**Output:**
- `region_growing_result.jpg`: Segmented binary image
- Visual comparison plots

## 📁 Project Structure

```
otsu-regiongrow-implementation/
│
├── 📄 otsuAlgorithm.py          # Otsu's thresholding implementation
├── 📄 regionGrow.py             # Region growing implementation
├── 📄 README.md                 # This file
│
├── 🖼️ image.jpg                 # Input image for Otsu algorithm
├── 🖼️ brain.jpg                 # Medical image for region growing
│
└── 📤 Output Files:
    ├── gn_img.jpg               # Noisy image
    ├── otsu_result.jpg          # Otsu thresholding result
    └── region_growing_result.jpg # Region growing result
```

## 🧮 Algorithm Details

### Otsu's Method
Otsu's algorithm finds the optimal threshold by:
1. Computing image histogram
2. Calculating between-class variance for all possible thresholds
3. Selecting threshold that maximizes variance between foreground/background

**Formula:**
```
σ²_B(t) = w₀(t) × w₁(t) × [μ₀(t) - μ₁(t)]²
```
Where:
- `w₀`, `w₁`: class probabilities
- `μ₀`, `μ₁`: class means
- `t`: threshold value

### Region Growing
Region growing segments images by:
1. Starting from seed pixel(s)
2. Adding neighboring pixels with similar intensity
3. Continuing until no more pixels meet criteria

**Connectivity:** 8-connected neighbors
**Similarity criterion:** `|pixel_value - seed_value| ≤ threshold`

## 📊 Example Results

### Otsu's Thresholding Pipeline:
1. **Original Image** → 2. **+ Gaussian Noise** → 3. **Otsu Threshold** → 4. **Binary Result**

### Region Growing Pipeline:
1. **Medical Image** → 2. **Seed Selection** → 3. **Region Growth** → 4. **Segmented Region**

## ⚙️ Customization

### Modify Otsu Parameters:
```python
# Adjust noise level
cv2.randn(gauss_noise, 128, 20)  # mean=128, std=20

# Change noise blend ratio
gauss_noise = (gauss_noise * 0.5).astype(np.uint8)  # 50% blend
```

### Modify Region Growing Parameters:
```python
# Change seed point
seed = (x, y)  # pixel coordinates

# Adjust similarity threshold
output = region_growing(img, seed, threshold=10)  # intensity difference
```

## 🎯 Applications

- **Medical Imaging**: Tumor detection, organ segmentation
- **Quality Control**: Defect detection in manufacturing
- **Document Processing**: Text extraction, background removal
- **Research**: Computer vision algorithm development

## 📈 Performance Notes

- **Otsu's Algorithm**: O(L²) where L is number of gray levels (256)
- **Region Growing**: O(N) where N is number of pixels in the region
- **Memory Usage**: Proportional to image size


⭐ **Star this repository if you found it helpful!**