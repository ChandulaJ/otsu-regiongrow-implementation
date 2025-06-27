import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def add_gaussian_noise(image, mean=0, std=25):
    """
    Add Gaussian noise to an image
    
    Args:
        image: Input image array
        mean: Mean of the Gaussian noise
        std: Standard deviation of the Gaussian noise
    
    Returns:
        Noisy image
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    # Clip values to valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def compute_histogram(image):
    """
    Compute histogram of grayscale image
    
    Args:
        image: Grayscale image array
    
    Returns:
        Histogram array of length 256
    """
    hist = np.zeros(256, dtype=int)
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    return hist

def otsu_threshold(image):
    """
    Implement Otsu's thresholding algorithm
    
    Args:
        image: Grayscale image array
    
    Returns:
        Optimal threshold value
    """
    # Compute histogram
    hist = compute_histogram(image)
    
    # Total number of pixels
    total_pixels = image.size
    
    # Initialize variables
    sum_total = sum(i * hist[i] for i in range(256))
    sum_background = 0
    weight_background = 0
    max_variance = 0
    optimal_threshold = 0
    
    # Try all possible thresholds
    for threshold in range(256):
        # Update background statistics
        weight_background += hist[threshold]
        if weight_background == 0:
            continue
            
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += threshold * hist[threshold]
        
        # Calculate means
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Calculate between-class variance
        between_class_variance = (weight_background * weight_foreground * 
                                (mean_background - mean_foreground) ** 2)
        
        # Update optimal threshold if variance is maximum
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = threshold
    
    return optimal_threshold

def apply_threshold(image, threshold):
    """
    Apply binary thresholding to image
    
    Args:
        image: Input grayscale image
        threshold: Threshold value
    
    Returns:
        Binary image
    """
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255
    return binary_image


# Load the provided image or create synthetic one
try:
    # Try to load the provided image
    img = Image.open('image.jpg')

    original_image = np.array(img)
    print("Loaded provided image: image.jpg")
except FileNotFoundError:
    # Create synthetic image if file not found
    original_image = create_synthetic_image()
    print("Created synthetic image (image.jpg not found)")

# Add Gaussian noise
noisy_image = add_gaussian_noise(original_image, mean=0, std=20)

# Apply Otsu's algorithm
threshold = otsu_threshold(noisy_image)
binary_result = apply_threshold(noisy_image, threshold)

# Compare with OpenCV's implementation
cv2_threshold, cv2_binary = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axes[0, 0].imshow(original_image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Noisy image
axes[0, 1].imshow(noisy_image, cmap='gray')
axes[0, 1].set_title('Noisy Image')
axes[0, 1].axis('off')

# Histogram of noisy image
axes[0, 2].hist(noisy_image.flatten(), bins=256, range=(0, 256), alpha=0.7, color='blue')
axes[0, 2].axvline(threshold, color='red', linestyle='--', label=f'Otsu Threshold: {threshold}')
axes[0, 2].set_title('Histogram with Otsu Threshold')
axes[0, 2].set_xlabel('Pixel Value')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()

# Our Otsu result
axes[1, 0].imshow(binary_result, cmap='gray')
axes[1, 0].set_title(f'Our Otsu Result (T={threshold})')
axes[1, 0].axis('off')

# OpenCV Otsu result
axes[1, 1].imshow(cv2_binary, cmap='gray')
axes[1, 1].set_title(f'OpenCV Otsu Result (T={int(cv2_threshold)})')
axes[1, 1].axis('off')

# Difference between methods
diff = np.abs(binary_result.astype(int) - cv2_binary.astype(int))
axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title('Difference (Our - OpenCV)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Print results
print(f"\nResults:")
print(f"Our Otsu threshold: {threshold}")
print(f"OpenCV Otsu threshold: {int(cv2_threshold)}")
print(f"Difference: {abs(threshold - int(cv2_threshold))}")

# Calculate some statistics
unique_original = np.unique(original_image)
unique_noisy = np.unique(noisy_image)
print(f"\nOriginal image unique values: {len(unique_original)} values")
print(f"Range: {unique_original.min()} to {unique_original.max()}")
print(f"Noisy image unique values: {len(unique_noisy)} values")
print(f"Range: {unique_noisy.min()} to {unique_noisy.max()}")

# Evaluate segmentation quality (if we have ground truth)
if len(unique_original) <= 5:  # Likely synthetic or simple image
    # Create ground truth binary mask
    # Assume anything above the middle value is foreground
    middle_val = sorted(unique_original)[len(unique_original)//2]
    ground_truth = (original_image > middle_val).astype(np.uint8) * 255
    
    # Calculate accuracy
    our_accuracy = np.sum(binary_result == ground_truth) / ground_truth.size
    cv2_accuracy = np.sum(cv2_binary == ground_truth) / ground_truth.size
    
    print(f"\nSegmentation Accuracy (compared to ground truth):")
    print(f"Our implementation: {our_accuracy:.3f}")
    print(f"OpenCV implementation: {cv2_accuracy:.3f}")