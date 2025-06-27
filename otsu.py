import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from collections import deque

# ==================== PART 1: OTSU'S ALGORITHM ====================

def create_synthetic_image(width=100, height=100):
    """
    Create a synthetic image with 2 objects and background (3 pixel values total)
    """
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Background: value 50
    image[:, :] = 50
    
    # Object 1: Circle in upper left, value 120
    center1 = (25, 25)
    radius1 = 15
    y1, x1 = np.ogrid[:height, :width]
    mask1 = (x1 - center1[0])**2 + (y1 - center1[1])**2 <= radius1**2
    image[mask1] = 120
    
    # Object 2: Rectangle in lower right, value 200
    image[60:80, 60:90] = 200
    
    return image

def add_gaussian_noise(image, mean=0, std=20):
    """
    Add Gaussian noise to the image
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(np.float64) + noise
    # Clip values to valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def otsu_threshold(image):
    """
    Implement Otsu's thresholding algorithm
    """
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(np.float64)
    
    # Normalize histogram
    hist_norm = hist / hist.sum()
    
    # Calculate cumulative sums
    cumsum = np.cumsum(hist_norm)
    cumsum_mean = np.cumsum(hist_norm * np.arange(256))
    
    # Global mean
    global_mean = cumsum_mean[-1]
    
    # Initialize variables
    max_variance = 0
    optimal_threshold = 0
    
    # Try all possible thresholds
    for t in range(256):
        # Weight of background class
        w0 = cumsum[t]
        # Weight of foreground class
        w1 = 1 - w0
        
        # Skip if one class is empty
        if w0 == 0 or w1 == 0:
            continue
            
        # Mean of background class
        mean0 = cumsum_mean[t] / w0 if w0 > 0 else 0
        # Mean of foreground class
        mean1 = (global_mean - cumsum_mean[t]) / w1 if w1 > 0 else 0
        
        # Between-class variance
        between_class_variance = w0 * w1 * (mean0 - mean1) ** 2
        
        # Update optimal threshold
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = t
    
    return optimal_threshold

def apply_threshold(image, threshold):
    """
    Apply threshold to create binary image
    """
    return (image > threshold).astype(np.uint8) * 255

def test_otsu_algorithm():
    """
    Test Otsu's algorithm with synthetic image and noise
    """
    print("=== Testing Otsu's Algorithm ===")
    
    # Create synthetic image
    original_image = create_synthetic_image()
    print(f"Original image shape: {original_image.shape}")
    print(f"Original image pixel values: {np.unique(original_image)}")
    
    # Add Gaussian noise
    noisy_image = add_gaussian_noise(original_image, std=15)
    print(f"Noisy image pixel value range: [{noisy_image.min()}, {noisy_image.max()}]")
    
    # Apply Otsu's algorithm
    threshold = otsu_threshold(noisy_image)
    print(f"Otsu's optimal threshold: {threshold}")
    
    # Create binary image
    binary_image = apply_threshold(noisy_image, threshold)
    
    # Compare with OpenCV's implementation
    cv_threshold, cv_binary = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"OpenCV Otsu threshold: {cv_threshold}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(binary_image, cmap='gray')
    axes[0, 2].set_title(f'Our Otsu Result (t={threshold})')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv_binary, cmap='gray')
    axes[1, 0].set_title(f'OpenCV Otsu Result (t={cv_threshold:.0f})')
    axes[1, 0].axis('off')
    
    # Histogram
    axes[1, 1].hist(noisy_image.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 1].axvline(threshold, color='red', linestyle='--', label=f'Our threshold: {threshold}')
    axes[1, 1].axvline(cv_threshold, color='green', linestyle='--', label=f'OpenCV threshold: {cv_threshold:.0f}')
    axes[1, 1].set_title('Histogram with Thresholds')
    axes[1, 1].legend()
    
    # Difference image
    diff_image = np.abs(binary_image.astype(int) - cv_binary.astype(int))
    axes[1, 2].imshow(diff_image, cmap='hot')
    axes[1, 2].set_title('Difference between implementations')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original_image, noisy_image, binary_image, threshold

# ==================== PART 2: REGION GROWING ====================

def region_growing(image, seeds, threshold_range=10):
    """
    Implement region growing segmentation
    
    Parameters:
    - image: input grayscale image
    - seeds: list of seed points [(x1, y1), (x2, y2), ...]
    - threshold_range: maximum difference from seed pixel values
    
    Returns:
    - segmented: binary mask of segmented region
    """
    height, width = image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)
    
    # Queue for BFS
    queue = deque()
    
    # Get seed pixel values
    seed_values = []
    for seed in seeds:
        x, y = seed
        if 0 <= x < width and 0 <= y < height:
            seed_values.append(image[y, x])
            queue.append((x, y))
            visited[y, x] = True
            segmented[y, x] = 255
    
    # Calculate mean of seed values for comparison
    mean_seed_value = np.mean(seed_values)
    
    # 8-connected neighborhood
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # BFS region growing
    while queue:
        x, y = queue.popleft()
        
        # Check all neighbors
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                # Check if pixel value is within threshold range
                pixel_value = image[ny, nx]
                if abs(pixel_value - mean_seed_value) <= threshold_range:
                    visited[ny, nx] = True
                    segmented[ny, nx] = 255
                    queue.append((nx, ny))
    
    return segmented

def region_growing_adaptive(image, seeds, threshold_range=10):
    """
    Advanced region growing with adaptive threshold based on local statistics
    """
    height, width = image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)
    
    # Queue for BFS
    queue = deque()
    region_pixels = []
    
    # Initialize with seeds
    for seed in seeds:
        x, y = seed
        if 0 <= x < width and 0 <= y < height:
            queue.append((x, y))
            visited[y, x] = True
            segmented[y, x] = 255
            region_pixels.append(image[y, x])
    
    # 8-connected neighborhood
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # BFS region growing with adaptive threshold
    while queue:
        x, y = queue.popleft()
        
        # Update statistics
        current_mean = np.mean(region_pixels)
        current_std = np.std(region_pixels)
        adaptive_threshold = max(threshold_range, current_std * 1.5)
        
        # Check all neighbors
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                pixel_value = image[ny, nx]
                
                # Check if pixel value is within adaptive threshold
                if abs(pixel_value - current_mean) <= adaptive_threshold:
                    visited[ny, nx] = True
                    segmented[ny, nx] = 255
                    queue.append((nx, ny))
                    region_pixels.append(pixel_value)
    
    return segmented

def test_region_growing():
    """
    Test region growing algorithm
    """
    print("\n=== Testing Region Growing ===")
    
    # Create test image
    original_image = create_synthetic_image(150, 150)
    noisy_image = add_gaussian_noise(original_image, std=10)
    
    # Define seeds for different objects
    seeds_object1 = [(25, 25)]  # Inside first object (circle)
    seeds_object2 = [(75, 70)]  # Inside second object (rectangle)
    seeds_background = [(10, 10), (140, 140)]  # Background seeds
    
    print(f"Test image shape: {noisy_image.shape}")
    print(f"Seeds for object 1: {seeds_object1}")
    print(f"Seeds for object 2: {seeds_object2}")
    print(f"Seeds for background: {seeds_background}")
    
    # Apply region growing with different thresholds
    thresholds = [15, 25, 35]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Show original and noisy images
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    # Mark seeds on the image
    seed_image = noisy_image.copy()
    for seed in seeds_object1:
        cv2.circle(seed_image, seed, 3, 255, -1)
    for seed in seeds_object2:
        cv2.circle(seed_image, seed, 3, 0, -1)
    for seed in seeds_background:
        cv2.circle(seed_image, seed, 3, 128, -1)
    
    axes[0, 2].imshow(seed_image, cmap='gray')
    axes[0, 2].set_title('Seeds (White=Obj1, Black=Obj2, Gray=BG)')
    axes[0, 2].axis('off')
    
    # Adaptive region growing
    adaptive_result = region_growing_adaptive(noisy_image, seeds_object1, threshold_range=20)
    axes[0, 3].imshow(adaptive_result, cmap='gray')
    axes[0, 3].set_title('Adaptive Region Growing')
    axes[0, 3].axis('off')
    
    # Test different thresholds
    for i, threshold in enumerate(thresholds):
        # Object 1
        result1 = region_growing(noisy_image, seeds_object1, threshold)
        axes[i+1, 0].imshow(result1, cmap='gray')
        axes[i+1, 0].set_title(f'Object 1 (threshold={threshold})')
        axes[i+1, 0].axis('off')
        
        # Object 2
        result2 = region_growing(noisy_image, seeds_object2, threshold)
        axes[i+1, 1].imshow(result2, cmap='gray')
        axes[i+1, 1].set_title(f'Object 2 (threshold={threshold})')
        axes[i+1, 1].axis('off')
        
        # Background
        result_bg = region_growing(noisy_image, seeds_background, threshold)
        axes[i+1, 2].imshow(result_bg, cmap='gray')
        axes[i+1, 2].set_title(f'Background (threshold={threshold})')
        axes[i+1, 2].axis('off')
        
        # Combined result
        combined = np.zeros_like(noisy_image)
        combined[result1 > 0] = 85   # Object 1 in dark gray
        combined[result2 > 0] = 170  # Object 2 in light gray
        combined[result_bg > 0] = 255  # Background in white
        
        axes[i+1, 3].imshow(combined, cmap='gray')
        axes[i+1, 3].set_title(f'Combined (threshold={threshold})')
        axes[i+1, 3].axis('off')
        
        # Print statistics
        print(f"\nThreshold {threshold}:")
        print(f"  Object 1 pixels: {np.sum(result1 > 0)}")
        print(f"  Object 2 pixels: {np.sum(result2 > 0)}")
        print(f"  Background pixels: {np.sum(result_bg > 0)}")
    
    plt.tight_layout()
    plt.show()
    
    return noisy_image, adaptive_result

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test Otsu's algorithm
    original, noisy, binary, threshold = test_otsu_algorithm()
    
    # Test region growing
    test_image, adaptive_result = test_region_growing()
    
    print("\n=== Analysis Complete ===")
    print("Both algorithms have been implemented and tested successfully!")
    print("\nKey observations:")
    print("1. Otsu's algorithm automatically finds optimal threshold for binarization")
    print("2. Region growing allows for flexible segmentation based on seed points")
    print("3. Noise affects both algorithms - preprocessing may be beneficial")
    print("4. Adaptive region growing can handle varying intensities better")