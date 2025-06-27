import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("image.jpg",0)
print(img.shape)

# Adding noise to the image
gauss_noise=np.zeros((1500,1500),dtype=np.uint8)
cv2.randn(gauss_noise,128,20)
gauss_noise=(gauss_noise*0.5).astype(np.uint8)

gn_img=cv2.add(img,gauss_noise)

fig=plt.figure(dpi=300)
fig.add_subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1,3,2)
plt.imshow(gauss_noise,cmap='gray')
plt.axis("off")
plt.title("Gaussian Noise")

fig.add_subplot(1,3,3)
plt.imshow(gn_img,cmap='gray')
plt.axis("off")
plt.title("Combined")

plt.show()
cv2.imwrite('gn_img.jpg', gn_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# Otsu's algorithm implementation

def otsu_threshold(image):
    """
    Implement Otsu's thresholding algorithm from scratch
    """
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    hist = hist.astype(float)
    total_pixels = image.size
    prob = hist / total_pixels
    
    # Calculate cumulative sums
    cum_sum = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256))
    global_mean = cum_mean[-1]
    
    max_variance = 0
    optimal_threshold = 0
    
    for t in range(256):
        w0 = cum_sum[t]
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            continue
            
        mu0 = cum_mean[t] / w0 if w0 > 0 else 0
        mu1 = (global_mean - cum_mean[t]) / w1 if w1 > 0 else 0
        
        between_class_variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = t
    
    return optimal_threshold, max_variance

def apply_threshold(image, threshold):
    """
    Apply threshold to create binary image
    """
    binary_image = np.zeros_like(image)
    binary_image[image >= threshold] = 255
    return binary_image

# Apply Otsu's algorithm to the noisy image
print("Applying Otsu's thresholding algorithm...")
optimal_thresh, max_var = otsu_threshold(gn_img)
print(f"Optimal threshold found: {optimal_thresh}")
print(f"Maximum between-class variance: {max_var:.6f}")

# Create binary image using threshold
binary_result = apply_threshold(gn_img, optimal_thresh)

# Display results
fig2 = plt.figure(figsize=(12, 8), dpi=100)

# Original noisy image
fig2.add_subplot(2, 2, 1)
plt.imshow(gn_img, cmap='gray')
plt.axis("off")
plt.title("Noisy Image")

# Histogram of the noisy image with Otsu threshold
fig2.add_subplot(2, 2, 2)
hist_counts, bins, patches = plt.hist(gn_img.flatten(), 256, [0, 256], color='black', alpha=0.7)
plt.axvline(x=optimal_thresh, color='red', linestyle='--', linewidth=3, label=f'Otsu Threshold: {optimal_thresh}')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram with Otsu Threshold')
plt.legend()
plt.grid(True, alpha=0.3)


# Otsu thresholded result
fig2.add_subplot(2, 2, 3)
plt.imshow(binary_result, cmap='gray')
plt.axis("off")
plt.title(f"Otsu Thresholded (T={optimal_thresh})")

# Original image for comparison
fig2.add_subplot(2, 2, 4)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original (No Noise)")
plt.tight_layout()
plt.show()

print("\n=== Image Statistics ===")
print(f"Original image - Mean: {np.mean(img):.2f}, Std: {np.std(img):.2f}")
print(f"Noisy image - Mean: {np.mean(gn_img):.2f}, Std: {np.std(gn_img):.2f}")
print(f"Binary image - Foreground pixels: {np.sum(binary_result == 255)} ({np.sum(binary_result == 255)/binary_result.size*100:.1f}%)")
print(f"Binary image - Background pixels: {np.sum(binary_result == 0)} ({np.sum(binary_result == 0)/binary_result.size*100:.1f}%)")

# Save the result
cv2.imwrite('otsu_result.jpg', binary_result)
print(f"\nOtsu thresholding result saved as 'otsu_result.jpg'")
