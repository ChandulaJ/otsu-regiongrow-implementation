import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_growing(image, seed_point, threshold=5):
    height, width = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    
    seed_value = image[seed_point[1], seed_point[0]]
    
    region = [seed_point]
    segmented[seed_point[1], seed_point[0]] = 255
    visited[seed_point[1], seed_point[0]] = True
    
    # 8-connectivity neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),         (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    
    while region:
        x, y = region.pop(0)
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                neighbor_value = image[ny, nx]
                if abs(int(neighbor_value) - int(seed_value)) <= threshold:
                    segmented[ny, nx] = 255
                    region.append((nx, ny))
                visited[ny, nx] = True

    return segmented

original_img = cv2.imread('brain.jpg', cv2.IMREAD_COLOR)
# Load grayscale image
img = cv2.imread('brain.jpg', cv2.IMREAD_GRAYSCALE)

# Choose a seed point inside the brain tumor
seed = (500, 800)

# Apply region growing
output = region_growing(img, seed, threshold=10)

# Save the output image
cv2.imwrite('region_growing_result.jpg', output)
print("Output image saved as 'region_growing_result.jpg'")

# Show result
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Greyscale Image with Seed Point")
plt.imshow(img, cmap='gray')
plt.plot(seed[0], seed[1], 'ro', markersize=8, label='Seed Point')
plt.legend()

plt.subplot(1, 3, 3)
plt.title("Region Grown Segmentation")
plt.imshow(output, cmap='gray')
plt.show()
