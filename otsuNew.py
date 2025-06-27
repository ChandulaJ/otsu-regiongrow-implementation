import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("image.png",0)
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

# Otsu's method
