import cv2
import numpy as np

# Create a 112x112 RGB image
img = np.zeros((112, 112, 3), dtype=np.uint8)
img[..., 0] = 255  # set the red channel to 255

# Save the image as a JPEG file
cv2.imwrite('my_image.jpg', img)

