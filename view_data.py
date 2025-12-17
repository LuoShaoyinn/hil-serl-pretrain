#! python3

import cv2
import numpy as np
import pickle

# 1. Load the data
with open('data/side_policy.pkl', 'rb') as f:
    data = pickle.load(f)  # Shape: (26001, 128, 128, 3)

# 2. Select the first image and ensure it's a NumPy array
img = np.array(data[3])

# 3. Convert Scale and Format
# If data is normalized (0 to 1), scale it to (0 to 255)
if img.max() <= 1.0:
    img = (img * 255).astype(np.uint8)
else:
    img = img.astype(np.uint8)

# 4. Convert RGB to BGR (OpenCV default)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 5. Display the image
cv2.imshow("Dataset Sample", img_bgr)
print("Press any key in the image window to close.")
cv2.waitKey(0)  # Wait indefinitely for a key press
cv2.destroyAllWindows()
