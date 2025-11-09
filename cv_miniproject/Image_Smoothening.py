#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ================================
# 🔧 INSTALLATION & IMPORTS
# ================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ================================
# 📤 SELECT IMAGE FILE
# ================================
Tk().withdraw()  # Hide the main Tkinter window
img_path = filedialog.askopenfilename(
    title="Select an image to apply filters",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.jfif")]
)

if not img_path:
    print("❌ No image selected. Exiting...")
    exit()

# Read and convert to RGB
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ================================
# 🧹 APPLY DIFFERENT FILTERS
# ================================
avg_blur = cv2.blur(img, (5,5))
gaussian_blur = cv2.GaussianBlur(img, (5,5), 0)
median_blur = cv2.medianBlur(img, 5)
bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)

# ================================
# 🖼️ DISPLAY RESULTS
# ================================
titles = ['Original Image', 'Average Blur', 'Gaussian Blur', 'Median Blur', 'Bilateral Blur']
images = [img, avg_blur, gaussian_blur, median_blur, bilateral_blur]

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
