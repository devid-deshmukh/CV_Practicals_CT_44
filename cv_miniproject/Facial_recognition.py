#!/usr/bin/env python
# coding: utf-8

# In[1]:


from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog


# In[ ]:


# ============================================
# 🔹 STEP 1: Install dependencies (run in a cell if not installed)
# ============================================
# !pip install deepface opencv-python matplotlib ipywidgets

# ============================================
# 🔹 STEP 2: Import libraries
# ============================================
from IPython.display import display
import ipywidgets as widgets
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import numpy as np

# ============================================
# 🔹 STEP 3: File Upload Widget
# ============================================
upload = widgets.FileUpload(accept='image/*', multiple=False)
display(upload)

# Wait until a file is uploaded
while not upload.value:
    pass

# Load uploaded image
file_info = list(upload.value.values())[0]
image_bytes = file_info['content']
nparr = np.frombuffer(image_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Save temporarily because DeepFace needs a file path
tmp_path = "temp_image.jpg"
cv2.imwrite(tmp_path, img)

# ============================================
# 🔹 STEP 4: Run Facial-Expression Analysis
# ============================================
analysis = DeepFace.analyze(
    img_path=tmp_path,
    actions=['emotion'],  # Can include ['age','gender','race'] too
    enforce_detection=False
)

# Ensure analysis is a list (for multiple faces)
if isinstance(analysis, dict):
    analysis = [analysis]

# ============================================
# 🔹 STEP 5: Print Analysis Results
# ============================================
print("\n✅ Facial-Expression Analysis Results:\n")
for idx, face in enumerate(analysis, start=1):
    print(f"Face {idx}:")
    print("  Dominant Emotion:", face['dominant_emotion'])
    print("  All Emotion Probabilities:", face['emotion'])
    print("  Region:", face['region'], "\n")

# ============================================
# 🔹 STEP 6: Draw Bounding Boxes and Labels
# ============================================
for face in analysis:
    region = face.get('region', {})
    x = int(region.get('x', 0))
    y = int(region.get('y', 0))
    w = int(region.get('w', 0))
    h = int(region.get('h', 0))

    if w > 0 and h > 0:
        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Label emotion
        cv2.putText(img, face['dominant_emotion'], (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# ============================================
# 🔹 STEP 7: Display Annotated Image
# ============================================
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Facial Expressions")
plt.show()

