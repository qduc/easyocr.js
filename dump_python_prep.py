import torch
import easyocr
import numpy as np
from PIL import Image
import sys

image_path = sys.argv[1]
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models', download_enabled=False)

# We want the preprocessed image that goes into the recognizer
# EasyOCR usually does this inside readtext -> recognize
img = Image.open(image_path).convert('L') # Gray
img_cv = np.array(img)

# Mocking the recognizer's preprocess
# In EasyOCR, recognizer preprocess is usually:
# 1. Resize height to 32, keep aspect ratio (capped at some width)
# 2. Normalize to [0, 1], then (x - 0.5) / 0.5

h, w = img_cv.shape
target_h = 32
scale = target_h / h
target_w = int(round(w * scale))

# EasyOCR actually uses some padding and capping
# Let's see what easyocr.Reader.recognize does

# I'll just use the internal recognizer's preprocess if possible
# or just replicate it.
resized = Image.fromarray(img_cv).resize((target_w, target_h), Image.BICUBIC)
resized_img = np.array(resized).astype(np.float32) / 255.0
normalized = (resized_img - 0.5) / 0.5

print(f"Shapes: {img_cv.shape} -> {normalized.shape}")
np.save("python_prep.npy", normalized)
