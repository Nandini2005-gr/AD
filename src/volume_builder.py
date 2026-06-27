import os
import cv2
import numpy as np

DATA_PATH = r"C:\Users\User\Documents\Data"

PATIENT_ID = "OAS1_0308"
CLASS_NAME = "Moderate Dementia"

class_path = os.path.join(DATA_PATH, CLASS_NAME)

slice_files = []

for file in os.listdir(class_path):

    if file.startswith(PATIENT_ID) and "_mpr-1_" in file:

        slice_number = int(file.split("_")[-1].replace(".jpg", ""))

        slice_files.append((slice_number, file))

slice_files.sort()

volume = []

for _, file in slice_files:

    img_path = os.path.join(class_path, file)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    volume.append(img)

volume = np.array(volume)
print("Volume Shape:", volume.shape)
print("Data Type:", volume.dtype)

print("Min Pixel:", volume.min())
print("Max Pixel:", volume.max())

print("Volume Memory (MB):",
      round(volume.nbytes / (1024*1024), 2))
