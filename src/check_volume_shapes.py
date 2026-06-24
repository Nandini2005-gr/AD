import os
import cv2
import numpy as np

DATA_PATH = r"E:\Alzheimer_Project\Data"

patients = [
    ("Non Demented", "OAS1_0001"),
    ("Very mild Dementia", "OAS1_0003"),
    ("Mild Dementia", "OAS1_0028"),
    ("Moderate Dementia", "OAS1_0308")
]

for class_name, patient_id in patients:

    class_path = os.path.join(DATA_PATH, class_name)

    slice_files = []

    for file in os.listdir(class_path):

        if file.startswith(patient_id) and "_mpr-1_" in file:

            slice_no = int(file.split("_")[-1].replace(".jpg", ""))

            slice_files.append((slice_no, file))

    slice_files.sort()

    volume = []

    for _, file in slice_files:

        img = cv2.imread(
            os.path.join(class_path, file),
            cv2.IMREAD_GRAYSCALE
        )

        volume.append(img)

    volume = np.array(volume)

    print(f"{patient_id} -> {volume.shape}")