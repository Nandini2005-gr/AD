import os
import cv2
import numpy as np

DATA_PATH = r"E:\Alzheimer_Project\Data"

LABELS = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

TARGET_SIZE = (64, 64)

def load_patient_volume(class_name, patient_id):

    class_path = os.path.join(DATA_PATH, class_name)

    slice_files = []

    for file in os.listdir(class_path):

        if file.startswith(patient_id) and "_mpr-1_" in file:

            slice_no = int(
                file.split("_")[-1].replace(".jpg", "")
            )

            slice_files.append((slice_no, file))

    slice_files.sort()

    volume = []

    for _, file in slice_files:

        img = cv2.imread(
            os.path.join(class_path, file),
            cv2.IMREAD_GRAYSCALE
        )

        img = cv2.resize(img, TARGET_SIZE)

        img = img.astype(np.float32) / 255.0

        volume.append(img)

    volume = np.array(volume)

    return volume


volume = load_patient_volume(
    "Moderate Dementia",
    "OAS1_0308"
)

print("Volume Shape:", volume.shape)
print("Min:", volume.min())
print("Max:", volume.max())
print("Data Type:", volume.dtype)