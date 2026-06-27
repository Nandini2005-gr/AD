import os
import cv2
import numpy as np

DATA_PATH = r"C:\Users\User\Documents\Data"

LABELS = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

volumes = []
labels = []

for class_name, label in LABELS.items():

    class_path = os.path.join(DATA_PATH, class_name)

    patients = set()

    for file in os.listdir(class_path):

        if "_mpr-1_" in file:

            patient = "_".join(file.split("_")[:2])

            patients.add(patient)

    for patient in sorted(patients):

        slice_files = []

        for file in os.listdir(class_path):

            if file.startswith(patient) and "_mpr-1_" in file:

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

            volume.append(img)

        volume = np.array(volume)

        volumes.append(volume)
        labels.append(label)

print("Total Volumes:", len(volumes))
print("Total Labels :", len(labels))

print("Sample Shape :", volumes[0].shape)
