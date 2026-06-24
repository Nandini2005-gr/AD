import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

DATA_PATH = r"E:\Alzheimer_Project\Data"

LABELS = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

TARGET_SIZE = (128, 128)


class AlzheimerDataset(Dataset):

    def __init__(self):

        self.samples = []

        for class_name, label in LABELS.items():

            class_path = os.path.join(DATA_PATH, class_name)

            patients = set()

            for file in os.listdir(class_path):

                if "_mpr-1_" in file:

                    patient = "_".join(file.split("_")[:2])

                    patients.add(patient)

            for patient in patients:

                self.samples.append(
                    (class_name, patient, label)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        class_name, patient_id, label = self.samples[idx]

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

            img = cv2.resize(
                img,
                TARGET_SIZE
            )

            img = img.astype(np.float32) / 255.0

            volume.append(img)

        volume = np.array(volume)

        volume = torch.tensor(
            volume,
            dtype=torch.float32
        )

        return volume, label


# dataset = AlzheimerDataset()

# print("Dataset Size:", len(dataset))

# volume, label = dataset[0]

# print("Volume Shape:", volume.shape)

# print("Label:", label)