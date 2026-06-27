import os
import cv2
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset

DATA_PATH = r"C:\Users\User\Documents\Data"

LABELS = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

TARGET_SIZE = (48, 48)
TARGET_DEPTH = 61


class AlzheimerDataset(Dataset):

    def __init__(self, mode="train"):

        self.mode = mode
        self.samples = []

        with open(os.path.join(os.path.dirname(__file__), "..", "split.json"), "r") as f:
            split_data = json.load(f)

        if mode == "train":
            allowed_patients = set(split_data["train_patients"])
        else:
            allowed_patients = set(split_data["test_patients"])

        for class_name, label in LABELS.items():

            class_path = os.path.join(DATA_PATH, class_name)

            patients = set()

            for file in os.listdir(class_path):

                if "_mpr-1_" in file:

                    patient = "_".join(file.split("_")[:2])

                    if patient in allowed_patients:
                        patients.add(patient)

            for patient in patients:
                self.samples.append((class_name, patient, label))

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, class_name, patient_id):

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

            img = cv2.resize(img, TARGET_SIZE)

            img = img.astype(np.float32) / 255.0

            volume.append(img)

        volume = np.array(volume, dtype=np.float32)

        return volume

    def _fix_depth(self, volume):

        current_depth = volume.shape[0]

        if current_depth == TARGET_DEPTH:
            return volume

        if current_depth > TARGET_DEPTH:
            return volume[:TARGET_DEPTH]

        pad_amount = TARGET_DEPTH - current_depth
        pad_shape = (pad_amount, volume.shape[1], volume.shape[2])
        padding = np.zeros(pad_shape, dtype=np.float32)

        return np.concatenate([volume, padding], axis=0)

    def _augment(self, volume):

        if random.random() > 0.5:
            volume = np.flip(volume, axis=2).copy()

        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = volume.shape[1], volume.shape[2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            rotated = np.empty_like(volume)
            for i in range(volume.shape[0]):
                rotated[i] = cv2.warpAffine(
                    volume[i], rot_matrix, (w, h),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                )
            volume = rotated

        if random.random() > 0.5:
            brightness = random.uniform(-0.05, 0.05)
            contrast = random.uniform(0.9, 1.1)
            volume = np.clip(volume * contrast + brightness, 0.0, 1.0)

        return volume

    def __getitem__(self, idx):

        class_name, patient_id, label = self.samples[idx]

        volume = self._load_volume(class_name, patient_id)
        volume = self._fix_depth(volume)

        if self.mode == "train":
            volume = self._augment(volume)

        volume = torch.tensor(volume, dtype=torch.float32)

        return volume, label
