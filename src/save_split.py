import os
import json
from sklearn.model_selection import train_test_split

DATA_PATH = r"C:\Users\User\Documents\Data"

LABELS = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

patients = []
labels = []

for class_name, label in LABELS.items():

    class_path = os.path.join(DATA_PATH, class_name)

    patient_set = set()

    for file in os.listdir(class_path):

        if "_mpr-1_" in file:

            patient = "_".join(file.split("_")[:2])

            patient_set.add(patient)

    for patient in patient_set:

        patients.append(patient)
        labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(
    patients,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

data = {
    "train_patients": X_train,
    "test_patients": X_test
}

with open("split.json", "w") as f:
    json.dump(data, f)

print("Train:", len(X_train))
print("Test :", len(X_test))
print("split.json saved")
