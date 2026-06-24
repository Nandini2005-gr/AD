import os
from collections import defaultdict

DATA_PATH = r"E:\Alzheimer_Project\Data"

LABELS = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

patients = []

for class_name, label in LABELS.items():

    class_path = os.path.join(DATA_PATH, class_name)

    patient_set = set()

    for file in os.listdir(class_path):

        if "_mpr-1_" in file:

            patient = "_".join(file.split("_")[:2])

            patient_set.add(patient)

    for patient in patient_set:

        patients.append((patient, label))

print("Total Patients:", len(patients))

for i in range(10):

    print(patients[i])