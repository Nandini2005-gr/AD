import os
from collections import defaultdict

DATA_PATH = r"C:\Users\User\Documents\Data"

patient_scans = defaultdict(set)

for class_name in os.listdir(DATA_PATH):

    class_path = os.path.join(DATA_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):

        if file.endswith(".jpg"):

            patient = "_".join(file.split("_")[:2])

            mpr = file.split("_")[3]

            patient_scans[patient].add(mpr)

for patient, scans in patient_scans.items():

    if "mpr-1" not in scans:

        print(patient, scans)
