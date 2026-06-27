import os
from collections import defaultdict

DATA_PATH = r"C:\Users\User\Documents\Data"

classes = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia"
]

for class_name in classes:

    class_path = os.path.join(DATA_PATH, class_name)

    patient_ids = set()

    for file in os.listdir(class_path):

        if file.endswith(".jpg"):

            patient_id = "_".join(file.split("_")[:2])

            patient_ids.add(patient_id)

    print("\n" + "=" * 60)
    print(class_name)
    print("=" * 60)

    for patient in sorted(patient_ids):
        print(patient)
