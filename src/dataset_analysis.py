import os
from collections import defaultdict

DATA_PATH = r"E:\Alzheimer_Project\Data"

classes = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia"
]

for class_name in classes:

    class_path = os.path.join(DATA_PATH, class_name)

    patient_ids = set()
    total_images = 0

    for file in os.listdir(class_path):

        if file.endswith(".jpg"):

            total_images += 1

            patient_id = "_".join(file.split("_")[:2])

            patient_ids.add(patient_id)

    print("=" * 50)
    print(f"Class: {class_name}")
    print(f"Total Images: {total_images}")
    print(f"Unique Patients: {len(patient_ids)}")