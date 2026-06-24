import os
from collections import defaultdict

DATA_PATH = r"E:\Alzheimer_Project\Data"

patient_slices = defaultdict(int)

for class_name in os.listdir(DATA_PATH):

    class_path = os.path.join(DATA_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):

        if file.endswith(".jpg"):

            patient_id = "_".join(file.split("_")[:2])

            patient_slices[patient_id] += 1

print("=" * 60)
print("SLICE COUNT PER PATIENT")
print("=" * 60)

counts = list(patient_slices.values())

print(f"Total Patients: {len(counts)}")
print(f"Minimum Slices: {min(counts)}")
print(f"Maximum Slices: {max(counts)}")

print("\nFirst 20 Patients:\n")

for patient, count in list(patient_slices.items())[:20]:
    print(f"{patient} -> {count}")