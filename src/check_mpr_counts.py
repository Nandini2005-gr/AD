import os
from collections import defaultdict

DATA_PATH = r"E:\Alzheimer_Project\Data"

mpr_counts = defaultdict(set)

for class_name in os.listdir(DATA_PATH):

    class_path = os.path.join(DATA_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):

        if file.endswith(".jpg"):

            patient = "_".join(file.split("_")[:2])

            mpr = file.split("_")[3]

            mpr_counts[patient].add(mpr)

print("Patients:", len(mpr_counts))

distribution = defaultdict(int)

for patient, scans in mpr_counts.items():

    distribution[len(scans)] += 1

print("\nDistribution:\n")

for k in sorted(distribution):
    print(f"{k} MPR scans -> {distribution[k]} patients")