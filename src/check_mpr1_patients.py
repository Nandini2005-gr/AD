import os

DATA_PATH = r"C:\Users\User\Documents\Data"

patients = set()

for class_name in os.listdir(DATA_PATH):

    class_path = os.path.join(DATA_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):

        if "_mpr-1_" in file:

            patient = "_".join(file.split("_")[:2])

            patients.add(patient)

print("Patients with mpr-1:", len(patients))
