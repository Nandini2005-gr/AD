import os
import json

DATA_PATH = r"C:\Users\User\Documents\Data"

LABELS = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

with open("split.json", "r") as f:
    split_data = json.load(f)

train_patients = set(split_data["train_patients"])
test_patients = set(split_data["test_patients"])

print(f"Total train patients (all classes combined): {len(train_patients)}")
print(f"Total test patients (all classes combined): {len(test_patients)}")
print()

for class_name in LABELS:
    class_path = os.path.join(DATA_PATH, class_name)

    if not os.path.isdir(class_path):
        print(f"{class_name}: folder not found at {class_path}")
        continue

    patients_in_class = set()
    for file in os.listdir(class_path):
        if "_mpr-1_" in file:
            patient = "_".join(file.split("_")[:2])
            patients_in_class.add(patient)

    train_count = len(patients_in_class & train_patients)
    test_count = len(patients_in_class & test_patients)
    total_count = len(patients_in_class)

    print(f"{class_name}:")
    print(f"  total patients in folder : {total_count}")
    print(f"  in train split           : {train_count}")
    print(f"  in test split            : {test_count}")
    print()