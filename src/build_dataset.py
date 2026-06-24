import os

DATA_PATH = r"E:\Alzheimer_Project\Data"

LABELS = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

for class_name in LABELS:

    class_path = os.path.join(DATA_PATH, class_name)

    patients = set()

    for file in os.listdir(class_path):

        if "_mpr-1_" in file:

            patient = "_".join(file.split("_")[:2])

            patients.add(patient)

    print(
        class_name,
        "-> Label:",
        LABELS[class_name],
        "| Patients:",
        len(patients)
    )