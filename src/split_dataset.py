import os
from sklearn.model_selection import train_test_split

DATA_PATH = r"E:\Alzheimer_Project\Data"

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

print("Training Patients:", len(X_train))
print("Testing Patients :", len(X_test))

print("\nClass Distribution (Train)")
for label in sorted(set(y_train)):
    print(label, "->", y_train.count(label))

print("\nClass Distribution (Test)")
for label in sorted(set(y_test)):
    print(label, "->", y_test.count(label))