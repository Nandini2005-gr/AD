import os

DATA_PATH = r"E:\Alzheimer_Project\Data"

target_patient = "OAS1_0308"

for class_name in os.listdir(DATA_PATH):

    class_path = os.path.join(DATA_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"\nChecking {class_name}")

    for file in sorted(os.listdir(class_path)):

        if file.startswith(target_patient):

            print(file)