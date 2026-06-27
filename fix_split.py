import json

with open("split.json") as f:
    data = json.load(f)

# move one Moderate Dementia patient to test
patient_to_move = "OAS1_0351"
data["train_patients"].remove(patient_to_move)
data["test_patients"].append(patient_to_move)

with open("split.json", "w") as f:
    json.dump(data, f, indent=2)

print("Moved", patient_to_move, "to test set")