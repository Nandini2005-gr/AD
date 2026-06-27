from sklearn.utils.class_weight import compute_class_weight
import numpy as np

labels = (
    [0] * 265 +
    [1] * 58 +
    [2] * 21 +
    [3] * 2
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

print("Class Weights:")
print(weights)
