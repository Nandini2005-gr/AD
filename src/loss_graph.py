import matplotlib.pyplot as plt

cnn_loss = [
    151.3173,
    139.1609,
    131.6448
]

mamba_loss = [
    159.3627,
    167.7618,
    156.4182
]

epochs = [1, 2, 3]

plt.plot(
    epochs,
    cnn_loss,
    marker="o",
    label="3D CNN"
)

plt.plot(
    epochs,
    mamba_loss,
    marker="o",
    label="Vision Mamba"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training Loss Comparison"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "loss_comparison.png"
)

plt.show()

print("Graph Saved")
