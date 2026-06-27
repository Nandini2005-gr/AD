import sys
import os
import time
import random

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score

from dataset_class import AlzheimerDataset
from models.vision_mamba import VisionMamba3D

# --- reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = AlzheimerDataset(mode="train")
test_dataset = AlzheimerDataset(mode="test")

# --- Weighted sampler ---
# Loss weighting alone (the `weights` tensor below) only changes how much
# each mistake "costs". It does NOT change what the model sees each batch.
# With Moderate having 1 train patient and Non Demented having 211, most
# batches never contain a minority-class example at all. The sampler fixes
# this by drawing patients so minority classes show up far more often than
# their raw count would suggest.
train_labels = [label for _, _, label in train_dataset.samples]
class_counts = np.bincount(train_labels, minlength=4)
print(f"Train patient counts per class: {class_counts.tolist()}")

# inverse-frequency sample weight per individual sample
sample_weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=sampler  # replaces shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

model = VisionMamba3D().to(device)

# Class weights kept, but lighter on Moderate than before (43.25 -> capped).
# Moderate has only 1 train / 1 test patient, so a very large weight just
# makes training unstable (one example dominates the gradient) without
# adding real signal. We keep some weight so it isn't totally ignored,
# but stop relying on the loss function to do the heavy lifting -- the
# sampler above is now doing most of that work.
weights = torch.tensor(
    [0.3264, 1.4913, 4.1190, 10.0], dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

# Lowered from 0.001 -> 0.0002. The previous LR combined with the 43x
# class weight was almost certainly the cause of the epoch-to-epoch
# accuracy collapse (0.79 -> 0.21 -> 0.76 swings) seen in earlier runs.
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0002, weight_decay=1e-4
)

EPOCHS = 30  # was 15; T_max now matches so the LR actually decays fully

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

best_val_f1 = 0.0
os.makedirs("../checkpoints", exist_ok=True)

CLASS_NAMES = ["Non Demented", "Very Mild", "Mild", "Moderate"]

# Moderate (index 3) has only 1 test patient -- its precision/recall is
# not statistically meaningful. We still train on it and still print it
# in the full report, but model-selection ("best" checkpoint) is based
# on macro-F1 over the 3 classes that have enough data to be learnable.
LEARNABLE_CLASS_INDICES = [0, 1, 2]


def evaluate(loader, print_report=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.unsqueeze(1).to(device)
            labels = labels.to(device)
            outputs = model(volumes)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    # macro-F1 restricted to the 3 learnable classes
    f1_per_class = f1_score(
        all_labels, all_preds,
        labels=LEARNABLE_CLASS_INDICES,
        average=None,
        zero_division=0
    )
    macro_f1_learnable = float(np.mean(f1_per_class))

    if print_report:
        print(classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES,
            labels=[0, 1, 2, 3],
            zero_division=0
        ))
        print(f"Macro-F1 over Non/Very Mild/Mild only: {macro_f1_learnable:.4f}")
        print("(Moderate excluded from this score -- only 1 test patient, "
              "not statistically meaningful)")

    return acc, macro_f1_learnable


for epoch in range(EPOCHS):
    epoch_start = time.time()

    model.train()
    running_loss = 0.0

    for volumes, labels in train_loader:
        volumes = volumes.unsqueeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    avg_loss = running_loss / len(train_loader)

    # Print full report every 5 epochs so you can watch Mild/Very Mild
    # recall move over time, not just the final accuracy number
    show_report = (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS
    val_acc, val_f1 = evaluate(test_loader, print_report=show_report)

    epoch_time = time.time() - epoch_start

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss = {avg_loss:.4f} | "
          f"Val Acc = {val_acc:.4f} | Val Macro-F1(3-class) = {val_f1:.4f} | "
          f"Time = {epoch_time:.1f}s")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "../checkpoints/vision_mamba_best.pth")
        print(f"  -> New best model saved (macro-F1={val_f1:.4f}, acc={val_acc:.4f})")

print(f"\nTraining Finished. Best Val Macro-F1 (3-class) = {best_val_f1:.4f}")

print("\n=== Final Classification Report ===")
evaluate(test_loader, print_report=True)

torch.save(model.state_dict(), "../checkpoints/vision_mamba_final.pth")
print("Final model also saved")