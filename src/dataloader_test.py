from torch.utils.data import DataLoader
from dataset_class import AlzheimerDataset

dataset = AlzheimerDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for volumes, labels in loader:

    print("Volume Batch Shape:", volumes.shape)
    print("Labels:", labels)

    break