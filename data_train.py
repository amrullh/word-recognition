import pandas as pd
import dataset
import config

from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader
import torch

df = pd.read_csv("labels_with_path.csv")
image_paths = df["filepath"].tolist()
targets = df["label"].tolist()

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, targets, test_size=0.2, random_state=42
)
print(f"Train size: {len(train_paths)}, Val size: {len(val_paths)}")

"""dataset = dataset.dataset(image_path=image_paths, targets=targets)
sample_data = dataset[90]
print(len (dataset))
print(sample_data['images'].shape)
print(sample_data['targets'])
print(sample_data['target_length'])"""

train_dataset = dataset.dataset(image_path=train_paths, targets=train_labels, resize=(300, 80))
val_dataset = dataset.dataset(image_path=val_paths, targets=val_labels, resize=(300, 80))
print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

sample_data = train_dataset[90]
print(sample_data['images'].shape)
print(sample_data['targets'])
print(sample_data['target_length'])

def ctc_collate_(batch):
    images = torch.stack([b['images']for b in batch])
    targets = [b['targets'] for b in batch]
    targets_lengths = torch.tensor([b['target_length'] for b in batch], dtype=torch.long)
    targets_padded = pad_sequence(targets, batch_first = true, padding_value = 0)
    return{
        "images":images,
        "targets":targets_padded,
        "targets_lengths":targets_lengths
    }

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers = config.NUM_WORKERS,
    collate_fn=ctc_collate_
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers = config.NUM_WORKERS,
    collate_fn=ctc_collate_
)