import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import yaml
from pathlib import Path

# Load configuration
config_path = Path('02_Adverserial_Network/250902-config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config.get('BATCH_SIZE')
MAX_SEQ_LENGTH = config['DATA']['MAX_SEQ_LENGTH']

class EventDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def pad_or_truncate(seq, target_len):
    seq_len, feature_dim = seq.size()
    if seq_len > target_len:
        seq = seq[:target_len]
        mask = torch.ones(target_len, dtype=torch.bool)
    else:
        pad_size = target_len - seq_len
        pad = torch.zeros(pad_size, feature_dim, dtype=seq.dtype)
        seq = torch.cat([seq, pad], dim=0)
        mask = torch.cat([torch.ones(seq_len, dtype=torch.bool), torch.zeros(pad_size, dtype=torch.bool)])
    return seq, mask

def collate_fn(batch):
    sequences, labels = zip(*batch)
    batch_max_len = min(max(seq.size(0) for seq in sequences), MAX_SEQ_LENGTH)
    padded_sequences = []
    attention_masks = []
    for seq in sequences:
        seq, mask = pad_or_truncate(seq, batch_max_len)
        padded_sequences.append(seq)
        attention_masks.append(mask)
    return (
        torch.stack(padded_sequences).to(torch.float16),           # [batch_size, batch_max_len, feature_dim]
        torch.tensor(labels, dtype=torch.long),                    # [batch_size]
        torch.stack(attention_masks).to(torch.bool)                # [batch_size, batch_max_len]
    )

def get_data_loader(data_path, batch_size=BATCH_SIZE, shuffle=True):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    dataset = EventDataset(data['sequences'], data['labels'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3, collate_fn=collate_fn)

# Example usage
if __name__ == "__main__":
    loader = get_data_loader("data/preprocessed_sequences.pkl", batch_size=BATCH_SIZE)
    for batch in loader:
        print(batch[0].shape, batch[1])  # Test shapes
        break