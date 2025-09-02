import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

class EventDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Pad sequences to max length in batch
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_size = max_len - seq.size(0)
        if pad_size > 0:
            padded_seq = torch.cat([seq, torch.zeros(pad_size, seq.size(1))], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    return torch.stack(padded_sequences), torch.tensor(labels)

def get_data_loader(data_path, batch_size=32, shuffle=True):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    dataset = EventDataset(data['sequences'], data['labels'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# Example usage
if __name__ == "__main__":
    loader = get_data_loader("../data/preprocessed_sequences.pkl", batch_size=4)
    for batch in loader:
        print(batch[0].shape, batch[1])  # Test shapes
        break