from utils import *

import numpy as np
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, data_tokens, seq_len):
        super().__init__()
        in_length = len(data_tokens)
        trim = in_length % (seq_len + 1)
        self.sequences = torch.tensor(data_tokens, dtype=torch.long)[:-trim].view(-1, seq_len + 1)

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx):
        return self.sequences[idx]


train_data_tokens = np.load('../data/train_dataset_fine_medium.npy', mmap_mode='r')
train_dataset = GPTDataset(train_data_tokens, max_seq_len)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=minibatch_size, num_workers=num_workers, persistent_workers=True, drop_last=True)

validation_data_tokens = np.load('../data/validation_dataset_fine_medium.npy', mmap_mode='r')
validation_dataset = GPTDataset(validation_data_tokens, max_seq_len)
validation_loader = DataLoader(validation_dataset, shuffle=True, batch_size=minibatch_size, num_workers=num_workers, persistent_workers=True, drop_last=True)
