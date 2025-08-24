from utils import *

import numpy as np
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, data_tokens, seq_len):
        super().__init__()
        self.data_tokens = data_tokens
        self.out_len = seq_len + 1

        in_length = len(data_tokens)
        trim = in_length % (self.out_len * batch_size)
        if trim:
            self.data_tokens = self.data_tokens[:-trim]
        self.n_sequences = len(self.data_tokens) // (self.out_len)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.out_len
        end = start + self.out_len
        seq = self.data_tokens[start:end]
        return torch.from_numpy(seq).long()


train_data_tokens = np.memmap('../data/train_dataset_fine_8M.tokens', dtype=np.uint16, mode='r+')
train_dataset = GPTDataset(train_data_tokens, max_seq_len)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=minibatch_size, num_workers=num_workers, persistent_workers=True, drop_last=True)

validation_data_tokens = np.memmap('../data/validation_dataset_fine_8M.tokens', dtype=np.uint16, mode='r+')
validation_dataset = GPTDataset(validation_data_tokens, max_seq_len)
validation_loader = DataLoader(validation_dataset, shuffle=True, batch_size=minibatch_size, num_workers=num_workers, persistent_workers=True, drop_last=True)
