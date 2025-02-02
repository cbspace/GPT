from utils import *
from model import TransformerBlock, GPTModel
# from data import train_dataset, validation_dataset, test_dataset, format_data
from data import tokenize, decode

import torch 
from torch import nn

model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=0.2)
model.to(device)

# Training
n_epochs = 1

for epoch in range(n_epochs):
    pass

# Do a forward pass on the model
model.eval()
t = tokenize("The cat sat on the mat")
t = torch.tensor(t, dtype=torch.long, device=device).unsqueeze(0)
completion = model.generate(t, 32)
print(completion)
