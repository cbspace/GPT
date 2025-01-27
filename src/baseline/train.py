from utils import *
from model import TransformerBlock, GPTModel
# from data import train_dataset, validation_dataset, test_dataset, format_data

import torch 
from torch import nn

model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=0.2)
model.to(device)
print(model.get_model_size())

# Training
n_epochs = 10

for epoch in range(n_epochs):
    pass
