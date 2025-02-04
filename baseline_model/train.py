from utils import *
from model import TransformerBlock, GPTModel
from data import tokenize, decode, train_loader, validation_loader

import torch 
from torch import nn

model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=0.2)
model.to(device)

# Training
n_epochs = 1

for epoch in range(n_epochs):
    for sequences in train_loader:
        sequences = sequences.to(device)
        input_tokens = sequences[:, 1:]
        labels = sequences[:, :-1]
    pass

# Do some forward passes on the model
model.eval()
prompts = tokenize(["The cat sat on the mat", "The dog ran away"])
completions = [model.generate(p, 24) for p in prompts]
for sequence in completions:
    print(sequence)
