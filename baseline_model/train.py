from utils import *
from model import TransformerBlock, GPTModel
from data import tokenize, decode, train_loader, validation_loader

import torch 
from torch import nn, optim
from tqdm import tqdm
import sys

model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=dropout_value)
model.to(device)

def loss_function(probs, labels):
    label_mask = torch.nn.functional.one_hot(labels, num_classes=n_vocab).float()
    return torch.nn.functional.cross_entropy(probs, label_mask)

optimiser = optim.AdamW(model.parameters(), lr=6e-4)
n_epochs = 2
n_print = 100

for epoch in range(n_epochs):
    model.train()
    for i,sequences in enumerate(tqdm(train_loader)):
        sequences = sequences.to(device)
        input_tokens = sequences[:, 1:]
        labels = sequences[:, :-1]

        logits = model(input_tokens)
        probs = nn.functional.softmax(logits[:,-1,:] / temperature, dim=1)
        train_loss = loss_function(probs, labels[:,-1])
        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()

        if (i+1) % n_print == 0:
            print(f'Epoch: {epoch+1} Train Loss: {train_loss.item():.3f}')

    model.eval()
    with torch.no_grad():
        for sequences in tqdm(validation_loader):
            sequences = sequences.to(device)
            input_tokens = sequences[:, 1:]
            labels = sequences[:, :-1]

            logits = model(input_tokens)
            probs = nn.functional.softmax(logits[:,-1,:] / temperature, dim=1)
            validation_loss = loss_function(probs, labels[:,-1])
        print(f'Epoch: {epoch+1} Validation Loss: {validation_loss.item():.3f}')

model_checkpoint = {'state_dict': model.state_dict()}
torch.save(model_checkpoint, f'{save_path}/model.pkl')
