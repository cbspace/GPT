from utils import *
from model import TransformerBlock, GPTModel
from data import tokenize, decode, train_loader, validation_loader

import torch 
from torch import nn, optim
from tqdm import tqdm
import sys

model = GPTModel(device, n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=dropout_value)
model = model.to(device)

model = torch.compile(model)
torch.backends.cuda.matmul.allow_tf32 = True

optimiser = optim.AdamW(model.parameters(), lr=learn_rate, betas=(0.9, 0.999), eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=2000, gamma=0.7)
loss_function = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    model.train()
    optimiser.zero_grad()
    for i,sequences in enumerate(train_loader):
        sequences = sequences.to(device)
        input_tokens = sequences[:, :-1] # B, T
        labels = sequences[:, 1:] # B, T

        logits = model(input_tokens) # B, T, V
        train_loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        (train_loss / n_minibatch).backward()

        if i > 0 and not i % (n_minibatch-1):
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step()

        if not i % n_print:
            print(f'Epoch: {epoch+1} Minibatch: {i}/{len(train_loader)} Train Loss: {train_loss.item():.3f}')

    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for i, sequences in enumerate(tqdm(validation_loader)):
            sequences = sequences.to(device)
            input_tokens = sequences[:, :-1]
            labels = sequences[:, 1:]

            logits = model(input_tokens)
            validation_loss += loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).detach()
        
        validation_loss = validation_loss / len(validation_loader)
        print(f'Epoch: {epoch+1} Validation Loss: {validation_loss.item():.3f}')

    model_checkpoint = {'state_dict': model.state_dict()}
    torch.save(model_checkpoint, f'{save_path}/model{epoch}.pkl')
