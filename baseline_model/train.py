from utils import *
from model import TransformerBlock, GPTModel
from data import tokenize, decode, train_loader, validation_loader

import torch 
from torch import nn, optim
from tqdm import tqdm
import sys

model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=dropout_value)
model = model.to(device)

model = torch.compile(model)
torch.backends.cuda.matmul.allow_tf32 = True

optimiser = optim.AdamW(model.parameters(), lr=learn_rate, betas=(0.9, 0.999), eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=10000, gamma=0.8)
loss_function = nn.CrossEntropyLoss(label_smoothing=0.15)

for epoch in range(n_epochs):
    model.train()
    for i,sequences in enumerate(train_loader):
        sequences = sequences.to(device)
        input_tokens = sequences[:, :-1] # B, T
        labels = sequences[:, 1:] # B, T

        logits = model(input_tokens) / train_temp # B, T, V
        loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        print(f'Epoch: {epoch+1} Minibatch: {i}/{len(train_loader)} Train Loss: {loss.item():.3f}')

    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for sequences in tqdm(validation_loader):
            sequences = sequences.to(device)
            input_tokens = sequences[:, :-1]
            labels = sequences[:, 1:]
            logits = model(input_tokens) / train_temp
            validation_loss += loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).detach()
        
        validation_loss = validation_loss / (len(validation_loader))
        print(f'Epoch: {epoch+1} Validation Loss: {validation_loss.item():.3f}')

    model_checkpoint = {'state_dict': model.state_dict()}
    torch.save(model_checkpoint, f'{save_path}/model.pkl')
