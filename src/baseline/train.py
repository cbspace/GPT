from model import TransformerBlock, GPTModel
# from data import train_dataset, validation_dataset, test_dataset, format_data

import torch 
from torch import nn
from transformers import GPT2Tokenizer

# Model
n_layers = 6
n_heads = 4
embed_dim = 32
ffn_dim = embed_dim*4
n_vocab = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, n_vocab, dropout=0.2)
model.to(device)

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer(s, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Training
n_epochs = 10

for epoch in range(n_epochs):
    pass