# Store definitions in this file

# Data
max_seq_len = 1024
batch_size = 512
n_vocab = 50257
num_workers = 8

# Model
n_layers = 12
n_heads = 12
embed_dim = 768
ffn_dim = embed_dim*4

# Torch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')