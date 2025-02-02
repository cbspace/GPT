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

# PyTorch device
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else: device = torch.device('cpu')