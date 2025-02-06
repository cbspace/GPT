import tiktoken

# Data
max_seq_len = 64
batch_size = 192 #512
n_vocab = 50257
num_workers = 8

# Model
n_layers = 12
n_heads = 12
embed_dim = 768
ffn_dim = embed_dim*4
temperature = 10.0

# Config
save_path = './checkpoints'

# PyTorch device
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else: device = torch.device('cpu')

# Tokenization
tokenizer = tiktoken.encoding_for_model('gpt2')

def tokenize(text_in):
    tokenized = [tokenizer.encode(sequence) for sequence in text_in]
    return tokenized

def decode(tokens_in):
    tokenized = tokenizer.decode(tokens_in)
    return tokenized