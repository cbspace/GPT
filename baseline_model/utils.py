import tiktoken

# Data
max_seq_len = 640
# batch_size = 512
n_vocab = 50257
num_workers = 12

# Model
n_layers = 8
n_heads = 12
embed_dim = 504
ffn_dim = embed_dim*4

# Training
learn_rate = 1e-4
dropout_value = 0.4
minibatch_size = 16
# n_minibatch = batch_size // minibatch_size
n_epochs = 3
n_print = 100
train_temp = 1.4

# Inference
topk_elements = 40
temperature = 1.4

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