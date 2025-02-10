import torch 
from utils import *
from model import TransformerBlock, GPTModel
from data import tokenize

checkpoint_loaded = torch.load(f'{save_path}/model.pkl')
model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=dropout_value)
model.state_dict = checkpoint_loaded['state_dict']
model.to(device)

# Do some forward passes on the model
prompts = tokenize(["My name is", 
                    "The dog ran away",
                    "I'm a language model",
                    "The capital of Australia is",
                    "A"])

completions = [model.generate(p, 24, greedy=True) for p in prompts]
for sequence in completions:
    print(sequence)
