import torch 
from utils import *
from model import TransformerBlock, GPTModel
from data import tokenize

model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=dropout_value)
model.eval()

# Load checkpoint and remove prefix to keys that was added by torch.compile
checkpoint_loaded = torch.load(f'{save_path}/model.pkl')
state_dict = {k.replace('_orig_mod.',''):v for k,v in checkpoint_loaded['state_dict'].items()}

model.load_state_dict(state_dict)
model.to(device)

# Do some forward passes on the model
prompts = tokenize(["My name is", 
                    "It's important to note that although the sky is blue it can sometimes be",
                    "I'm a language model and I am going to fully leverage this input context to tell you that",
                    "It's always a good idea to using longer context inputs when running LLM inference as",
                    "A"])

completions = [model.generate(p, 64, greedy=False) for p in prompts]
for sequence in completions:
    print(sequence)
