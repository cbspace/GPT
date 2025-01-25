# This will be a baseline model that uses standard PyTorch library calls 
# to create a GPT 2 style model. Once this one  is working I will replace 
# elements of the model with my own implementations. I will also compare 
# performance between my model and this baseline throughout.

import torch 
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, embed_dim, ffn_dim, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim),
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(ffn_dim, embed_dim),
                                 nn.Dropout(p=dropout))

    def forward(self, x):
        x_in = self.layer_norm1(x)
        x = x + self.self_attention(x_in, x_in, x_in)[0]
        x = x + self.ffn(self.layer_norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, n_layers, n_heads, embed_dim, ffn_dim, dropout):
        super().__init__()

        self.transformer = nn.Sequential(*[TransformerBlock(n_heads, embed_dim, ffn_dim, dropout) for _ in range(n_layers)])
        self.output = nn.Sequential(nn.LayerNorm(embed_dim),
                                    nn.Linear(embed_dim, n_vocab))

    def forward(self, x):
        x = self.transformer(x)
        x = self.output(x)
        return x

n_layers = 6
n_heads = 4
embed_dim = 32
ffn_dim = embed_dim*4
n_vocab = 100

model = GPTModel(n_layers, n_heads, embed_dim, ffn_dim, dropout=0.2)

