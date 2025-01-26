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
        self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        
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
    def __init__(self, n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout):
        super().__init__()

        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.transformer = nn.Sequential(*[TransformerBlock(n_heads, embed_dim, ffn_dim, dropout) for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, n_vocab, bias=False)
        self.output_projection.weight = self.embedding.weight

    def forward(self, input_tokens):
        input_embed = self.embedding(input_tokens)
        positions = torch.arange(0, input_tokens.size(1)).unsqueeze(0)
        input_embed = input_embed + self.positional_embedding(x)

        x = self.transformer(input_embed)
        x = self.layer_norm(x)
        x = self.output_projection(x)
        return x

    def get_model_size(self):
        # print([p.numel() for p in self.parameters()])
        total = sum(p.numel() for p in self.parameters())
        return f'Model Size: {total/1e6:.1f}M'
