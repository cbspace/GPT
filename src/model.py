# This will be a baseline model that uses standard PyTorch library calls 
# to create a GPT 2 style model. Once this one  is working I will replace 
# elements of the model with my own implementations. I will also compare 
# performance between my model and this baseline throughout.

from utils import *

import torch 
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

class TransformerBlock(nn.Module):
    def __init__(self, device, n_heads, embed_dim, ffn_dim, dropout):
        super().__init__()
        self.device = device
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim),
                                 nn.SiLU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(ffn_dim, embed_dim),
                                 nn.Dropout(p=dropout))

    def forward(self, x):
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        x_in = self.layer_norm1(x)
        x = x + self.self_attention(x_in, x_in, x_in, attn_mask=causal_mask)[0]
        x = x + self.ffn(self.layer_norm2(x))
        return x


class GPTModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, device, n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.transformer = nn.Sequential(*[TransformerBlock(device, n_heads, embed_dim, ffn_dim, dropout) for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, n_vocab, bias=False)
        self.output_projection.weight = self.embedding.weight # Using weight sharing

    def forward(self, input_tokens):
        input_embed = self.embedding(input_tokens)
        positions = torch.arange(0, input_tokens.size(1), device=input_tokens.device).unsqueeze(0)
        input_embed = input_embed + self.positional_embedding(positions)

        x = self.transformer(input_embed)
        x = self.layer_norm(x)
        x = self.output_projection(x)
        return x

    # Generate a completion from the model
    def generate(self, input_ctx, max_length, temperature=1.0, top_p=None, top_k=None):
        assert max_length <= self.max_seq_len

        self.eval()
        context_list = [i for i in input_ctx]
        with torch.no_grad():
            while len(context_list) < max_length:
                context = torch.tensor(context_list, dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self(context)[0,-1,:]

                if top_p:
                    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumulative_probs <= top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = True
                    filtered_probs = sorted_probs * sorted_mask
                    filtered_probs = filtered_probs / filtered_probs.sum()
                    probs_sampled = torch.multinomial(filtered_probs, 1).item()
                    selected_token = sorted_indices[probs_sampled].item()
                elif top_k:
                    scaled_logits = logits / temperature
                    topk_probs, topk_indices = scaled_logits.topk(topk_elements)
                    probs = nn.functional.softmax(topk_probs, dim=-1)
                    probs_sampled = torch.multinomial(probs, 1).item()
                    selected_token = topk_indices[probs_sampled].item()
                else: # Greedy decoding
                    selected_token = logits.argmax(dim=-1).item()

                context_list.append(selected_token)
        return context_list

    def get_model_size(self):
        total = sum(p.numel() for p in self.parameters())
        return f'Model Size: {total/1e6:.1f}M'
