# Implementation of Word2Vec Algorithm to create embedding

import torch
import torch.nn.functional as F
from torch import nn, optim

# Input: Vector of tokens
# Output: Embedding vector of size [len(tokens), len(hidden_size[-1])]
def word2vec(tokens):

    hidden_size = [6, 5]
    model = nn.Sequential(nn.Linear(len(tokens), hidden_size[0]),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.Linear(hidden_size[0], hidden_size[1]),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.Linear(hidden_size[1], len(tokens)),
                          nn.LogSoftmax(dim=1))

    # Embedding magic here

    return model[6].weight
