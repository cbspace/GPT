from tokenizer import create_vocab, tokenize
from embedding import word2vec

import torch
import torch.nn.functional as F
from torch import nn, optim

training_text = "The quick brown fox jumps over the lazy dog."

# Create vocab and print
vocab = create_vocab(training_text)
print("Vocab:", vocab)

# Let's try out the tokenizer!
test_case = 'The quick fox.'
tokens = tokenize(test_case.split(), vocab)
print(f'Test Case: "{test_case}" = {tokens}')

# Let's try out the embedding function
print(word2vec(tokens))