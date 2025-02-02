from utils import *

from datasets import load_dataset
import tiktoken

# Start with a small simple dataset of sentences (1.7M rows)
# train_dataset = load_dataset('agentlans/high-quality-english-sentences', 'default', split='train')
# test_dataset = load_dataset('agentlans/high-quality-english-sentences', 'default', split='test')

# Tokenizer
tokenizer = tiktoken.encoding_for_model('gpt2')

def tokenize(text_in):
    #tokenized = tokenizer(text_in, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
    tokenized = tokenizer.encode(text_in)
    return tokenized

def decode(tokens_in):
    tokenized = tokenizer.decode_single_token_bytes(tokens_in)
    return tokenized

# Process a single training example using sliding window
def process_input(text_in, max):
    inputs, labels = [], []
    tokenized = tokenizer.encode(text_in)
    for i in range(len(tokenized)-1):
        inputs.append(tokenized[i:])
        labels.append(tokenized[i-1:])
    return [inputs, labels]

# View a sample of the dataset
# print(train_dataset[0])

