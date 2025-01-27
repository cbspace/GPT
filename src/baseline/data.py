from utils import *

from datasets import load_dataset
from transformers import GPT2Tokenizer

# Start with a small simple dataset of sentences (1.7M rows)
train_dataset = load_dataset('agentlans/high-quality-english-sentences', 'default', split='train')
test_dataset = load_dataset('agentlans/high-quality-english-sentences', 'default', split='test')

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Process a single training example
def process_input(text_in):
    tokenized = tokenizer(text_in, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
    return tokenized

# View a sample of the dataset
print(train_dataset[0])

# Tokenize
print(process_input(train_dataset[0]['text']))

