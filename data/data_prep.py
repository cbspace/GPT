import numpy as np
from datasets import load_dataset

import tiktoken
tokenizer = tiktoken.encoding_for_model('gpt2')

# Start with a small simple dataset of sentences (1.7M rows, 90% train, 10% test split)
train_dataset = load_dataset('agentlans/high-quality-english-sentences', 'default', split='train')
valid_dataset = load_dataset('agentlans/high-quality-english-sentences', 'default', split='test')

# Let's see some stats from out dataset
print(f"Training Dataset Length:   {len(train_dataset)/1e6:.3f} M, Max Len: {len(max(train_dataset, key=len)['text'])}")
print(f"Validation Dataset Length: {len(valid_dataset)/1e6:.3f} M, Max Len: {len(max(valid_dataset, key=len)['text'])}")
print('Samples:')

# View samples from the dataset
for s in range(3):
    print(f"{s}: {train_dataset[s]['text']}")

# Tokenize and store in files
def create_token_file(dataset, ds_name):
    all_tokens = []
    for text in dataset:
        for t in tokenizer.encode(text['text']):
            all_tokens.append(t)
    numpy_array = np.array(all_tokens, dtype=np.uint16)
    np.save(ds_name, numpy_array)
    print(f'Saved {len(all_tokens)/1e6:.3f}M tokens to {ds_name}.npy')

create_token_file(train_dataset, 'train_dataset')
create_token_file(valid_dataset, 'validation_dataset')