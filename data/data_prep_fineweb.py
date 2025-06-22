import numpy as np
from datasets import load_dataset

import tiktoken
tokenizer = tiktoken.encoding_for_model('gpt2')
sep_token = 50256

# 43M rows total - Using 4.3M train, 0.4M validation
train_dataset = load_dataset('skymizer/fineweb-edu-dedup-45B', 'default', split='train[:1000000]')
valid_dataset = load_dataset('skymizer/fineweb-edu-dedup-45B', 'default', split='train[1000000:1050000]')

# Let's see some stats from out dataset
print(f"Training Dataset Length:   {len(train_dataset)/1e6:.3f} M, Max Len: {len(max(train_dataset, key=len)['text'])}")
print(f"Validation Dataset Length: {len(valid_dataset)/1e6:.3f} M, Max Len: {len(max(valid_dataset, key=len)['text'])}")
print('Samples:')

# View samples from the dataset
for s in range(1):
    print(f"{s}: {train_dataset[s]['text']}")

# Tokenize and store in files
def create_token_file(dataset, ds_name):
    all_tokens = []
    for text in dataset:
        for t in tokenizer.encode(text['text']):
            all_tokens.append(t)
        # all_tokens.append(sep_token)
    numpy_array = np.array(all_tokens, dtype=np.uint16)
    np.save(ds_name, numpy_array)
    print(f'Saved {len(all_tokens)/1e6:.3f}M tokens to {ds_name}.npy')

create_token_file(train_dataset, 'train_dataset_fine_1M')
create_token_file(valid_dataset, 'validation_dataset_fine_1M')