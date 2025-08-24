import numpy as np
from datasets import load_dataset

import tiktoken
tokenizer = tiktoken.encoding_for_model('gpt2')
sep_token = 50256

# 43M rows total - Using 8M train, 80k validation
train_dataset = load_dataset('skymizer/fineweb-edu-dedup-45B', 'default', split='train[:8000000]')
valid_dataset = load_dataset('skymizer/fineweb-edu-dedup-45B', 'default', split='train[8000000:8080000]')

# Let's see some stats from out dataset
print(f"Training Dataset Length:   {len(train_dataset)/1e6:.3f} M, Max Len: {len(max(train_dataset, key=len)['text'])}")
print(f"Validation Dataset Length: {len(valid_dataset)/1e6:.3f} M, Max Len: {len(max(valid_dataset, key=len)['text'])}")
print('Samples:')

# View samples from the dataset
for s in range(1):
    print(f"{s}: {train_dataset[s]['text']}")

# Tokenize and store in files
def create_token_file(dataset, ds_name):
    file = open(f'{ds_name}.tokens', 'ab')

    total_tokens = 0
    buffer = []
    for i,text in enumerate(dataset):
        for t in tokenizer.encode(text['text']):
            buffer.append(t)
            if i and not (i+1) % 80_000:
                numpy_array = np.array(buffer, dtype=np.uint16)
                numpy_array.tofile(file)
                total_tokens += len(numpy_array)
                buffer = []
        # all_tokens.append(sep_token)

    file.close()
    print(f'Saved {total_tokens/1e6:.3f}M tokens to {ds_name}.tokens')

# create_token_file(train_dataset, 'train_dataset_fine_8M')
create_token_file(valid_dataset, 'validation_dataset_fine_8M')