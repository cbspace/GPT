from utils import *

from datasets import load_dataset
from transformers import GPT2Tokenizer

# HuggingFace smoltalk dataset (~4GB)
# train_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[:70%]', num_proc=num_workers)
# # train_dataset = train_dataset.with_format('torch', device=device)
# # train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

# validation_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[70%:90%]', num_proc=num_workers)
# test_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[-10%:]', num_proc=num_workers)

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

example = [{'content': 'How many positive integers with four digits have a thousands digit of 2?', 'role': 'user'}, 
           {'content': 'Since the thousands digit must be 2, we have only one choice for that digit.\nFor the hundreds digit, we have 10 choices (0-9).\nFor the tens and units digits, we also have 10 choices each.\nTherefore, there are $1 \\times 10 \\times 10 \\times 10 = \\boxed{1000}$ positive integers with four digits that have a thousands digit of 2.\nThe answer is: 1000', 'role': 'assistant'}]

# Process a single training example
def process_input(text_in):
    tokenized = tokenizer(text_in, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
    return tokenized

def format_data(data_in):
    data_str = ''
    for entry in data_in:
        data_str += f"{entry['role'].title()}: {entry['content']}\n"
    return data_str

# View a sample of the dataset
print(format_data(example))

# Tokenize, pad and shift
print(process_input(format_data(example)))