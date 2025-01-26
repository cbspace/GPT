from datasets import load_dataset

# HuggingFace smoltalk dataset (~4GB)
train_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[:70%]', num_proc=9)
validation_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[70%:90%]', num_proc=9)
test_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[-10%:]', num_proc=9)

# Wikipedia dataset (~72GB) [Future Use]
#dataset = load_dataset('wikimedia/wikipedia', '20231101.en')

def format_data(data_in):
    data_str = ''
    for entry in data_in:
        data_str += f"{entry['role'].title()}: {entry['content']}\n"
    return data_str

# View a sample of the dataset
print(format_data(train_dataset[100]['messages']))