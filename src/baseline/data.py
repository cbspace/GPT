from datasets import load_dataset
import pprint

# HuggingFace smoltalk dataset (~4GB)
train_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[:70%]', num_proc=9)
validation_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[70%:90%]', num_proc=9)
test_dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train[-10%:]', num_proc=9)

# Wikipedia dataset (~72GB) [Future Use]
#dataset = load_dataset('wikimedia/wikipedia', '20231101.en')

# View a sample of the dataset
pprint.pp(train_dataset[100])