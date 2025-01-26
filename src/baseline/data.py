from datasets import load_dataset

# HuggingFace smoltalk dataset (~4GB)
dataset = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train')

# Wikipedia dataset (~72GB)
#dataset = load_dataset('wikimedia/wikipedia', '20231101.en')