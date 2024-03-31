# A Simple tokenizer that separates tokens by spaces and
# Allows the special character "."

training_text = "The quick brown fox jumps over the lazy dog."

# Take input text and create a vocabulary
# Input: A string containing entire vocabulary
# Output: Vocab list
def create_vocab(input_text):
    # Create the vocab list and insert special characters
    # Numbers and other special characters not yet supported
    vocab = ['.']
    for w in input_text.split():
        if w.isalpha():
            vocab.append(w)
        else:
            vocab.append(''.join([c for c in w if c.isalpha()]))
    return vocab

# Input: A list of strings to tokenize
# Output: A list of tokens
# Assumes any "." characters are at the end of the word
def tokenize(input_strings):
    tokenized = []

    for s in input_strings:
        if s[-1] == '.':
            # Append word token followed by "." token
            tokenized.append(vocab.index(s[:-1]))
            tokenized.append(0)
        else:
            tokenized.append(vocab.index(s))

    return tokenized

# Create vocab and print
vocab = create_vocab(training_text)
print("Vocab:", vocab)

# Let's try out the tokenizer!
test_case = 'The quick fox.'
tokens = tokenize(test_case.split())
print(f'Test Case: "{test_case}" = {tokens}')