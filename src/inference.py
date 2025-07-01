import torch 
from utils import *
from model import TransformerBlock, GPTModel
from data import tokenize

model = GPTModel(device, n_layers, n_heads, embed_dim, ffn_dim, n_vocab, max_seq_len, dropout=dropout_value)
model.eval()

# Load checkpoint and remove prefix to keys that was added by torch.compile
checkpoint_loaded = torch.load(f'{save_path}/model.pkl')
state_dict = {k.replace('_orig_mod.',''):v for k,v in checkpoint_loaded['state_dict'].items()}

model.load_state_dict(state_dict)
model.to(device)

# Do some forward passes on the model
prompts = tokenize(["My name is", 
                    "It's important to note that",
                    "I'm a language model",
                    "It's always a good idea to",
                    "A"])

# completions = [model.generate(p, 42, temperature=temperature, top_p=top_p_value) for p in prompts]
# for sequence in completions:
#     print(decode(sequence))

# Yield test
update_interval = 10
def generate(prompt,out_tokens,temperature,top_k_value):
    model.to(device)
    outputs = tokenizer.encode(prompt)
    tokens_remaining = int(out_tokens)
    out_text = prompt
    yield out_text

    while tokens_remaining:
        new_inputs_len = update_interval if tokens_remaining >= update_interval else tokens_remaining % update_interval
        outputs = model.generate(outputs, len(outputs)+new_inputs_len, temperature, top_k=top_k_value)
        tokens_remaining -= new_inputs_len
        out_text += tokenizer.decode(outputs[-new_inputs_len:])
        yield out_text

fn_outputs = list(generate(input("Prompt: "), out_tokens=200, temperature=temperature, top_k_value=topk_elements))
for i, output in enumerate(fn_outputs):
    if i==len(fn_outputs)-1:
        print(output)