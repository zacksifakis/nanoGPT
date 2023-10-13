import os
import pickle
import requests
import numpy as np

with open('data/my_dataset/input.txt', 'r') as f:
    gatsby = f.read()
print(f"length of dataset in characters: {len(gatsby):,}")

with open('data/shakespeare_char/input.txt', 'r') as f:
    shakespeare = f.read()
print(f"length of dataset in characters: {len(shakespeare):,}")

new_shakespeare = set(shakespeare)
new_shakespeare.add('@')
chars_shakespeare = sorted(list(new_shakespeare))

# get all the unique characters that occur in this text
chars_gatsby = sorted(list(set(gatsby)))

# replace all characters in chars_gatsby not in chars_shakespeare with '@'
# new_chars_gatsby = []
# for char in chars_gatsby:
#     if char in chars_shakespeare:
#         new_chars_gatsby.append(char)
#     else:
#         new_chars_gatsby.append('@')
# chars_gatsby = new_chars_gatsby

#new_data = ''
for char in chars_gatsby:
    if char not in new_shakespeare:
        replace_char = '@'
        gatsby = gatsby.replace(char, replace_char)
    else:
        continue

chars_gatsby = sorted(list(set(gatsby)))
vocab_size = len(chars_gatsby)
print("all the unique characters:", ''.join(chars_gatsby))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars_gatsby) }
itos = { i:ch for i,ch in enumerate(chars_gatsby) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = 1*len(gatsby)
gatsby = gatsby[:int(n)]
#TODO: Change these to vary length of train_data
train_data = gatsby[:int(n*0.9)]
val_data = gatsby[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens