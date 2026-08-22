import torch
import torch.nn as nn
from torch.nn import functional as F

# Make random results reproducible.
torch.manual_seed(1337)


# Number of sequences processed at once.
batch_size = 64

# Maximum number of tokens the model can
# look at in one sequence.
block_size = 256

# Training iterations.
max_iters = 5000

# How often we will evaluate the model.
eval_interval = 500

# Learning rate for the optimizer.
learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Number of batches used during evaluation.
eval_iters = 200

# Size of token embeddings.
n_embd = 384

# Number of attention heads.
n_head = 6

# Number of Transformer blocks.
n_layer = 6

# Dropout helps reduce overfitting.
dropout = 0.2

# LOAD DATASET

# Open the Tiny Shakespeare dataset.
#
# Make sure input.txt is inside the same
# folder as this Python file.
with open('input.txt', 'r', encoding='utf-8') as f:

    text = f.read()

print("Dataset length:", len(text))
print("First 100 characters:")
print(text[:100])  

# Tokenizer

# Find every unique character in the dataset.
chars = sorted(list(set(text)))

# Count how many unique characters we have.
# This is our vocabulary size.
vocab_size = len(chars)

print("Unique characters:", chars)
print("Vocabulary size:", vocab_size)

# char- int

# Create a dictionary that gives every
# character a unique number.

stoi = {ch: i for i, ch in enumerate(chars)}

# int - char

itos = {i: ch for i, ch in enumerate(chars)}

# Convert text into numbers.

encode = lambda s: [stoi[c] for c in s]

# Convert numbers back into text.

decode = lambda l: ''.join([itos[i] for i in l])

sample = "Hello"

encoded = encode(sample)

print("Original:", sample)
print("Encoded:", encoded)
print("Decoded:", decode(encoded))