import torch
import torch.nn as nn
from torch.nn import functional as F


with open('input.txt', 'r', encoding='utf-8') as f: # open the input file
    text = f.read() # read input.txt

print('Lenth of dataset in character: ', len(text)) # check dataset length

# Find every unique character in the entire dataset and put them into a sorted list.
chars = sorted(list(set(text)))

vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Mapping from char to int

# str to int 
# i int     ch character
# enumerate ---> For every character, give it a number. 
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

# Take text and convert every character into its number.
# lambda makes a fucntion name s
# take a str n output list of int
encode = lambda s: [stoi[c] for c in s]
# take a list of int n output a str
decode = lambda l: ''.join([itos [i] for i in l])

test = encode('hii there') # conv string to int
print(test)
print(decode(test)) # conv int to str 

# convert your data into a PyTorch tensor so PyTorch can 
# perform neural-network operations on it.
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9*len(data)) # get 90% of data
train_data = data[:n]  # train first 90% of data
val_data = data[n:] # rest val

#   Given some characters, predict the next character
block_size = 8
train_data[:block_size + 1] #give 8 character and 1 to predict the next character
# print(train_data[:block_size + 1])

# Look at the previous tokens → predict the next token.
x = train_data[:block_size]
y= train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when inpput is {context} the target: {target}")


# sampling different 
torch.manual_seed(1337)
# Take 4 different sequences, where each sequence contains 8 tokens
batch_size = 4 # group of training
block_size = 8 # each seq cont 8 tokens

# Create a batch of input (x) and target (y) sequences
def get_batch(split):
    # If we are creating a training batch use train_data.
    # Otherwise use validation data
    data = train_data if split == 'train' else val_data

    # Give me 4 random starting positions in my dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # torch.stack() combines the 4 sequences into one tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Create the target sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Return the input and target batches.
    return x, y

# get one batch from the training data
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

# Loop through each sequence in the batch.
for b in range(batch_size): # batch dimension
    # Go through each position/time step in the sequence
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")


# Bigram Language Model
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    # each token directly looks up a row in this table.
    # each row contains `vocab_size` values, where each value
    # represents the score (logit) for a possible next token
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    # idx contains the input token IDs.
    # targets contains the correct next-token IDs.
    def forward(self, idx, targets = None):
        B, T = idx.shape

        # look up the logits for every input token
        # embedding table converts (b,t) to (b,t,c)
        logits = self.token_embedding_table(idx) # (Batch, num of token, number ofpossible token in vocab)
        if targets is None:
            loss = None
        else:
            
            B,T,C = logits.shape
            # flatten logits
            logits = logits.view(B*T, C)
            # each target now corresponds to one row of logits
            # flatten targets
            targets= targets.view(B * T)
            # Calculate loss
            #CE produces a single number representing
            # how wrong the model is
            # lower loss = better predictions.
            loss = F.cross_entropy(logits,targets)

        # logits  model predictions
        # loss   how wrong the predictions are
        return logits,loss

    def generate(self, idx, max_new_token):
        # idx contains the tokens we already have.
        for _ in range(max_new_token):
            # only use the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, loss = self(idx)
            # Get predictions for the final position
            logits= logits[:, -1, :]
             # Convert logits to probabilities
            probs = F.softmax(logits, dim = -1)
            # Sample next character
            idx_next = torch.multinomial(probs, num_samples=1) 
             # Add new character to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb) # calling(input, target)
print(logits.shape) # (logits, target, prediction score)
print(loss)
print(decode(m.generate(idx = torch.zeros((1,1),dtype=torch.long), max_new_token=100).tolist()))