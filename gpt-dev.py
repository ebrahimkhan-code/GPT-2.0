import torch


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

