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


x = train_data[:block_size]
y= train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when inpput is {context} the target: {target}")