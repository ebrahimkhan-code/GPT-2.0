with open('Gpt-2.0/input.txt', 'r', encoding='utf-8') as f: # open the input file
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

test = encode('hii there')
print(test)
print(decode(test))