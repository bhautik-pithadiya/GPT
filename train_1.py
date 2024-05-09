import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import pandas as pd

from tokenizer.gpt import GptTokenizer
import concurrent.futures

tokenizer_path = './tokenizer/models/gpt.model'
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 100
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 6
n_layer = 6
dropout = 0.2
checkpoint_steps = 500
vocab_size = 10000


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

print("Loading Model")
model = GPTLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print('Loading Dataset')
DATA = pd.read_csv('./data/dataset_v1.csv')

def get_batch(split,counter,batch_size):
    
    data = TRAIN_DATA if split == 'train' else VAL_DATA
    x = torch.stack([data['X'].iloc[i]] for i in range(counter,counter+batch_size))
    y = torch.stack([data['y'].iloc[i]] for i in range(counter,counter+batch_size))
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(model,eval_iters):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        
        for k in tqdm(range(eval_iters)):
            X,y = get_batch(split)
            
            logits , loss = model(X,y)
            del X,y
            losses[k] = loss.item()
    
        out[split] = losses.mean()
    model.train()
    
    return out


print('Laoding Tokenizer')
tokenizer = GptTokenizer()
tokenizer.load(tokenizer_path)

tokenized_data = pd.DataFrame(columns=["X", "y"])

def tokenize_data(data):
    return tokenizer.encode(data)


print('Tokenizing Dataset ')
# tokenized_data['X'] = DATA['X'].apply(lambda x: tokenize_data(x))
# tokenized_data['y'] = DATA['y'].apply(lambda x : tokenize_data(x))
num_processes = 12
def tokenize_chunk(chunk):
    tokenized_chunk = pd.DataFrame(columns=["X", "y"])
    tokenized_chunk['X'] = chunk['X'].apply(tokenize_data)
    tokenized_chunk['y'] = chunk['y'].apply(tokenize_data)
    return tokenized_chunk

# Split the dataset into chunks for parallel processing
chunk_size = len(DATA) // num_processes
chunks = [DATA.iloc[i:i+chunk_size] for i in range(0, len(DATA), chunk_size)]

# Process each chunk in parallel using ProcessPoolExecutor
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = []
    # Use tqdm to create a progress bar
    with tqdm(total=len(chunks)) as pbar:
        for chunk in chunks:
            futures.append(executor.submit(tokenize_chunk, chunk))
            pbar.update(1)

    # Gather results from all futures
    tokenized_chunks = [future.result() for future in concurrent.futures.as_completed(futures)]

# Concatenate results from all chunks
tokenized_data = pd.concat(tokenized_chunks)

TRAIN_DATA,VAL_DATA = train_test_split(tokenized_data,test_size=0.3,shuffle=True, random_state=42)

del DATA,tokenized_data

ITERS = len(tokenize_data) // batch_size
    
EPOCHS = 10

for epoch in tqdm(range(EPOCHS)):
    for iter in range(ITERS):
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model,eval_iters=eval_iters)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # if (iter+1) % checkpoint_steps == 0:
        #     with open(f'./checkpoints/chkpt_{iter+1}.pkl','wb') as f:
        #         pickle.dump(model,f)
        #     print('Checkpoints Saved')