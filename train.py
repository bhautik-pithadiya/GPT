import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from transformer.transformer import GPTLanguageModel
from tokenizer.gpt import GptTokenizer

# print('****** Loading Dataset *********')
# DATA = pd.read_csv('./data/dataset_v1.csv')
# TRAIN_DATA,VAL_DATA = train_test_split(DATA,test_size=0.3,shuffle=True, random_state=42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(n_embd = 64,
               n_head = 6,
               dropout = 6,
               block_size = 256,
               vocab_size = 10000,
               n_layer = 6):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GPTLanguageModel(n_embd,
                             n_head,
                             dropout,
                             block_size,
                             vocab_size,
                             n_layer
    )
    
    model = model.to(device)
    
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GptTokenizer()
    tokenizer.load(tokenier_path)
    
    return tokenizer

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

if __name__ == "__main__":
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 100
    eval_interval = 50
    learning_rate = 3e-4
    eval_iters = 200
    n_embd = 64
    n_head = 6
    n_layer = 6
    dropout = 0.2
    checkpoint_steps = 500
    vocab_size = 10000
    
    T_path = '/home/ksuser/Bhautik/GPT/tokenizer/models/gpt.model'
    
    print('******* Loading Model *********')
    model = load_model(n_embd,
                       block_size,
                       dropout,
                       n_head,
                       n_layer,
                       vocab_size)
    model = model.to(device)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print('******** Loading Tokenizer **********')
    tokenizer = load_tokenizer(T_path)
    
    def tokenize_data(data):
        return tokenizer.encode(data)
    
    tokenized_data = pd.DataFrame(columns=["X", "y"])
    
    print('************ Tokenizing Dataset *************')
    tokenized_data['X'] = DATA['X'].apply(lambda x: tokenize_data(x))
    tokenized_data['y'] = DATA['y'].apply(lambda x : tokenize_data(x))
    
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
            
            if (iter+1) % checkpoint_steps == 0:
                with open(f'./checkpoints/chkpt_{iter+1}.pkl','wb') as f:
                    pickle.dump(model,f)
                print('Checkpoints Saved')
            
    
    