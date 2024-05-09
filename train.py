import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


from transformer.transformer import GPTLanguageModel
from tokenizer.gpt import GptTokenizer

def load_model(n_embd,
               n_head,
               dropout,
               block_size,
               vocab_size,
               n_layer):
    
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

def load_tokenzier(tokenier_path):
    tokenizer = GptTokenizer()
    tokenizer.load(tokenier_path)
    
    return tokenizer

def load_dataset(dataset_path):
    
    data = open(dataset_path,'r')
    return data

def get_chunks(data):
    