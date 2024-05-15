import concurrent.futures
import pandas as pd
from tqdm import tqdm
import ultraimport
import torch

GptTokenizer = ultraimport('tokenizer/gpt.py','GptTokenizer',recurse=True)

tokenizer = GptTokenizer()
tokenizer.load('./tokenizer/models/nolan/gpt.model')
special_tokens = {
    '<eos>' : 10000,
    '<pad>': 10001
}
tokenizer.register_special_tokens(special_tokens)


#### Approach 1

# Define your tokenization function
# def tokenize_data(data):
#     return tokenizer.encode(data)

# # Define your main function

#     # Your dataset
# DATA = pd.read_csv('./data/dataset_v1.csv')  # Your dataset here

# # Define the number of processes
# num_processes =  12 # Choose the number of processes

# # Create a new DataFrame to store tokenized data
# tokenized_data = pd.DataFrame(columns=["X", "y"])

# # Define a function for parallel tokenization
# def tokenize_chunk(chunk):
#     tokenized_chunk = pd.DataFrame(columns=["X", "y"])
#     tokenized_chunk['X'] = chunk['X'].apply(tokenize_data)
#     tokenized_chunk['y'] = chunk['y'].apply(tokenize_data)
#     return tokenized_chunk

# # Split the dataset into chunks for parallel processing
# chunk_size = len(DATA) // num_processes
# chunks = [DATA.iloc[i:i+chunk_size] for i in range(0, len(DATA), chunk_size)]

# # Process each chunk in parallel using ProcessPoolExecutor
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     futures = []

#     for chunk in chunks:
#         futures.append(executor.submit(tokenize_chunk, chunk))


#     # Gather results from all futures
#     tokenized_chunks = [future.result() for future in concurrent.futures.as_completed(futures)]

# # Concatenate results from all chunks
# tokenized_data = pd.concat(tokenized_chunks)

# # Now you have your final DataFrame with tokenized data
# # print(tokenized_data)
# print('Saving Data')
# tokenized_data.to_csv('./data/tokenized_data_v1.csv',index=False,sep='|')


#### Approach 2 

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]
        # print(text)
        input_tokens = text.Text.split(' ')
        input_token = input_tokens[:-1] # Leave one position for the EOS token
        input_ids = self.tokenizer.encode(" ".join(tokens for tokens in input_token))
        # print(input_ids)
        
        # Target sequence (shifted by one)
        target_tokens = input_tokens[1:]
        target_ids = self.tokenizer.encode(" ".join(tokens for tokens in target_tokens))

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)

        return input_ids_tensor, target_ids_tensor

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_batch = pad_sequence(inputs, batch_first=True, padding_value=10001)
    target_batch = pad_sequence(targets, batch_first=True, padding_value=10001)
    return input_batch, target_batch

