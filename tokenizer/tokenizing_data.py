import concurrent.futures
import pandas as pd
from tqdm import tqdm
import ultraimport

GptTokenizer = ultraimport('tokenizer/gpt.py','GptTokenizer',recurse=True)

tokenizer = GptTokenizer()
tokenizer.load('./tokenizer/models/gpt.model')
# Define your tokenization function
def tokenize_data(data):
    return tokenizer.encode(data)

# Define your main function

    # Your dataset
DATA = pd.read_csv('./data/dataset_v1.csv')  # Your dataset here

# Define the number of processes
num_processes =   # Choose the number of processes

# Create a new DataFrame to store tokenized data
tokenized_data = pd.DataFrame(columns=["X", "y"])

# Define a function for parallel tokenization
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

    for chunk in chunks:
        futures.append(executor.submit(tokenize_chunk, chunk))


    # Gather results from all futures
    tokenized_chunks = [future.result() for future in concurrent.futures.as_completed(futures)]

# Concatenate results from all chunks
tokenized_data = pd.concat(tokenized_chunks)

# Now you have your final DataFrame with tokenized data
# print(tokenized_data)
print('Saving Data')
tokenized_data.to_csv('./data/tokenized_data_v1.csv',index=False)
