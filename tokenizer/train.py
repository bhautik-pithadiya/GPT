"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
# from .base import Tokenizer
# from .gpt import GptTokenizer
import ultraimport
Tokenizer = ultraimport('base.py', 'Tokenizer')
GptTokenizer = ultraimport('gpt.py','GptTokenizer',recurse=True)


PATH = 'models/'
VOCAB_SIZE = 2000
# open some text and train a vocab of 512 tokens
text = open("./wiki_dataset.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory

os.makedirs(PATH, exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([GptTokenizer], ["gpt"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, VOCAB_SIZE, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join(PATH, name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")


