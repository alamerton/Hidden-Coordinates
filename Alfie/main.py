import torch
from transformer_lens import HookedTransformer
import pandas as pd
import numpy as np
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import einops
from ivy import to_numpy
import plotly_express as px

torch.set_grad_enabled(True)

model = HookedTransformer.from_pretrained("gpt2-small")

# Initialise model components as variables
n_layers = model.cfg.n_layers  # Number of transformer layers
d_model = model.cfg.d_model  # Dimension of the model
n_heads = model.cfg.n_heads  # Number of attention heads
d_head = model.cfg.d_head  # Dimension of each attention head
d_mlp = model.cfg.d_mlp  # Dimension of the MLP
d_vocab = model.cfg.d_vocab  # Size of the vocabulary

# Load common words from Hugging Face
common_words = open("../common_words.txt", "r").read().split("\n")

# Calculate the number of tokens for each common word
num_tokens = [
    len(model.to_tokens(" " + word, prepend_bos=False).squeeze(0))
    for word in common_words
]

prefix = """The United States Declaration of Independence received its first 
formal public reading, in Philadelphia.\nWhen"""

# Create a DataFrame of words and their token counts
word_df = pd.DataFrame({"word": common_words, "num_tokens": num_tokens})
word_df = word_df.query('num_tokens < 4')  # Filter words < 4 tokens
word_df.value_counts("num_tokens")

# Define the prefix for context and set parameters
PREFIX_LENGTH = len(model.to_tokens(prefix, prepend_bos=True).squeeze(0))
NUM_WORDS = 7
MAX_WORD_LENGTH = 3

# Split the data into training and testing sets
train_filter = np.random.rand(len(word_df)) < 0.8
train_word_df = word_df.iloc[train_filter]
test_word_df = word_df.iloc[~train_filter]

# Group words by their token length
train_word_by_length_array = [np.array([" " + j for j in train_word_df[train_word_df.num_tokens == i].word.values]) for i in range(1, MAX_WORD_LENGTH + 1)]
test_word_by_length_array = [np.array([" " + j for j in test_word_df[test_word_df.num_tokens == i].word.values]) for i in range(1, MAX_WORD_LENGTH + 1)]

def gen_batch(batch_size, word_by_length_array):
    word_lengths = torch.randint(1, MAX_WORD_LENGTH+1, (batch_size, NUM_WORDS))
    words = []
    for i in range(batch_size):
        row = []
        for word_len in word_lengths[i].tolist():
            word = word_by_length_array[word_len-1][np.random.randint(len(word_by_length_array[word_len-1]))]
            row.append(word)
        words.append("".join(row))
    full_tokens = torch.ones((batch_size, PREFIX_LENGTH + MAX_WORD_LENGTH*NUM_WORDS), dtype=torch.int64)
    tokens = model.to_tokens([prefix + word for word in words], prepend_bos=True)
    full_tokens[:, :tokens.shape[-1]] = tokens
    
    first_token_indices = torch.concatenate([
        torch.zeros(batch_size, dtype=int)[:, None], word_lengths.cumsum(dim=-1)[..., :-1]
    ], dim=-1) + PREFIX_LENGTH
    
    last_token_indices = word_lengths.cumsum(dim=-1) - 1 + PREFIX_LENGTH
    return full_tokens, words, word_lengths, first_token_indices, last_token_indices
