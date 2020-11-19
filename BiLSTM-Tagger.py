import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import numpy as np

import time
import random
import os


SEED = 2020

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

TEXT = data.Field(lower=True)
UD_TAGS = data.Field(unk_token=None)
PTB_TAGS = data.Field(unk_token=None)

fields = (("text", TEXT), ("udtags", UD_TAGS))

train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

# print(type(train_data))
# <class 'torchtext.datasets.sequence_tagging.UDPOS'>

# print(vars(train_data.examples[0]))
# {'text': ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh',
# 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque',
# 'in', 'the', 'town', 'of', 'qaim', ',', 'near', 'the', 'syrian', 'border',
# '.'], udtags': ['PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'ADJ', 'NOUN', 'VERB',
# 'PROPN', PROPN', 'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'DET', 'NOUN', 'ADP',
# 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP', 'PROPN', 'PUNCT', 'ADP', 'DET',
# 'ADJ', 'NOUN', 'PUNCT']}

MIN_FREQ = 2

TEXT.build_vocab(train_data,
                 min_freq=MIN_FREQ,
                 # vectors="glove.6B.100d",
                 # unk_init=torch.Tensor.normal_
                 )

# NEED .vector_cache/glove.6B.zip

UD_TAGS.build_vocab(train_data)
'''
print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in UD_TAG vocabulary: {len(UD_TAGS.vocab)}")
Unique tokens in TEXT vocabulary: 8866
Unique tokens in UD_TAG vocabulary: 18
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


class BiLSTMPOSTagger(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(
            input_dim, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)

        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [sent len, batch size]

        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]

        return predictions


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(UD_TAGS.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = BiLSTMPOSTagger(INPUT_DIM,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        OUTPUT_DIM,
                        N_LAYERS,
                        BIDIRECTIONAL,
                        DROPOUT,
                        PAD_IDX)
