import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets

from CRF import CRF_model
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import random
from utils import *

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


MIN_FREQ = 2

TEXT.build_vocab(train_data,
                 min_freq=MIN_FREQ,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_
                 )

# NEED .vector_cache/glove.6B.zip

UD_TAGS.build_vocab(train_data)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


path = "testing_corpus.txt"
test = datasets.UDPOS(path=path, fields=fields)


test_iterator = data.BucketIterator(
    test,
    batch_size=4,
    device=device,
    shuffle=False)


class BI_LSTM_CRF(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 tagset_size,
                 num_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        super(BI_LSTM_CRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        self.crf = CRF_model(hidden_dim*2, self.tagset_size)

        self.dropout = nn.Dropout(dropout)

    def __build_features(self, sentences):

        masks = sentences != PAD_IDX

        embeds = self.dropout(self.embedding(sentences.long()))

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = pack_padded_sequence(
            embeds, lengths=sorted_seq_length, batch_first=True)
        packed_output, _ = self.lstm(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)

        return scores, tag_seq


INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BIDIRECTIONAL = True
N_LAYERS = 2
OUTPUT_DIM = len(UD_TAGS.vocab)


DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
model = BI_LSTM_CRF(INPUT_DIM,
                    EMBEDDING_DIM,
                    HIDDEN_DIM,
                    OUTPUT_DIM,
                    N_LAYERS,
                    BIDIRECTIONAL,
                    DROPOUT,
                    PAD_IDX).to(device)

model.load_state_dict(torch.load('Bi_LSTM_CRF.pt', map_location=device))


def test(model, iterator,  tag_pad_idx):
    epoch_acc = 0
    predict = []
    tags = []

    with torch.no_grad():

        for batch in iterator:

            text = batch.text.permute(1, 0).to(device)
            tags = batch.udtags.permute(1, 0).to(device)

            _, pred = model(text)
            predict += pred
            acc = bilstm_crf_acc(pred, tags.cpu(), tag_pad_idx)

            epoch_acc += acc

    return epoch_acc / len(iterator), predict


valid_acc, predict = test(model, test_iterator, TAG_PAD_IDX)
print("Testing on new corpus:\nacc:", valid_acc)
