import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import numpy as np

import time
import random

import argparse
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from CRF import CRF_model
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--bidir', default=True)
parser.add_argument('--hidden_size', default=128)
parser.add_argument('--epoch', default=30)
parser.add_argument('--layers', default=2)


args = parser.parse_args()

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

UD_TAGS.build_vocab(train_data)

'''
Unique tokens in TEXT vocabulary: 8866
Unique tokens in UD_TAG vocabulary: 18
'''

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
BATCH_SIZE = 256
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


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
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)

        return scores, tag_seq


INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100
HIDDEN_DIM = int(args.hidden_size)
BIDIRECTIONAL = bool(args.bidir)
N_EPOCHS = int(args.epoch)
N_LAYERS = int(args.layers)
OUTPUT_DIM = len(UD_TAGS.vocab)


DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = BI_LSTM_CRF(INPUT_DIM,
                    EMBEDDING_DIM,
                    HIDDEN_DIM,
                    OUTPUT_DIM,
                    N_LAYERS,
                    BIDIRECTIONAL,
                    DROPOUT,
                    PAD_IDX)


model.apply(init_weights)
print(model)


print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = optim.Adam(model.parameters())


TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]


model.to(device)


def train(model, iterator, optimizer, tag_pad_idx):

    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:

        text = batch.text.permute(1, 0).to(device)
        tags = batch.udtags.permute(1, 0).to(device)
        # permutation to transform tensor size
        # [sent len, batch size] --> [batch size, sent len]

        optimizer.zero_grad()
        loss = model.loss(text, tags)
        # 不使用CrossEntropyLoss, loss函数由model.loss()方法生成

        _, pred = model(text)
        # 得到tag list
        acc = bilstm_crf_acc(pred, tags.cpu(), tag_pad_idx)
        # 计算batch accuracy

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, tag_pad_idx):

    epoch_loss = 0
    epoch_acc = 0
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in iterator:

            text = batch.text.permute(1, 0).to(device)
            tags = batch.udtags.permute(1, 0).to(device)
            # permutation to transform tensor size
            # [sent len, batch size] --> [batch size, sent len]

            loss = model.loss(text, tags)
            # 不使用CrossEntropyLoss, loss函数由model.loss()方法生成

            _, pred = model(text)
            # 得到tag list

            acc = bilstm_crf_acc(pred, tags.cpu(), tag_pad_idx)
            # 计算batch accuracy

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(
        model, train_iterator, optimizer, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(
        model, valid_iterator, TAG_PAD_IDX)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'Bi_LSTM_CRF.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # model.load_state_dict(torch.load('model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, TAG_PAD_IDX)

    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
