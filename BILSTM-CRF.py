import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import numpy as np

import time
import random
import os
from torchtext.vocab import Vectors
import argparse
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from CRF import CRF_model
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--bidir', default=True)
parser.add_argument('--hidden_size', default=128)
parser.add_argument('--epoch', default=15)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
BATCH_SIZE = 512

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

    def __build_features(self, sentences):

        masks = sentences != PAD_IDX

        embeds = self.embedding(sentences.long())

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


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


model.apply(init_weights)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = optim.Adam(model.parameters())


TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)


def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


print(model)


def bilstm_crf_acc(preds, y, tag_pad_idx):
    batch_size = y.shape[0]
    target_len = y.shape[1]
    pad_pred = []
    for sen in preds:
        sen += [tag_pad_idx]*(target_len - len(sen))
        pad_pred.append(sen)
    target_pred = torch.tensor(pad_pred)
    train_correct = ((target_pred == y)).sum()
    pad_eq = (y == tag_pad_idx).sum()
    correct = train_correct.item() - pad_eq.item()
    ratio = correct / (batch_size * target_len - pad_eq.item())
    return ratio



def train(model, iterator, optimizer, tag_pad_idx):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        text = batch.text.permute(1, 0)
        tags = batch.udtags.permute(1, 0)

        optimizer.zero_grad()
        loss = model.loss(text.to(device), tags.to(device))
        # predictions = model(text)

        _, pred = model(text)

        acc = bilstm_crf_acc(pred, tags, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, tag_pad_idx):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            text = batch.text.permute(1, 0)
            tags = batch.udtags.permute(1, 0)

            loss = model.loss(text.to(device), tags.to(device))

            _, pred = model(text)

            acc = bilstm_crf_acc(pred, tags, tag_pad_idx)
            # acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')