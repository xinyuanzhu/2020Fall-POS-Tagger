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
from utils import *
parser = argparse.ArgumentParser()

parser.add_argument('--bidir', default=True)
parser.add_argument('--hidden_size', default=128)
parser.add_argument('--epoch', default=5)
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
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_
                 )

# NEED .vector_cache/glove.6B.zip

UD_TAGS.build_vocab(train_data)


'''
print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")

Unique tokens in TEXT vocabulary: 8866
Unique tokens in UD_TAG vocabulary: 18
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
BATCH_SIZE = 5

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

tagset_size = len(UD_TAGS.vocab)
print(f"Unique tokens in UD_TAG vocabulary: {len(UD_TAGS.vocab)}")

tag2idx = {}
for i in range(tagset_size):
    tag2idx[UD_TAGS.vocab.itos[i]] = i

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag2idx[START_TAG] = len(tag2idx)
tag2idx[STOP_TAG] = len(tag2idx)

print(tag2idx)


class BiLSTMCRFPOSTagger(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx,
                 tag2idx):

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
        self.target_size = output_dim
        self.dropout = nn.Dropout(dropout)

        self.transitions = nn.Parameter(torch.randn(
            output_dim, output_dim
        ))
    '''
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        return predictions
    '''

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(
            batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)  # [batch_size, 1, 16]
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions +
                                           log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _score_sentence(self, feats, label_ids):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(
            batch_size, self.tagset_size, self.tagset_size)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.tagset_size+label_ids[:, t-1]).view(-1, 1)) \
                + feats[:, t].gather(-1, label_ids[:,
                                                   t].view(-1, 1)).view(-1, 1)
        return score

    def _bert_enc(self, x):
        """
        x: [batchsize, sent_len]
        enc: [batch_size, sent_len, 768]
        """
        with torch.no_grad():
            encoded_layer, _ = self.bert(x)
            enc = encoded_layer[-1]
        return enc

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.tagset_size,self.tagset_size)

        log_delta = torch.Tensor(
            batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0.

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.tagset_size),
                          dtype=torch.long)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t +
                             1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # [batch_size, max_len, 16]
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)

    def _get_lstm_features(self, sentence):
        """sentence is the ids"""
        # self.hidden = self.init_hidden()
        embeds = self.dropout(self.embedding(sentence))  # [batch_size, sentence_len, embd_size]
        # 过lstm
        enc, _ = self.lstm(embeds)
        lstm_feats = self.fc(enc)
        return lstm_feats  # [8, 75, 16]

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)  # [8, 180,768]

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100
HIDDEN_DIM = int(args.hidden_size)
BIDIRECTIONAL = bool(args.bidir)
N_EPOCHS = int(args.epoch)
N_LAYERS = int(args.layers)
OUTPUT_DIM = len(UD_TAGS.vocab)


DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = BiLSTMCRFPOSTagger(INPUT_DIM,
                           EMBEDDING_DIM,
                           HIDDEN_DIM,
                           OUTPUT_DIM,
                           N_LAYERS,
                           BIDIRECTIONAL,
                           DROPOUT,
                           PAD_IDX,
                           tag2idx)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


model.apply(init_weights)


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


def train(model, iterator, optimizer, criterion, tag_pad_idx):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        text = batch.text.permute(1, 0)
        tags = batch.udtags.permute(1, 0)
        # text: tensor [sentence_len, batch_size]
        optimizer.zero_grad()

        predictions = model(text)

        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]

        predictions = predictions.view(-1, predictions.shape[-1])
        # sentence_len * batch_size, 18
        tags = tags.view(-1)
        # torch.Size([7424])

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]

        loss = criterion(predictions, tags)

        acc = categorical_accuracy(predictions, tags, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            text = batch.text
            tags = batch.udtags

            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

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
        model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(
        model, valid_iterator, criterion, TAG_PAD_IDX)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


model.load_state_dict(torch.load('model_01.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion, TAG_PAD_IDX)

print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
