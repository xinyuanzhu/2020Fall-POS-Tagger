# 2020Fall-NLP-Course-Project-POS-Tagger
## Existing Methods
1. PGM based: HMM, MEMM, **CRF**
2. RNN based: RNN, **LSTM**, **Bi-LSTM**
3. **Bi-LSTM-CRF**

我们将要复现基于**LSTM**, **Bi-LSTM**, **Bi-LSTM-CRF**这三种方法.

## Dataset
UDPOS in torchtext

Train, valid, Test #=12543, 2002, 2077 Samples/Sentences.


## Usage
for LSTM or BiLSTM

python BiLSTM-Tagger.py --bidir True --layers 2 --hidden_size 128 --epoch 30 

for Bi-LSTM-CRF

python BiLSTM-CRF.py --bidir True --layers 2 --hidden_size 128 --epoch 30 

### Tips
word embedding因为使用GloVe预训练向量初始化, dim 暂不可通过命令行指定, 需要修改TEXT.build_vocab()等部分.

