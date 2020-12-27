# 2020Fall-NLP-Course-Project-POS-Tagger
https://github.com/xinyuanzhu/2020Fall-POS-Tagger
## Existing Methods
1. PGM based: HMM, MEMM, **CRF**
2. RNN based: RNN, **LSTM**, **Bi-LSTM**
3. **Bi-LSTM-CRF**


We reproduced the POS tagger based on **LSTM**, **Bi-LSTM**, **Bi-LSTM-CRF**.


## Dataset
UDPOS in torchtext

Train, valid, Test #=12543, 2002, 2077 Samples/Sentences.


## Usage
- for LSTM or BiLSTM

> python BiLSTM-Tagger.py --bidir True --layers 2 --hidden_size 128 --epoch 40 

- for Bi-LSTM-CRF

> python BiLSTM-CRF.py --bidir True --layers 2 --hidden_size 128 --epoch 40 

- To test your own corpus, data should be preprocessed to the form in "testing_corpus.txt" and you need to edit the file path in "model_pred_testing.py".

> python model_pred_testing.py


### Tips
To specify **EMBEDDING DIM** of our models, you need to modify the build_vocab process of TEXT and the corresponding model parameters.

