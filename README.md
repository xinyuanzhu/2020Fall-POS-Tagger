# 2020Fall-NLP-Course-Project-POS-Tagger
## Common Methods
1. PGM based: HMM, MEMM, **CRF**
2. RNN based: RNN, **LSTM**, **BI-LSTM**
3. **BI-LSTM-CRF**


## Usage
python BiLSTM-Tagger.py --bidir True --bidir


### Tips
现在用了glove.6B.100d, so embedding_size不可指定, 就是100
## Progress
### Dataset
UDPOS in torchtext

Train, valid, Test #=12543, 2002, 2077 Sentences.



### Baselines
- LSTM
- Bi-LSTM
- CRF 




## TODO
*0.* 人工标注小数据集, 暂定2020大选新闻

1. BI-LSTM 
2. CRF
3. BI-LSTM-CRF
