# BertForNer
使用Bert-BiLstm-CRF做中文命名实体识别，使用的数据集来自https://aistudio.baidu.com/aistudio/competition/detail/802/0/datasets

# bert-base-chinese权重以及参数文件
文件夹与main文件同级

# 有关参数
batch_size=32 （每次训练的batch大小，根据个人的显存大小修改）

epochs=10 （训练轮次）

max_len=200 （训练单次句子最大长度）

lr=0.0001 （学习率）

lstm_hidden=128 （lstm的隐藏层大小）

crf=1（0为不要crf，1为要crf）

lstm=0（0为不要lstm，1为lstm，2为Bilstm）

# 运行
main文件F5直接运行

# 结果

| 模型名称        | F1         | accuracy   | Precision  | Recall |
| --------------- | ---------- | ---------- | ---------- | ------ |
| BERT            | 79.18%     | 94.86%     | 77.52%     | 80.90% |
| BERT+LSTM       | 80.19%     | 94.83%     | 78.88%     | 81.54% |
| BERT+BiLSTM     | 80.62%     | 94.88%     | 78.08%     | 83.32% |
| BERT+CRF        | 80.08%     | 94.88%     | 79.08%     | 81.09% |
| BERT+LSTM+CRF   | 80.52%     | 94.84%     | 79.25%     | 81.83% |
| BERT+BiLSTM+CRF | **80.98%** | **94.93%** | **80.27%** | 81.70% |

