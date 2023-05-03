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

crf=1（1为Bert-Bilstm-CRF模型，0为Bert-lstm模型）

# 运行
main文件F5直接运行
