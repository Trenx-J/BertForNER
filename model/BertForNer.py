import torch
import torch.nn as nn
from torchcrf import CRF
import os
from transformers import BertTokenizer
from transformers import BertModel

class Bert_NerModel(nn.Module):
    def __init__(self,model_path,lstm_hidden,class_num):
        super().__init__()

        self.bert=BertModel.from_pretrained(model_path)

        self.classifier=nn.Linear(768,class_num)

        self.loss=nn.CrossEntropyLoss()

    def forward(self,batch_index,batch_label=None):
        output=self.bert(batch_index)
        bert_out=output[0]

        pre=self.classifier(bert_out)
        if batch_label is not None:
            loss=self.loss(pre.reshape(-1,pre.shape[-1]),batch_label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)

class Bert_CRF_NerModel(nn.Module):
    def __init__(self,model_path,lstm_hidden,class_num):
        super().__init__()

        self.bert=BertModel.from_pretrained(model_path)

        self.classifier=nn.Linear(768,class_num)

        self.crf=CRF(class_num,batch_first=True)
        
    def forward(self,batch_index,batch_label=None):
        output=self.bert(batch_index)
        bert_out=output[0]
        pre=self.classifier(bert_out)

        if batch_label is not None:
            loss =-self.crf(pre,batch_label)
            return loss
        else:
            pre=self.crf.decode(pre)
            return pre

class Bert_Lstm_NerModel(nn.Module):
    def __init__(self,model_path,lstm_hidden,class_num):
        super().__init__()

        self.bert=BertModel.from_pretrained(model_path)

        self.lstm=nn.LSTM(768,lstm_hidden,batch_first=True,num_layers=1)
        self.classifier=nn.Linear(lstm_hidden,class_num)

        self.loss=nn.CrossEntropyLoss()

    def forward(self,batch_index,batch_label=None):
        output=self.bert(batch_index)
        bert_out=output[0]
        lstm_out,_=self.lstm(bert_out)

        pre=self.classifier(lstm_out)
        if batch_label is not None:
            loss=self.loss(pre.reshape(-1,pre.shape[-1]),batch_label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)
        
class Bert_Lstm_CRF_NerModel(nn.Module):
    def __init__(self,model_path,lstm_hidden,class_num):
        super().__init__()

        self.bert=BertModel.from_pretrained(model_path)
        self.lstm=nn.LSTM(768,lstm_hidden,batch_first=True,num_layers=1)
        self.classifier=nn.Linear(lstm_hidden,class_num)

        self.crf=CRF(class_num,batch_first=True)

        
    def forward(self,batch_index,batch_label=None):
        output=self.bert(batch_index)
        bert_out=output[0]
        lstm_out,_=self.lstm(bert_out)
        pre=self.classifier(lstm_out)

        if batch_label is not None:
            loss =-self.crf(pre,batch_label)
            return loss
        else:
            pre=self.crf.decode(pre)
            return pre

class Bert_BiLstm_NerModel(nn.Module):
    def __init__(self,model_path,lstm_hidden,class_num):
        super().__init__()

        self.bert=BertModel.from_pretrained(model_path)

        self.lstm=nn.LSTM(768,lstm_hidden,batch_first=True,num_layers=1,bidirectional=True)
        self.classifier=nn.Linear(lstm_hidden*2,class_num)

        self.loss=nn.CrossEntropyLoss()

    def forward(self,batch_index,batch_label=None):
        output=self.bert(batch_index)
        bert_out=output[0]
        lstm_out,_=self.lstm(bert_out)

        pre=self.classifier(lstm_out)
        if batch_label is not None:
            loss=self.loss(pre.reshape(-1,pre.shape[-1]),batch_label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)
        
class Bert_BiLstm_CRF_NerModel(nn.Module):
    def __init__(self,model_path,lstm_hidden,class_num):
        super().__init__()

        self.bert=BertModel.from_pretrained(model_path)
        self.lstm=nn.LSTM(768,lstm_hidden,batch_first=True,num_layers=1,bidirectional=True)
        self.classifier=nn.Linear(lstm_hidden*2,class_num)

        self.crf=CRF(class_num,batch_first=True)

        
    def forward(self,batch_index,batch_label=None):
        output=self.bert(batch_index)
        bert_out=output[0]
        lstm_out,_=self.lstm(bert_out)
        pre=self.classifier(lstm_out)

        if batch_label is not None:
            loss =-self.crf(pre,batch_label)
            return loss
        else:
            pre=self.crf.decode(pre)
            return pre