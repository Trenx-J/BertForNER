import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import tqdm

from model.BertForNer import BertTokenizer
from model.BertForNer import Bert_NerModel,Bert_CRF_NerModel
from model.BertForNer import Bert_Lstm_NerModel,Bert_Lstm_CRF_NerModel
from model.BertForNer import Bert_BiLstm_NerModel,Bert_BiLstm_CRF_NerModel

from dataset.dataload import load_data,split_data,reload_data

from seqeval.metrics import f1_score,accuracy_score,precision_score,recall_score


class BertDataset(Dataset):
    def __init__(self,texts,lablels,label2index,max_len,tokenizer):
        self.texts=texts
        self.labels=lablels
        self.label2index=label2index
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __getitem__(self, index):
        text=self.texts[index]
        label=self.labels[index][:self.max_len]

        length=len(label)
        text_index=self.tokenizer.encode(text,add_special_token=True,max_length=self.max_len+2,padding="max_length",truncation=True,return_tensors="pt")
        label_index=[0]+[self.label2index.get(i,1) for i in label]+[0]*(self.max_len-length+1)

        label_index=torch.tensor(label_index)

        return text_index.reshape(-1),label_index,length
    def __len__(self):
        return self.texts.__len__()


batch_size=32
epochs=10
max_len=128
lr=0.0001
lstm_hidden=128
crf=0
lstm=0

device="cuda:0"if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    #texts,labels,label2index,index2label=load_data(os.path.join("data","train_data_public.csv"))
    #train_text,train_label,vali_text,vali_label=split_data(texts,labels)
    train_text,train_label,vali_text,vali_label,label2index,index2label=reload_data()

    model_path=os.path.join("..","bert-base-chinese")

    tokenizer=BertTokenizer.from_pretrained(model_path)
    train_dataset=BertDataset(train_text,train_label,label2index,max_len,tokenizer)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    vali_dataset=BertDataset(vali_text,vali_label,label2index,max_len,tokenizer)
    vali_dataloader=DataLoader(vali_dataset,batch_size=batch_size,shuffle=False)
    with open(os.path.join("result.txt"),"+a") as f:
        f.write('-----------------------------------------------------------------------------bert')
        if lstm==2:
            f.write('+bilstm')
            if crf:
                model=Bert_BiLstm_CRF_NerModel(model_path,lstm_hidden,len(label2index)).to(device)
                f.write('+crf\n')
            else:
                model=Bert_BiLstm_NerModel(model_path,lstm_hidden,len(label2index)).to(device)
                f.write('\n')
        elif lstm==1:
            f.write('+lstm')
            if crf:
                model=Bert_Lstm_CRF_NerModel(model_path,lstm_hidden,len(label2index)).to(device)
                f.write('+crf\n')
            else:
                model=Bert_Lstm_NerModel(model_path,lstm_hidden,len(label2index)).to(device)
                f.write('\n')
        else:
            if crf:
                model=Bert_CRF_NerModel(model_path,lstm_hidden,len(label2index)).to(device)
                f.write('+crf\n')
            else:
                model=Bert_NerModel(model_path,lstm_hidden,len(label2index)).to(device)
                f.write('\n')
    opt=torch.optim.AdamW(model.parameters(),lr)

    

    for epoch in range(epochs):
        

        print(f'epoch:{epoch+1}/{epochs}:')
        #train
        model.train()
        for batch_text_index,batch_label_index,batch_len in tqdm.tqdm(train_dataloader):
            batch_label_index=batch_label_index.to(device)
            batch_text_index=batch_text_index.to(device)

            loss=model(batch_text_index,batch_label_index)
            loss.backward()

            opt.step()
            opt.zero_grad()
        print(f'loss:{loss:.4f}')

        #validation
        model.eval()
        all_pre=[]
        all_tag=[]
        for batch_text_index,batch_label_index,batch_len in tqdm.tqdm(vali_dataloader):
            batch_label_index=batch_label_index.to(device)
            batch_text_index=batch_text_index.to(device)

            pre=model(batch_text_index)

            if not crf:
                pre=pre.cpu().numpy().tolist()
            tag=batch_label_index.cpu().numpy().tolist()

            for p,t,l in zip(pre,tag,batch_len):
                p=p[1:1+l]
                t=t[1:1+l]
                p=[index2label[i] for i in p]
                t=[index2label[i] for i in t]
                all_pre.append(p)
                all_tag.append(t)
        with open(os.path.join("result.txt"),"+a") as f:
            f1=f1_score(all_tag,all_pre)
            acc=accuracy_score(all_tag,all_pre)
            prec=precision_score(all_tag,all_pre)
            rec=recall_score(all_tag,all_pre)
            f.write(f'epoch:{epoch},f1:{f1:.4f},accuracy:{acc:.4f},precision:{prec:.4f},recall:{rec:.4f}\n')
        
        