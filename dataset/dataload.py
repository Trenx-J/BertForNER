import pandas as pd
import os
import random
import json

def load_data(path):
    all_data=pd.read_csv(path,encoding="utf-8")
    all_text=all_data["text"].tolist()
    texts=[]
    for text in all_text:
        tmp=[]
        for char in text:
            tmp.append(char)
        texts.append(tmp)
    all_label=all_data["BIO_anno"].tolist()
    labels=[]
    for label in all_label:
        label=label.split(" ")
        labels.append(label)

    label2index={"PAD":0,"UNK":1}
    for label in labels:
        for l in label:
             if l not in label2index:
                 label2index[l] =len(label2index)

    return texts,labels,label2index,list(label2index)

def split_data(texts,labels):
    n_total=len(texts)
    offset=int(n_total*4/5)
    if n_total==0:
        return []
    index=[i for i in range(n_total)]
    random.shuffle(index)
    train_index=index[:offset]
    vali_index=index[offset:]

    train_text=[texts[i] for i in train_index]
    train_label=[labels[i] for i in train_index]
    vali_text=[texts[i] for i in vali_index]
    vali_label=[labels[i] for i in vali_index]
    return train_text,train_label,vali_text,vali_label

def reload_data():
    with open(os.path.join("data","train_text.txt"),encoding='utf-8') as f:
        tmp = f.read()
        train_text=json.loads(tmp)
    with open(os.path.join("data","train_label.txt")) as f:
        tmp = f.read()
        train_label=json.loads(tmp)
    with open(os.path.join("data","vali_text.txt"),encoding='utf-8') as f:
        tmp = f.read()
        vali_text=json.loads(tmp)
    with open(os.path.join("data","vali_label.txt")) as f:
        tmp = f.read()
        vali_label=json.loads(tmp)
    with open(os.path.join("data","i2l.txt")) as f:
        tmp = f.read()
        index2label=json.loads(tmp)
    with open(os.path.join("data","l2i.txt")) as f:
        tmp = f.read()
        label2index=json.loads(tmp)
        
    return train_text,train_label,vali_text,vali_label,label2index,index2label


'''
def read_data(file):
    with open(file,"r",encoding="utf-8") as f:
        all_data=f.read().split("\n")

    all_text=[]
    all_label=[]
    text=[]
    label=[]
    for data in all_data:
        if data =="":
            all_text.append(text)
            all_label.append(label)
            text=[]
            label=[]

        else:
            t,l=data.split(" ")
            text.append(t)
            label.append(l)
    return all_text,all_label

def build_label(train_label):
    label_2_index={"PAD":0,"UNK":1}
    for label in train_label:
        for l in label:
             if l not in label_2_index:
                 label_2_index[l] =len(label_2_index)
    return label_2_index,list(label_2_index)
'''
