#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create by ZhanWang Zhang
复现论文Everybody's Talkin:
  ID-Removing network 部分

pip install "tqdm==4.43.0"
pip install "pandas==1.1.0"
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import signal_utils as sig
from scipy.io import wavfile
from id_removing_net import IDR_net
import argparse

from vggm_mfcc import VGGM
VGG_MODEL_PATH="models/VGGM_F.pth"
LR=0.01
B_SIZE=96
N_EPOCHS=10
N_CLASSES=30 #for CMLR Dataset
transformers=transforms.ToTensor()
LOCAL_DATA_DIR="data/mfcc"
MODEL_DIR="models/IDR"

class mfccDataset(Dataset):
    '''
    this is for CMLR lip reading Dataset , and this is not using audio fft feature(512,300),but using 
    mfcc feature(115,13) insteade 
    '''
    def __init__(self, txt_file, data_dir, croplen=115, is_train=True):
        if isinstance(txt_file, str):
            txt_file=os.path.join(data_dir,txt_file)
            if os.path.exists(txt_file):
                with open(txt_file) as f:
                    self.lines=f.readlines()
            else:
                raise Exception("not exist:%s"%txt_file)
        self.data_dir=data_dir
        self.is_train=is_train
        self.croplen=croplen
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line=self.lines[idx].strip()
        label=line.split('/')[0][1:] #s11->11
        label=int(label)-1    #11->10 label s11对应 10
        mfcc_path=os.path.join(self.data_dir,'audios',line) 
        mfcc=np.load(mfcc_path)#(115,13)mfcc 
        mfcc=mfcc[:,1:] ##(115,12) ,remove 0 channel
        if(self.is_train):
            start=np.random.randint(0,mfcc.shape[0]-self.croplen+1)
            mfcc=mfcc[start:start+self.croplen]
             
        mfcc=mfcc.astype(np.float32)
        #mfcc=np.expand_dims(mfcc, 2) #(1,115,12)
        #print(mfcc.shape)
        #mfcc=transformers(mfcc)
        
        #mfcc=torch.from_numpy(mfcc) 
        mfcc=transformers(mfcc) 
        mfcc=torch.squeeze(mfcc,(0))
        #print(mfcc.shape) 
        return mfcc, label

class IDRLoss(nn.Module):
    
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(IDRLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        #print(targets) #shape(n,num_classes)
        #print(log_probs) # #shape(n,num_classes)
        #out=(- targets * log_probs).mean(0).sum()
        
        #out=(-log_probs).mean(0).sum()/self.num_classes
           
        out=(-log_probs).mean().sum()
        return out
      

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul(100.0 / batch_size)).item())
        return res

def test(model, Dataloader,vggvox=None):
    counter=0
    top1=0
    top5=0    
    for audio, labels in Dataloader:
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(audio)
        if vggvox:
            outputs=vggvox(outputs) 
        corr1, corr5=accuracy(outputs, labels, topk=(1,5))
        #Cumulative values
        top1+=corr1
        top5+=corr5
        counter+=1
    print("Cumulative Val:\nTop-1 accuracy: %.5f\nTop-5 accuracy: %.5f"%(top1/counter, top5/counter))
    return top1/counter, top5/counter



if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./data/mfcc/")
    args=parser.parse_args()
    DATA_DIR=args.dir

    Datasets={
        "train":mfccDataset("train_cmlr_word_all_100_train.txt", DATA_DIR),
        "val":mfccDataset("train_cmlr_word_all_100_test.txt", DATA_DIR, is_train=False) ,
        "test":mfccDataset("train_cmlr_word_all_100_test.txt", DATA_DIR, is_train=False)}
    batch_sizes={
            "train":B_SIZE,
            "val":16,
            "test":16}
    Dataloaders={}
    Dataloaders['train']=DataLoader(Datasets['train'], batch_size=batch_sizes['train'], shuffle=True, num_workers=4)
    Dataloaders['val']=DataLoader(Datasets['val'], batch_size=batch_sizes['test'], shuffle=False, num_workers=2) 
    Dataloaders['test']=DataLoader(Datasets['test'], batch_size=batch_sizes['test'], shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    vggvox=VGGM(N_CLASSES)
    vggvox.load_state_dict(torch.load(VGG_MODEL_PATH, map_location=device))
    vggvox.to(device)
    vggvox.eval()

    model=IDR_net(N_CLASSES)
    model.to(device)
    #loss_func=nn.CrossEntropyLoss()
    loss_func=IDRLoss(N_CLASSES,label_smooth=False)
    optimizer=SGD(model.parameters(), lr=LR, momentum=0.99, weight_decay=5e-4)
    scheduler=lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/1.17)
    #Save models after accuracy crosses 75
    best_acc=10   #for id-removing loss
    update_grad=1
    best_epoch=0
    print("Start Training")
    for epoch in range(N_EPOCHS):
        model.train()
        running_loss=0.0
        corr1=0
        corr5=0
        top1=0
        top5=0
        random_subset=None
        loop=tqdm(Dataloaders['train'])
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        for counter, (audio, labels) in enumerate(loop, start=1):
            optimizer.zero_grad()
            #print(audio.shape,labels)
            audio = audio.to(device)
            labels = labels.to(device)
            
            if counter==32:
                random_subset=audio
            outputs = model(audio)
                
            outputs=vggvox(outputs)
            #print('output grad:', outputs.requires_grad)
            loss = loss_func(outputs, labels)
            running_loss+=loss.detach().item() #加入.detach().item() 来避免内存泄漏
            #running_loss=running_loss.detach().item()
            corr1, corr5=accuracy(outputs, labels, topk=(1,5))
            top1+=corr1
            top5+=corr5
            if(counter%update_grad==0):
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                loop.set_postfix(loss=(running_loss/(counter)), top1_acc=top1/(counter), top5_acc=top5/counter)
            #for name,parms in model.named_parameters():
                #print(name)
            #    if name=="WI":
            #        print(parms.grad)    
            #print(model.WI.grad)
           
        model(random_subset)
        model.eval()
        with torch.no_grad():
            acc1, _=test(model,Dataloaders['val'],vggvox)
            #if acc1>best_acc: #for CrossEntropyLoss
            if acc1<best_acc: #for id remove  loss ,精度越低越好  
                best_acc=acc1
                best_model=model.state_dict()
                best_epoch=epoch
                torch.save(best_model, os.path.join(MODEL_DIR,"IDR_BEST_%d_%.2f.pth"%(best_epoch, best_acc)))
        scheduler.step()
        
        PATH = os.path.join(MODEL_DIR,"IDR_F.pth")
        torch.save(model.state_dict(), PATH)   

    print('Finished Training..')
    model.eval()
    with torch.no_grad():
        #PATH = os.path.join(MODEL_DIR,"VGGM_F.pth")
        #torch.save(model.state_dict(), PATH)
        acc1=test(model, Dataloaders['test'],vggvox)
