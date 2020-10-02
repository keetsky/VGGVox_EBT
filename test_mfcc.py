
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
from vggm_mfcc import VGGM
import argparse


LR=0.01
B_SIZE=1
N_EPOCHS=150
N_CLASSES=11
transformers=transforms.ToTensor()
DATA_DIR="data/mfcc"
MODEL_PATH="models/VGGM_F.pth"

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
        mfcc=np.expand_dims(mfcc, 2) #(1,115,12)
        
        return transformers(mfcc), label,line

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

def test(model, Dataloader):
    counter=0
    top1=0
    top5=0    
    for audio, labels,pths in Dataloader:
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(audio)
        corr1, corr5=accuracy(outputs, labels, topk=(1,5))
        _, pred = outputs.topk(1, 1, True, True)
        print(pths,labels.cpu().numpy(),pred.cpu().numpy())
        #Cumulative values
        top1+=corr1
        top5+=corr5
        counter+=1
    print("Cumulative Val:\nTop-1 accuracy: %.5f\nTop-5 accuracy: %.5f"%(top1/counter, top5/counter))
    return top1/counter, top5/counter



if __name__=="__main__":
  testDataset=mfccDataset("train_cmlr_word_all_100_test.txt", DATA_DIR, is_train=False)
  testDataloader=DataLoader(testDataset, batch_size=1, shuffle=False)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model=VGGM(N_CLASSES)
  model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
  model.to(device)
  model.eval()
  with torch.no_grad():
    acc1, _=test(model, testDataloader)
    
