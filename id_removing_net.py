import torch
import torch.nn as nn
# from pts3d import *
import torchvision.models as models
import functools
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)




class IDR_net(nn.Module):
    def __init__(self, K=6):
        super(IDR_net,self).__init__()
        #self.mfcc_eocder = nn.Sequential(
        #    nn.Linear(12,64),
        #    nn.BatchNorm1d(64, momentum=0.5)),
        #    nn.ReLU(True),
        #    nn.Linear(64,128),
        #    nn.BatchNorm1d(128, momentum=0.5)),
        #    nn.ReLU(True),
        #    nn.Linear(128,256),
        #    nn.BatchNorm1d(256, momentum=0.5)),
        #    nn.ReLU(True),
        #    )        
        self.lstm = nn.LSTM(12,12,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(12,64),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.BatchNorm1d(128, momentum=0.5),
            nn.ReLU(True),
            nn.Linear(128,K),            
            )
        #self.WI=[]
        #for i in range(K):
            #wi=nn.Parameter(torch.randn(12, 12),requires_grad=True).cuda()
            #wi=torch.randn(12, 12,requires_grad=True).cuda() 
            #wi=nn.Parameter(torch.FloatTensor(12, 12).normal_(mean=0, std=0.01),requires_grad=True).cuda()
            #nn.init.normal_(wi.data, 0, 0.01)         
            #self.WI.append(wi)
        self.WI=nn.Parameter(torch.FloatTensor(K,12, 12).normal_(mean=0, std=0.01),requires_grad=True)#.cuda()
        #nn.init.normal_(self.WI.data, 0, 0.01)  
        self.I=torch.eye(12,12,requires_grad=False).cuda()
        self.K=K


    def forward(self,mfcc):
        #mfcc (n,115,12)
        #print(mfcc.shape)
        hidden = ( torch.autograd.Variable(torch.zeros(3, mfcc.size(0), 12).cuda()),
                      torch.autograd.Variable(torch.zeros(3, mfcc.size(0), 12).cuda()))
        lstm_out, _ = self.lstm(mfcc, hidden)
        lambs=[]
        for step_t in range(mfcc.size(1)): #115
            fc_in = lstm_out[:,step_t,:]
            
            lamb=self.lstm_fc(fc_in)
            lambs.append(lamb) 
                
        lambs=torch.stack(lambs,dim=1)   #(n,115,K)
        outs=[]
        for b in range(mfcc.size(0)):
            b_out=[]
            for step in range(mfcc.size(1)): #115
                ks=lambs[b,step,:]
                wi=self.I
                for i in range(self.K): 
                    wi =wi + ks[i]*self.WI[i]  
                      
                b_out.append(torch.matmul(mfcc[b,step],wi)) #torch.Size([12])
            b_out=torch.stack(b_out)
            outs.append(b_out) 
        #print(self.WI[0].grad)
        outs=torch.stack(outs)
        #print(outs.grad)
        #print(outs.shape) #(n,115,12)
        outs=torch.unsqueeze(outs,1)
        #print(outs.shape) ##(n,1,115,12)
        return outs

if __name__=="__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=IDR_net(11)
    model.to(device)
    #print(summary(model, (115,12)))
    #mfcc(115,13) ->(115,12)->(1,115,12)
    x=torch.rand(2,115,12)
    x=x.to(device)
    model(x)




