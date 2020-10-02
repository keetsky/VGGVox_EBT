#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:52:25 2020

@author: darp_lord
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


class VGGM(nn.Module):
    
    def __init__(self, n_classes=1251):
        super(VGGM, self).__init__()
        self.n_classes=n_classes
        self.features=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7,7), stride=(2,2), padding=1)),
            ('bn1', nn.BatchNorm2d(96, momentum=0.5)),
            ('relu1', nn.ReLU()),
            ('mpool1', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=1)),
            ('bn2', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu2', nn.ReLU()),
            ('mpool2', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn3', nn.BatchNorm2d(384, momentum=0.5)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn5', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu5', nn.ReLU()),
            ('mpool5', nn.MaxPool2d(kernel_size=(5,3), stride=(3,2))),
            ('fc6', nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9,1), stride=(1,1))),
            ('bn6', nn.BatchNorm2d(4096, momentum=0.5)),
            ('relu6', nn.ReLU()),
            ('apool6', nn.AdaptiveAvgPool2d((1,1))),
            #('flatten', nn.Flatten()),
            ('flatten', Flatten()) 
            ]))
            
        self.classifier=nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(4096, 1024)),
            #('drop1', nn.Dropout()),
            ('relu7', nn.ReLU()),
            ('fc8', nn.Linear(1024, n_classes))]))
    
    def forward(self, inp):
        inp=self.features(inp)
        #inp=inp.view(inp.size()[0],-1)
        inp=self.classifier(inp)
        return inp

if __name__=="__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=VGGM(1251)
    model.to(device)
    print(summary(model, (1,512,300)))
