'''
- A simple network with CDDD Descriptors + Task FC Layers 
'''

# dgl imports 
#import dgl
#from dgl import DGLGraph
#from dgl.nn.pytorch import Set2Set,GATConv,GINConv

# torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F

# generic imports
import copy 
import numpy as np
device='cpu'


class CDDD_FC(nn.Module):

    def __init__(self,ntasks=6,input_dim = 512 , intermediate_dim_1= 128 , intermediate_dim_2 = 32 ,  intermediate_dim_3=8) : 
        super(CDDD_FC,self).__init__()
        # Generic params 
        self.ntasks=ntasks
        self.input_dim = input_dim
        self.intermediate_dim_1=intermediate_dim_1
        
        # Input CDDD feature size is : 512 
        # Ordered List --- from our prior knowledge on the ordering of property prediction tasks
        self.task_layers = nn.ModuleList()
        for i in range(0,self.ntasks):
            self.task_layers.append(nn.Sequential(
            nn.Linear(input_dim,intermediate_dim_1).to(device),
            nn.ReLU(),
            nn.Linear(intermediate_dim_1,intermediate_dim_2).to(device),
            nn.ReLU(),
            nn.Linear(intermediate_dim_2,intermediate_dim_3).to(device),
            nn.ReLU(),
            nn.Linear(intermediate_dim_3,1).to(device)))
            
    def forward(self,data):
        drug_descriptors = data[0]
        # drug_descriptors= drug_descriptors.float()
        # Concatenate the output features 
        for i,layer in enumerate(self.task_layers):
            z = layer(drug_descriptors)
            if(i==0):
                z_out=z
            else:
                z_out=torch.cat((z_out,z))
        return(z_out)
        







