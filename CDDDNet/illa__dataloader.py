'''
All 3 Types of Dataloader 
1. Normal Dataloder 
2. Performance Dataloader 
3. Weighted Dataloder  
'''

# generic libraries
import ast
import copy  
import sys
import numpy as np 

# torch imports 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset

# file imports 
from illa__utils import * 

# Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device='cpu'

# collate 
def collate(batch):
    drug_graphs,taskMasks,labels = map(list, zip(*batch))
    drug_graphs=torch.tensor(drug_graphs).to(device).to(torch.float)
    taskMasks=torch.tensor(taskMasks).to(device).to(torch.float)
    labels=torch.tensor(labels).to(device).to(torch.float)
    return drug_graphs,taskMasks,labels


# Performance Based Dataset Sampler 
class PerformanceBasedDatasetSampler(torch.utils.data.sampler.Sampler):
    '''
    Assigns a metric value to each sample in the train dataset after every epoch . 
    Dynamic Resampling , instead of fixed initialisation.
    '''
    def __init__(self, dataset,weights=None,indices=None, num_samples=None):
          # Datset owning ..
        self.dataset = dataset
        if weights is None : 
            print('Weights for PRS not loader ...')
            sys.exit()

        self.weights = torch.DoubleTensor(weights)
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
            
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """
    def __init__(self, dataset,weightList=None,indices=None, num_samples=None, callback_get_label=None):
        # Datset owning ..
        self.dataset = dataset
        self.weightList=weightList
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices
        # define custom callback
        self.callback_get_label = callback_get_label
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        weights = [ self._get_label(self.dataset, idx) for idx in self.indices ]
        self.weights = torch.DoubleTensor(weights)
    
    def _get_label(self, dataset, idx):
        return self.weightList[idx]
     
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class Dataclass(Dataset):
    def __init__(self, dataset,lenDescriptors=512,imbalanced=True,performance=True):
        self.dataset = dataset
        self.dataset=self.dataset.dropna(inplace = False) 
        self.lengthofDescriptors=lenDescriptors
        self.y_desc_names = [col  for col in list(self.dataset.columns) if 'cddd_' in col]
        self.y_columns =[col  for col in list(self.dataset.columns) if '_new' in col]
        self.lendataset = len(self.dataset)
        self.dataset.loc[:, 'taskMask'] = self.dataset.taskMask.apply(lambda x: ast.literal_eval(x))
        self.classWeights=None 
        # Incase of imbalanced , we generate weights
        if(imbalanced or performance):
            classWeights = []
            df = copy.deepcopy(self.dataset)
            for i,task in enumerate(self.y_columns):
                df['taskSpecific_{}'.format(i)]=df.taskMask.apply(lambda row: row[i])
                res_i=df.loc[df['taskSpecific_{}'.format(i)]>0]
                classWeights.append(len(res_i)/len(self.dataset))
                # print('Column : {} , Num of Samples : {} , ClassWeight : {}'.format(task,len(res_i),len(res_i)/len(self.dataset)))
        
            self.classWeights=sum(classWeights)-np.asarray(classWeights,dtype=np.float)   
            self.weights =list(df.taskMask.apply(lambda x: round(np.dot(np.asarray(x,dtype=np.float),self.classWeights),2)))
            self.dataset['weights']=self.weights
            print('Class Weights are : {}'.format(self.classWeights))
            del res_i,df

    def __len__(self):
        return len(self.dataset)

    def fetch(self,idx):
        try : 
            y_desc=self.dataset.loc[idx,self.y_desc_names].tolist()
            y_desc=np.asarray(y_desc,dtype=np.float)
            taskMask=self.dataset.loc[idx]['taskMask']
            # taskMask=converttoList(taskMask)
            taskMask=np.asarray(taskMask,dtype=np.float)
            currtasks=np.count_nonzero(taskMask==1.0)
            y_gd = self.dataset.loc[idx,self.y_columns].tolist()
            y=np.asarray(y_gd).astype(np.float) 
        except Exception as exp : 
            print('Dataloading Errror : {}'.format(exp))
            return None 
        
        return [y_desc,taskMask,[y]]

    # Dataframe format 
    def __getitem__(self,idx):
        opt = self.fetch(idx)
        if(opt is None ):
            res=None
            found=False
            while(found==False):
                new_idx = np.random.randint(0,len(self.dataset)-1)
                res=self.fetch(new_idx)
                if(res==None):
                    continue
                [y_desc,taskMask,[y]]=res
                found=True
        else:
            [y_desc,taskMask,[y]]=opt
            
        return [y_desc,taskMask,[y]]