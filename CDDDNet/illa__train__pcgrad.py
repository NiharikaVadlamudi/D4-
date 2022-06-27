'''
Performance + Imbalanced Train Setup 
'''

#Libraries
import csv 
import sys
import csv 
import json
from random import shuffle
import numpy as np
import copy 
# python imports
import os
import sys 
import ast
import argparse
import warnings
import pandas as pd
import wandb
import functools

warnings.filterwarnings("ignore")

# torch imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics.functional import accuracy,precision,recall,auc,mean_squared_error,spearman_corrcoef,mean_absolute_error
from torch.utils.data import BatchSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# File Imports 
# from illa__network import *
from illa__network import *
from illa__dataloader import * 
from illa__utils import *
from pcgrad import PCGrad

# CUDA Device Settings 
print('---------- Device Information ---------------')
if torch.cuda.is_available():
    print('__CUDNN VERSION  : {} '.format(torch.backends.cudnn.version()))
    print('__Number CUDA Devices  : {}'.format(torch.cuda.device_count()))
    print('Allocated GPUs : {} , CPUs : {}'.format(torch.cuda.device_count(),os.cpu_count()))
    device= torch.device("cpu")
      
else:
    print('Allocated CPUs : {}'.format(os.cpu_count()))
    device=torch.device('cpu')
    print('Only CPU Allocation ..')
print('--------------------------------------------------')

taskList = ['caco','lipophilicity','aqsol','ppbr','tox','clearance']

# Parser 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name',default='reg_tdc_task',help='name of the project')
    parser.add_argument('--weightedEpoch',default=3,type=int,help='Enter weighted epoch number ')
    parser.add_argument('--wid',default='illa',help='wid initialisation')
    parser.add_argument('--expdir',default='./pcg_corr_tdc_reg_random',help='experiment folder')
    parser.add_argument('--imbalancedDataloader',default=True,type=bool,help='Nature of dataloader')
    parser.add_argument('--mode',default='train',help='train/evaluate the model')
    parser.add_argument('--ntasks',default=6,type=int,help='N Classification Tasks')
    parser.add_argument('--learningRate',default=0.003,type=float,help='initial learning rate')
    parser.add_argument('--batchSize',default=128,type=int,help='batchsize for both test/train')
    parser.add_argument('--maxEpochs',default=25,type=int,help='maximum epochs')
    parser.add_argument('--modelfile',default=None,help='Model Weight file path')
    parser.add_argument('--resume',default=None,type=int,help='Enter model checkpoint path')
    args = parser.parse_args()
    return args



# Folder Creation 
def create_folder(args):

    try : 
        path=args.expdir
        pathCheckpoints=args.expdir+'/checkpoints'
        pathResults=args.expdir+'/test/results/'
        if(args.mode=='train'):
            os.system('mkdir -p %s' % pathCheckpoints)
        if(args.mode=='test'):
            os.system('mkdir -p %s' % pathResults)
        print('Experiment folder created ')

        # WandB Initialisation 
        WANDB_API_KEY="4169d431880a3dd9883605d90d1fe23a248ea0c5"
        WANDB_ENTITY="amber1121"
        os.environ["WANDB_RESUME"] ="allow"
        os.environ["WANDB_RUN_ID"] = args.wid
        wandb.init(project="[ILLA] D4 Regression",id=args.wid,resume='allow',config=args)
        print('WandB login completed ..')

    except Exception as ex : 
        print('Folder Creation Error , Exiting : {}'.format(ex))
        sys.exit()



class Trainer(object):

    def __init__(self, args):

        self.args=args 
        self.batchSize = args.batchSize
        self.ntasks=args.ntasks
        self.lr = args.learningRate
        self.maxEpochs=args.maxEpochs
        self.testCheckpointPath = args.modelfile
        self.expdir = args.expdir
        self.epoch=0
        self.weightedEpoch = self.args.weightedEpoch
        self.taskList=['caco','lipophilicity','aqsol','ppbr','tox','clearance']

        # Create folders 
        create_folder(args)

        # Model Initialisation 
        self.network=CDDD_FC()
        self.network.to(device)

        # Specify the data csv files  
        
        self.train_df = pd.read_csv('./data/illa_corr_reg_train_descriptors.csv')
        self.valid_df = pd.read_csv('./data/illa_corr_reg_test_descriptors.csv')
        # Converting Dataframes to Dataclass 
        self.train_dataset = Dataclass(self.train_df)
        self.valid_dataset = Dataclass(self.valid_df)

        # DataLoaders 
        # if(self.args.imbalancedDataloader):
        self.train_loader=DataLoader(self.train_dataset, collate_fn=collate, batch_size=self.batchSize,sampler=ImbalancedDatasetSampler(self.train_dataset,self.train_dataset.weights))
        # self.train_loader = DataLoader(self.train_dataset, collate_fn=collate, batch_size=self.batchSize, shuffle=True)
        # self.valid_loader = DataLoader(self.valid_dataset, collate_fn=collate, batch_size=self.batchSize,shuffle=True)
        self.test_loader = DataLoader(self.valid_dataset, collate_fn=collate, batch_size=1,shuffle=True)
        self.weight_train_loader=DataLoader(self.train_dataset, collate_fn=collate, batch_size=1,shuffle=False)

        # Optimisers 
        self.optimizer =torch.optim.Adam(
            self.network.parameters(),
            lr=args.learningRate,
            amsgrad=False)
        self.optimizer.zero_grad()
        self.optimizer=PCGrad(self.optimizer)
        self.optimizer.zero_grad()

        # self.lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        
        # Loss functions
        self.mse_loss_fn = torch.nn.MSELoss(reduce=True)
        self.mae_loss_fn =torch.nn.L1Loss(reduce=True)

        self.illa2=torch.nn.L1Loss(reduce=True)
        self.illa=torch.nn.MSELoss(reduce=True)
        # self.illa=torch.nn.L1Loss(reduce=True)

        self.mse_loss_fn1=torch.nn.MSELoss(reduce=True)
        self.mae_loss_fn1=torch.nn.L1Loss(reduce=False)



        # Resume 
        if args.resume is not None:
            self.resume(self.args.resume)
    

    def weightGeneration(self):
        self.samplebatchSize=1
        epochWeights=[0.0]*(len(self.weight_train_loader))
        self.network.eval() 
        skipped=0
        for step,samples in enumerate(self.weight_train_loader):
            try:
                self.optimizer.zero_grad()
                drug_graphs=samples[0].to(device).float()
                taskMasks=torch.tensor((samples[1])).reshape(( self.samplebatchSize,1,self.ntasks)).to(device).float()
                labels=torch.tensor(samples[2]).to(device).float()
                y_out= self.network([drug_graphs])
                y_out=y_out.reshape(( self.samplebatchSize,self.ntasks,1))

                # Reshaping .. 
                taskMasks= taskMasks.reshape((self.samplebatchSize,self.ntasks))
                labels=labels.reshape((self.samplebatchSize,self.ntasks))
                y_out=y_out.reshape((self.samplebatchSize,self.ntasks))
                y_out_new = taskMasks*y_out            
                loss = self.mae_loss_fn(labels,y_out_new)
                loss= torch.mean(loss,dim=0)
                illa_loss = self.illa(labels,y_out_new)
                epochWeights[step]=illa_loss.cpu().detach().numpy().item()
            
            except Exception as exp : 
                print('WeightGenerationProblem : {}'.format(exp))
                skipped+=1
                epochWeights[step]=0.5
                skipped+=1
                continue
                
        print('Weighted Loader , missed {} samples at epoch : {}'.format(skipped,self.epoch))
        epochWeights=np.asarray(epochWeights,dtype=np.float)
        # Normalise , by dividing by max of the weight list .. 
        epochWeights/=np.max(epochWeights)
        # print('Weights look like : {}'.format(epochWeights))
        return(epochWeights)


    def resume(self,epochNumber):
        print(' Model Training Resumed .. !')
        checkpointPath = os.path.join(self.args.expdir, 'checkpoints', 'epoch_%d.pth'%(epochNumber))
        save_state = torch.load(checkpointPath)
        try :
            self.network.load_state_dict(save_state['network_state_dict'])
            print('Model loaded ...  ')
        except Exception as ex :
            print('Model loading error ___ {}'.format(ex)) 
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        print('Model reloaded to resume from Epoch %d' % (self.epoch))
    

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'network_state_dict': self.network.state_dict()
        }
        save_name = os.path.join(self.args.expdir, 'checkpoints', 'epoch_%d.pth'%(epoch))
        torch.save(save_state, save_name)
        print('Saved model @ Epoch : {}'.format(self.epoch))

    def loop(self):
        for epoch in range(self.epoch,self.maxEpochs):
            self.epoch = epoch
            # if self.epoch > self.weightedEpoch : 
            #     print('Loading new dataset ... ')
            #     new_wgt_train_loader=DataLoader(self.train_dataset, collate_fn=collate, batch_size=self.batchSize,sampler=PerformanceBasedDatasetSampler(self.train_dataset,weights_))
            #     weights_= self.train(epoch,wgt_train_loader=new_wgt_train_loader)
            # else: 
            #     weights_= self.train(epoch,None)
            # wandb.log({'current_lr':self.optimizer.param_groups[0]['lr'],'epoch':self.epoch})
            weights_=self.train(self.epoch,None)
            print('Epoch : {} ---- Instances Length : {} '.format(self.epoch,len(self.train_loader)))

    def train(self,epoch,wgt_train_loader=None):
        print('Starting training...')
        self.network.train()
        wandb.watch(self.network,criterion=torch.nn.BCELoss(reduce=False),log="all")
        countErr=0
        running_mse_loss = []
        running_mae_loss = []
        print('Train Epoch : {}'.format(epoch))
        weights=None

        if wgt_train_loader is not None:
            self.data_loader = copy.deepcopy(wgt_train_loader)
            print('New train loader loaded ..')
        else:
            self.data_loader=copy.deepcopy(self.train_loader)
        
        for step,samples in enumerate(self.data_loader):
            try:
                self.optimizer.zero_grad()
                drug_graphs=samples[0].to(device).float()
                taskMasks=torch.tensor((samples[1])).reshape((self.batchSize,1,self.ntasks)).to(device).float()
                labels=torch.tensor(samples[2]).to(device).float()
                y_out= self.network([drug_graphs])
                y_out=y_out.reshape((self.batchSize,1,self.ntasks))

                # Reshaping .. 
                taskMasks= taskMasks.reshape((self.batchSize,self.ntasks))
                labels=labels.reshape((self.batchSize,self.ntasks))
                y_out=y_out.reshape((self.batchSize,self.ntasks))

                batchTaskLoss=[]
                batchMSELoss=0
                for i in range(0,self.ntasks):
                    task_filter =  taskMasks[:,i]>0
                    task_indices_ =  torch.nonzero(task_filter.int())
                    task_indices= task_indices_.squeeze()
                    numElements = len(task_indices.tolist())
                    if(numElements==0):
                        batchTaskLoss=[]
                        print('0  numElements of batch : Exiting..')
                        continue
        
                    # Pick task specific results . 
                    task_mask = torch.index_select(taskMasks,0,task_indices)
                    task_labels = torch.index_select(labels,0,task_indices)
                    task_output = torch.index_select(y_out,0,task_indices)
                    task_output_new = task_mask*task_output 
                    task_loss = self.mse_loss_fn1(task_labels,task_output_new)
                    batchTaskLoss.append(task_loss)
                    batchMSELoss+=task_loss.item()
                    
                self.optimizer.pc_backward(batchTaskLoss)
                self.optimizer.step()
                # Running MSE Loss 
                running_mse_loss.append(batchMSELoss/self.batchSize)
                print('Batch Step : {} MSE Loss : {} '.format(step,running_mse_loss[-1]))

            except Exception as exp : 
                countErr=countErr+1
                print('TrainException : {} Errors : {}'.format(exp,countErr))
                continue
        
    
        # Weighted Sampler 
        weights=self.weightGeneration()
        # print('Here > {}'.format(weights))
        # Validation 
        print('Testing .......')
        metrics_dict,val_loss=self.test()
        # self.lr_decay.step(val_loss)
       
        print('\n')
        print('Epoch : {}'.format(epoch))
        print("TRAIN - Epoch: {}  Training Loss:{} ".format(epoch,(np.mean(np.array(running_mse_loss)))))
        print("TEST - Epoch: {} , Test Loss : {} ".format(epoch,val_loss))
        print('\n')
        print('000000')

        # WandB Logging 
        wandb.log({'avg_train_error':np.mean(np.array(running_mse_loss)),'epoch':epoch})
        wandb.log({'mae_avg_train_error':np.mean(np.array(running_mae_loss)),'epoch':epoch})
        # wandb.log({'learning_rate':self.optimizer.param_groups[0]['lr'],'epoch':epoch})
        wandb.log({'train_data_errors':countErr,'epoch':epoch})
        wandb.log({'avg_test_error':val_loss,'epoch':epoch})
        
        for i in range(self.ntasks):
            for key in metrics_dict['task_'+str(taskList[i])].keys():
                wandb.log({'task_{}_{}'.format(str(taskList[i]),key):metrics_dict['task_'+str(taskList[i])][key],'epoch':epoch})

        # Save model checkpoint ..
        self.save_checkpoint(self.epoch)
        return(weights)
    
    def test(self):
        # Regression 
        errorPoints=0
        total_loss=0.0
        epoch_loss=[]
        tasks=[]
        # Metrics holders for all 
        metrics=['mae','spcoeff']
        for i in range(self.ntasks):
            task={'mae':[],'spcoeff':0.0}
            tasks.append(task)
        
        self.network.eval()
        y_gd_clearance = []
        y_out_clearance = []
        self.testbatchsize=1
        print('Testing .....')
        for step,samples in enumerate(self.test_loader):
            try : 
                drug_descriptors=samples[0]
                taskMasks=torch.tensor((samples[1])).reshape((self.testbatchsize,self.ntasks)).to(device).int()
                labels=torch.tensor(samples[2]).reshape((self.testbatchsize,self.ntasks)).to(device).float()

                # Network Output
                y_out= self.network([drug_descriptors])
                y_out=y_out.reshape((self.testbatchsize,self.ntasks))


                task_mask =taskMasks.reshape((self.ntasks,1))
                task_filter =  task_mask>0
                task_indices =  torch.nonzero(task_filter.int()).tolist()[0]

                for i in task_indices:
                    task = tasks[i]
                    y_gd_i  = labels[:,i]
                    y_out_i  = y_out[:,i]
                    loss_i = self.mae_loss_fn(y_gd_i,y_out_i)
                    task['mae'].append(loss_i.item())
                    epoch_loss.append(loss_i.cpu().detach().numpy())
                    if(self.taskList[i]=='clearance'):
                        y_gd_clearance.append(y_gd_i.item())
                        y_out_clearance.append(y_out_i.item())
                        
            except Exception as exp : 
                print('Testing Datapoint Error -- {} at step {}'.format(exp,step))
        
        # Loss 
        epoch_loss_net = np.mean(np.asarray(epoch_loss))

        # Compute Clearance 
        tasks[5]['spcoeff']=spearman_corrcoef(torch.FloatTensor(y_out_clearance),torch.FloatTensor(y_gd_clearance))
        print('Net Spcoeff Value  IS : {}'.format(tasks[5]['spcoeff']))    
        spcoeffval = tasks[5]['spcoeff']

        finalDict={}
        for i,task in enumerate(tasks):
            finalDict['task_'+str(taskList[i])]={}
            for metric in metrics :
                finalDict['task_'+str(taskList[i])][metric]=np.mean(np.asarray(task[metric],dtype=np.float))
        
        print('TEST RESULTS  : ')
        for k,v in finalDict.items():
            print('{}'.format(k),'{}\n'.format(v))

        # Average MSE Loss
        epoch_mae_net_loss = np.mean(np.array(epoch_loss,dtype=np.float).flatten())
        
        # WandB logging 
        wandb.log({'validation_data_errors':errorPoints,'epoch':self.epoch})
        wandb.log({'spearman coefficient [clearance]': spcoeffval,'epoch':self.epoch})
        
        return(finalDict,epoch_mae_net_loss)
        


if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    if(args.mode=='train'):
        trainer.loop()
    else:
        trainer.test()




        
