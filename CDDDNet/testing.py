'''
Performance + Imbalanced Train Setup 
'''



'''
Incorporating Performance Based Random Sampler 
# PerformanceBasedDatasetSampler
'''

#Libraries
import csv 
import sys
import csv 
import json
from random import shuffle
import numpy as np

# python imports
import copy
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# File Imports 
from illa__network import *
from illa__dataloader import * 
from illa__utils import *

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

# Parser  -- Test specific 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file',default='test_result.csv',help='name csv file')
    parser.add_argument('--project_name',default='test_reg_tdc_task',help='name of the project')
    parser.add_argument('--wid',default=wandb.util.generate_id(),help='wid initialisation')
    parser.add_argument('--expdir',default='./TDC/performance_loss_change/',help='experiment folder')
    parser.add_argument('--mode',default='test',help='train/evaluate the model')
    parser.add_argument('--ntasks',default=6,type=int,help='N Classification Tasks')
    parser.add_argument('--batchSize',default=1,type=int,help='batchsize for both test/train')
    # parser.add_argument('--checkpointNumber',default=29,type=int,help='Model Weight file path number')
    parser.add_argument('--checkpointfile',default='./optimal_checkpoint',type=str,help='Model Weight file path number')
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
        wandb.init(project="[ILLA REG_TEST] D4 Molecular Property Prediction",id=args.wid,resume='allow',config=args)
        print('WandB login completed ..')

    except Exception as ex : 
        print('Folder Creation Error , Exiting : {}'.format(ex))
        sys.exit()


class Tester(object):
    def __init__(self, args):

        # Training Parameters 
        self.args=args 
        self.batchSize = args.batchSize
        self.ntasks=args.ntasks
        self.expdir = args.expdir
        self.epoch=0
        self.taskList=['caco','lipophilicity','aqsol','ppbr','tox','clearance']


        # Create folders 
        create_folder(args)
        # Model Initialisation 
        self.network=CDDD_FC()
        self.network.to(device)
        # Specify the data csv files 
        self.test_df = pd.read_csv('./data/illa_corr_reg_test_descriptors.csv')
        self.test_df_1 = copy.deepcopy(self.test_df)
       
        # Converting Dataframes to Dataclass 
        self.test_dataset = Dataclass(self.test_df)
        # DataLoaders
        self.test_loader = DataLoader(self.test_dataset, collate_fn=collate, batch_size=1,shuffle=True)
        # Loss functions
        self.mae_loss_fn =torch.nn.L1Loss(reduce=True)   
        #  checkpointPath = os.path.join(self.args.expdir, 'checkpoints', 'epoch_%d.pth'%(self.args.checkpointNumber))
        checkpointPath = 'optimal_checkpoint.pth'       
        save_state = torch.load(checkpointPath)
        self.network.load_state_dict(save_state['network_state_dict']) 
      
    def test(self):
        # Set the model to evaluation model 
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
        print('Testing .....')
        instancedict = {}
        y_outs = [] 
        y_tasks=[]
        print('Testing started.....')
        df_res= pd.DataFrame()
        for step,samples in enumerate(self.test_loader):
            try : 
                drug_descriptors=samples[0]
                taskMasks=torch.tensor((samples[1])).reshape((self.batchSize,self.ntasks)).to(device).int()
                labels=torch.tensor(samples[2]).reshape((self.batchSize,self.ntasks)).to(device).float()

                # Network Output
                y_out= self.network([drug_descriptors])
                y_out=y_out.reshape((self.batchSize,self.ntasks))
                y_res = y_out.tolist()

                tm = taskMasks.tolist()

                print('Step : {}'.format(step))
        

                instancedict[step]={'y_out':str(y_out[0]),'taskMask':str(tm)}
                # df_res['step']=step 
                df_res['y_out']=str(y_out[0])
                df_res['taskMask']=str(tm)

                y_outs.append([step,y_res[0][0],y_res[0][1],y_res[0][2],y_res[0][3],y_res[0][4],y_res[0][5]])
                y_tasks.append(tm)

                task_mask =taskMasks.reshape((self.ntasks,1))
                task_filter =  task_mask>0
                task_indices =  torch.nonzero(task_filter.int()).tolist()[0]

                for i in task_indices:
                    task = tasks[i]
                    y_gd_i  = labels[:,i]
                    y_out_i  = y_out[:,i]
                    loss_i= self.mae_loss_fn(y_gd_i,y_out_i)
                    task['mae'].append(loss_i.item())
                    if(self.taskList[i]=='clearance'):
                        y_gd_clearance.append(y_gd_i.item())
                        y_out_clearance.append(y_out_i.item())
                    
            except Exception as exp : 
                print('Testing Datapoint Error -- {} at step {}'.format(exp,step))

        # Write to CSV File 
        details=['sno','y_caco','y_lipophilicity','y_aqsol','y_ppbr','y_tox','y_clearance']
        with open(self.args.csv_file,'w') as f: 
            write = csv.writer(f) 
            write.writerow(details) 
            write.writerows(y_outs)

        f.close()
        print('Completed the csv file ...')
        print('Done~~ ')

        


if __name__ == '__main__':
    args = get_args()
    tester = Tester(args)
    tester.test()
    print('Testing procedure is completely done ...')


