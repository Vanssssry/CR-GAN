# -*- coding: utf-8 -*-


import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import tqdm
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings



def create_loader(opt,kwargs):
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    dataset1 = datasets.MNIST('data-mnist', train=True, download=True,transform=data_transform)
    data1 = dataset1.data
    target1 = np.array(dataset1.targets)

    normal_cate = np.arange(opt.nk)
    normal_cate = set(normal_cate)
    #normal_cate = {opt.normal_digit}

    if(opt.gamma_p==0):
        if(opt.nk == 1):
            data1_p = data1[[(target in normal_cate and target != opt.normal_digit) for target in target1]]
            data1_n = data1[[target not in normal_cate for target in target1]]
            target1_p = target1[[(target in normal_cate and target != opt.normal_digit) for target in target1]]
            target1_n = target1[[target not in normal_cate for target in target1]]

            randIdx = np.arange(data1_n.shape[0])
            np.random.shuffle(randIdx)
            # normal_num = data1_p.shape[0]
            print(data1_p.shape[0])
            normal_num = data1[[(target in normal_cate) for target in target1]].shape[0]
            abnormal_num = 0

            p_randIdx = np.arange(data1_p.shape[0])
            np.random.shuffle(p_randIdx)
            #normal_num = data1_p.shape[0]
            a_normal_num = int((normal_num*opt.gamma_a)/(1-opt.gamma_a))
            #a_normal_num = int(opt.gamma_a)
            print(normal_num, abnormal_num, a_normal_num)
            
        
            dataset1.data = np.concatenate((data1_p,data1_n[randIdx[:abnormal_num]]),axis=0)
            dataset1.targets = np.concatenate((target1_p,target1_n[randIdx[:abnormal_num]]),axis=0)
            print(dataset1.data.shape)
            
            data3_p = data1[target1==opt.normal_digit]
            print(data3_p.shape)
            n_randIdx = np.arange(data3_p.shape[0])
            np.random.shuffle(n_randIdx)
            data_tpos = data3_p[n_randIdx[:a_normal_num]]
            target3_p = target1[target1==opt.normal_digit]
            target_tpos = target3_p[n_randIdx[:a_normal_num]]
            
            dataset1.data = np.concatenate((dataset1.data, data3_p[n_randIdx[a_normal_num:]]),axis=0)
            dataset1.targets = np.concatenate((dataset1.targets, target3_p[n_randIdx[a_normal_num:]]),axis=0)
            dataset1.data = torch.from_numpy(dataset1.data)
            dataset1.targets = torch.from_numpy(dataset1.targets)
            print(dataset1.data.shape)
        else:
            au_normal_cate = np.arange(opt.nk)
            np.random.shuffle(au_normal_cate)
            #print(type(opt.nk))
            print(au_normal_cate)
            au_normal_cate = set(au_normal_cate[:int(opt.nk)])
            print(au_normal_cate)
            data1_p = data1[[(target in normal_cate and target not in au_normal_cate) for target in target1]]
            data1_n = data1[[target not in normal_cate for target in target1]]
            target1_p = target1[[(target in normal_cate and target not in au_normal_cate) for target in target1]]
            target1_n = target1[[target not in normal_cate for target in target1]]

            randIdx = np.arange(data1_n.shape[0])
            np.random.shuffle(randIdx)
            # normal_num = data1_p.shape[0]
            normal_num = data1[[(target in normal_cate) for target in target1]].shape[0]
            abnormal_num = 0

            p_randIdx = np.arange(data1_p.shape[0])
            np.random.shuffle(p_randIdx)
            #normal_num = data1_p.shape[0]
            a_normal_num = int((normal_num*opt.gamma_a)/(1-opt.gamma_a))
            #a_normal_num = int(opt.gamma_a)
            print(normal_num, abnormal_num, a_normal_num)
            
        
            dataset1.data = np.concatenate((data1_p,data1_n[randIdx[:abnormal_num]]),axis=0)
            dataset1.targets = np.concatenate((target1_p,target1_n[randIdx[:abnormal_num]]),axis=0)
            
            data3_p = data1[[target in au_normal_cate for target in target1]]
            n_randIdx = np.arange(data3_p.shape[0])
            np.random.shuffle(n_randIdx)
            data_tpos = data3_p[n_randIdx[:a_normal_num]]
            target3_p = target1[[target in au_normal_cate for target in target1]]
            target_tpos = target3_p[n_randIdx[:a_normal_num]]
            
            dataset1.data = np.concatenate((dataset1.data, data3_p[n_randIdx[a_normal_num:]]),axis=0)
            dataset1.targets = np.concatenate((dataset1.targets, target3_p[n_randIdx[a_normal_num:]]),axis=0)

            dataset1.data = torch.from_numpy(dataset1.data)
            dataset1.targets = torch.from_numpy(dataset1.targets)
            print(dataset1.data.shape)
    else:
        if(opt.nk == 1):
            data1_p = data1[[(target in normal_cate and target != opt.normal_digit) for target in target1]]
            data1_n = data1[[target not in normal_cate for target in target1]]
            target1_p = target1[[(target in normal_cate and target != opt.normal_digit) for target in target1]]
            target1_n = target1[[target not in normal_cate for target in target1]]

            randIdx = np.arange(data1_n.shape[0])
            np.random.shuffle(randIdx)
            # normal_num = data1_p.shape[0]
            #print(data1_p.shape[0])
            normal_num = data1[[(target in normal_cate) for target in target1]].shape[0]
            abnormal_num = int((normal_num*opt.gamma_p)/(1-opt.gamma_p))

            p_randIdx = np.arange(data1_p.shape[0])
            np.random.shuffle(p_randIdx)
            #normal_num = data1_p.shape[0]
            a_normal_num = int((normal_num*opt.gamma_a)/(1-opt.gamma_a))
            #a_normal_num = int(opt.gamma_a)
            print(normal_num, abnormal_num, a_normal_num)
            
        
            dataset1.data = np.concatenate((data1_p,data1_n[randIdx[:abnormal_num]]),axis=0)
            dataset1.targets = np.concatenate((target1_p,target1_n[randIdx[:abnormal_num]]),axis=0)
            print(dataset1.data.shape)
            
            data3_p = data1[target1==opt.normal_digit]
            print(data3_p.shape)
            n_randIdx = np.arange(data3_p.shape[0])
            np.random.shuffle(n_randIdx)
            data_tpos = data3_p[n_randIdx[:a_normal_num]]
            target3_p = target1[target1==opt.normal_digit]
            target_tpos = target3_p[n_randIdx[:a_normal_num]]
            
            dataset1.data = np.concatenate((dataset1.data, data3_p[n_randIdx[a_normal_num:]]),axis=0)
            dataset1.targets = np.concatenate((dataset1.targets, target3_p[n_randIdx[a_normal_num:]]),axis=0)
            dataset1.data = torch.from_numpy(dataset1.data)
            dataset1.targets = torch.from_numpy(dataset1.targets)
            print(dataset1.data.shape)
        else:
            au_normal_cate = np.arange(opt.nk)
            np.random.shuffle(au_normal_cate)
            #print(type(opt.nk))
            print(au_normal_cate)
            au_normal_cate = set(au_normal_cate[:int(opt.nk)])
            print(au_normal_cate)
            data1_p = data1[[(target in normal_cate and target not in au_normal_cate) for target in target1]]
            data1_n = data1[[target not in normal_cate for target in target1]]
            target1_p = target1[[(target in normal_cate and target not in au_normal_cate) for target in target1]]
            target1_n = target1[[target not in normal_cate for target in target1]]

            randIdx = np.arange(data1_n.shape[0])
            np.random.shuffle(randIdx)
            # normal_num = data1_p.shape[0]
            normal_num = data1[[(target in normal_cate) for target in target1]].shape[0]
            abnormal_num = int((normal_num*opt.gamma_p)/(1-opt.gamma_p))

            p_randIdx = np.arange(data1_p.shape[0])
            np.random.shuffle(p_randIdx)
            #normal_num = data1_p.shape[0]
            a_normal_num = int((normal_num*opt.gamma_a)/(1-opt.gamma_a))
            #a_normal_num = int(opt.gamma_a)
            print(normal_num, abnormal_num, a_normal_num)
            
        
            dataset1.data = np.concatenate((data1_p,data1_n[randIdx[:abnormal_num]]),axis=0)
            dataset1.targets = np.concatenate((target1_p,target1_n[randIdx[:abnormal_num]]),axis=0)
            
            data3_p = data1[[target in au_normal_cate for target in target1]]
            n_randIdx = np.arange(data3_p.shape[0])
            np.random.shuffle(n_randIdx)
            data_tpos = data3_p[n_randIdx[:a_normal_num]]
            target3_p = target1[[target in au_normal_cate for target in target1]]
            target_tpos = target3_p[n_randIdx[:a_normal_num]]
            
            dataset1.data = np.concatenate((dataset1.data, data3_p[n_randIdx[a_normal_num:]]),axis=0)
            dataset1.targets = np.concatenate((dataset1.targets, target3_p[n_randIdx[a_normal_num:]]),axis=0)

            dataset1.data = torch.from_numpy(dataset1.data)
            dataset1.targets = torch.from_numpy(dataset1.targets)
            print(dataset1.data.shape)
    train_pos = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    
    dataset3 = datasets.MNIST('data-mnist', train=True, download=True,transform=data_transform)
    dataset3.data = data_tpos
    dataset3.targets = torch.from_numpy(target_tpos)
    if(opt.gamma_a == 0.1):
        train_tpos = torch.utils.data.DataLoader(dataset3, batch_size=opt.batch_size//9 + 1, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_a == 0.05):
        train_tpos = torch.utils.data.DataLoader(dataset3, batch_size=opt.batch_size//19, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_a == 0.2):
        train_tpos = torch.utils.data.DataLoader(dataset3, batch_size=opt.batch_size//4, shuffle=True, drop_last = False,**kwargs)
    else:
        train_tpos = torch.utils.data.DataLoader(dataset3, batch_size=10, shuffle=True, drop_last = False, **kwargs)
  
    
    dataset2 = datasets.MNIST('data-mnsit', train=True, download=True,transform=data_transform)
    data2 = dataset2.data
    target2 = np.array(dataset2.targets)
    if(opt.gamma_p==0):
        data2 = data2[[target not in normal_cate for target in target2]]
        target2 = target2[[target not in normal_cate for target in target2]]
    else:

        data2 = data1_n[randIdx[abnormal_num:]]
        target2 = target1_n[randIdx[abnormal_num:]]
            
        
    if(opt.k==1):
        data2 = data2[target2==opt.auxiliary_digit]
        target2 = target2[target2==opt.auxiliary_digit]
    else:
        anomaly_list = list(np.arange(opt.nk, 10))
        # anomaly_list.remove(opt.normal_digit)
        randIdx_list = np.arange(len(anomaly_list))
        np.random.shuffle(randIdx_list)
        #print(anomaly_list[randIdx_list[:int(opt.k)]])
        if(opt.k==2):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])] 
        elif(opt.k==3):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]]) |(target2==anomaly_list[randIdx_list[2]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]])] 
        elif(opt.k==4):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]]) |(target2==anomaly_list[randIdx_list[2]])|(target2==anomaly_list[randIdx_list[3]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]])|(target2==anomaly_list[randIdx_list[3]])] 
        elif(opt.k==5):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])] 
        elif(opt.k==6):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])|(target2==anomaly_list[randIdx_list[5]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])|(target2==anomaly_list[randIdx_list[5]])] 
        elif(opt.k==7):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])|(target2==anomaly_list[randIdx_list[5]])|(target2==anomaly_list[randIdx_list[6]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])|(target2==anomaly_list[randIdx_list[5]])|(target2==anomaly_list[randIdx_list[6]])]
    randIdx = np.arange(data2.shape[0])
    np.random.shuffle(randIdx)
    unlabeled_num = dataset1.data.shape[0]
    auxiliary_num = int((unlabeled_num*opt.gamma_l)/(1-opt.gamma_l))
    #auxiliary_num = int(opt.gamma_l)
    print(auxiliary_num)
    dataset2.data = data2[randIdx[:auxiliary_num]]
    dataset2.targets = torch.from_numpy(np.array(target2)[randIdx[:auxiliary_num]])
    
    if(opt.gamma_l == 0.1):
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//9, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_l == 0.05):
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//19, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_l == 0.2):
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//4, shuffle=True, drop_last = False,**kwargs)
    else:
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=10, shuffle=True, drop_last = False, **kwargs)

    # dataset3 = datasets.CIFAR10('data-cifar', train=True, download=True,transform=data_transform)
    # data3 = dataset3.data
    # target3 = np.array(dataset3.targets)

    # data3 = data3[target3==opt.normal_digit]
    # target3 = target3[target3==opt.normal_digit]

    # randIdx = np.arange(data3.shape[0])
    # np.random.shuffle(randIdx)
    # # unlabeled_num = dataset1.data.shape[0]
    # # auxiliary_num = int((unlabeled_num*opt.gamma_a)/(1-opt.gamma_a))
    # dataset3.data = data3[randIdx[:a_normal_num]]
    # dataset3.targets = np.array(target3)[randIdx[:a_normal_num]]
    # if(opt.gamma_a == 0.1):
    #     train_tpos = torch.utils.data.DataLoader(dataset3, batch_size=opt.batch_size//9, shuffle=True, drop_last = False,**kwargs)
    # elif(opt.gamma_a == 0.05):
    #     train_tpos = torch.utils.data.DataLoader(dataset3, batch_size=opt.batch_size//19, shuffle=True, drop_last = False,**kwargs)
    # elif(opt.gamma_a == 0.2):
    #     train_tpos = torch.utils.data.DataLoader(dataset3, batch_size=opt.batch_size//4, shuffle=True, drop_last = False,**kwargs)
    # else:
    #     train_tpos = torch.utils.data.DataLoader(dataset3, batch_size=10, shuffle=True, drop_last = False, **kwargs)


    
    dataset_val = datasets.MNIST('data-mnist', train=False, download=True,transform=data_transform)
    data_val = dataset_val.data
    target_val = np.array(dataset_val.targets)
    data_val_normal = data_val[[target in normal_cate for target in target_val]]
    target_val_normal = target_val[[target in normal_cate for target in target_val]]
    data_val_abnormal = data_val[[target not in normal_cate for target in target_val]]
    target_val_abnormal = target_val[[target not in normal_cate for target in target_val]]

    randIdx_normal = np.arange(data_val_normal.shape[0])
    randIdx_abnormal = np.arange(data_val_abnormal.shape[0])
    np.random.shuffle(randIdx_normal)
    np.random.shuffle(randIdx_abnormal)
    dataset_val.data=np.concatenate((data_val_normal[randIdx_normal[:200]],data_val_abnormal[randIdx_abnormal[:1800]]),axis=0)
    dataset_val.targets = np.concatenate((target_val_normal[randIdx_normal[:200]],target_val_abnormal[randIdx_abnormal[:1800]]),axis=0)
    dataset_val.data = torch.from_numpy(dataset_val.data)
    dataset_val.targets = torch.from_numpy(dataset_val.targets)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)

    dataset_test = datasets.MNIST('data-mnist', train=False, download=True,transform=data_transform)
    dataset_test.data=np.concatenate((data_val_normal[randIdx_normal[200:]],data_val_abnormal[randIdx_abnormal[1800:]]),axis=0)
    dataset_test.targets = np.concatenate((target_val_normal[randIdx_normal[200:]],target_val_abnormal[randIdx_abnormal[1800:]]),axis=0)
    dataset_test.data = torch.from_numpy(dataset_test.data)
    dataset_test.targets = torch.from_numpy(dataset_test.targets)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    return train_pos,train_neg, train_tpos, val_loader,test_loader
