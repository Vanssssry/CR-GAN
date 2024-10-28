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



def test_eva(G,E,D,epoch,val_loader,test_loader,device,opt):
    #data_path=data_path=PACK_PATH+"/normal_test"
    #test_loader = load_dataset(256, data_path, 1)
    
    # normal_cate = np.arange(opt.nk)
    # normal_cate = set(normal_cate)
    normal_cate = {opt.normal_digit}
    #print(normal_cate)

    G.eval()
    E.eval()
    D.eval()
    
    target_all_val = []
    rec_all_val = []
    z_score_val = []
    zr_score_val = []
   
    target_all_test = []
    rec_all_test = []
    z_score_test = []
    zr_score_test = []
    with torch.no_grad():
        
        
        for idx, (image, target) in enumerate(val_loader):
            image = image.to(device)
            target = target.to(device)
            target_all_val.append(target.data.cpu().numpy())
           
            score1= torch.sum((G(E(image))-image)**2,dim=(1,2,3))
            rec_all_val.append(score1.data.cpu().numpy())
            
            
            score4 = (torch.sum(E(image)**2,dim=1))
            z_score_val.append(score4.data.cpu().numpy())

            # score8 = score4 + score1
            # # score8 = torch.mean(score1) / torch.mean(score4) * score4 + score1
            # zr_score_val.append(score8.data.cpu().numpy())
            
        
        for idx, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)
            target_all_test.append(target.data.cpu().numpy())
           
            score1= torch.sum((G(E(image))-image)**2,dim=(1,2,3))
            rec_all_test.append(score1.data.cpu().numpy())
            
            score4 = (torch.sum(E(image)**2,dim=1))
            z_score_test.append(score4.data.cpu().numpy())

            # score8 = score4 + score1
            # # score8 = torch.mean(score1) / torch.mean(score4) * score4 + score1
            # zr_score_test.append(score8.data.cpu().numpy())
            
            
    target_all_val = np.concatenate(target_all_val,axis=0)
    rec_all_val = np.concatenate(rec_all_val,axis=0)
    z_score_val = np.concatenate(z_score_val,axis=0)
    # zr_score_val = np.concatenate(zr_score_val,axis=0)
    n_rec_all_val = (rec_all_val - np.min(rec_all_val)) / (np.max(rec_all_val) - np.min(rec_all_val))
    n_z_score_val = (z_score_val - np.min(z_score_val)) / (np.max(z_score_val) - np.min(z_score_val))
    # rank_score_val = np.where(n_rec_all_val - n_z_score_val > 0, n_rec_all_val, n_z_score_val)
    rank_score_val = n_rec_all_val + 16 * n_z_score_val
    
    target_all_test = np.concatenate(target_all_test,axis=0)
    rec_all_test = np.concatenate(rec_all_test,axis=0)
    z_score_test = np.concatenate(z_score_test,axis=0)
    # zr_score_test = np.concatenate(zr_score_test,axis=0)
    n_rec_all_test = (rec_all_test - np.min(rec_all_val)) / (np.max(rec_all_val) - np.min(rec_all_val))
    n_z_score_test = (z_score_test - np.min(z_score_val)) / (np.max(z_score_val) - np.min(z_score_val))
    # print(n_rec_all_test)
    # print(n_z_score_test)
    # print(n_rec_all_test - n_z_score_test > 0)
    # rank_score_test = np.where(n_rec_all_test - n_z_score_test > 0, n_rec_all_test, n_z_score_test)
    #rank_score_test = n_rec_all_test * (opt.k + opt.nk) **(-1) + (opt.k + opt.nk) * n_z_score_test
    #rank_score_test = n_rec_all_test * opt.k **(-1) + opt.k  * n_z_score_test
    rank_score_test = n_rec_all_test + 16 * n_z_score_test    

    gt_val = (np.array([target in normal_cate for target in target_all_val])).astype(int)
    # print(target_all_val)
    # print(gt_val)
    auc_recon_val = roc_auc_score(gt_val,-1*rec_all_val) 
    auc_score_val = roc_auc_score(gt_val,-1*z_score_val)
    auc_rank_val = roc_auc_score(gt_val,-1*rank_score_val)
    # auc_zr_score_val = roc_auc_score(gt_val,-1*zr_score_val)
    

    
    gt_test = (np.array([target in normal_cate for target in target_all_test])).astype(int)
    # print(target_all_test)
    # print(gt_test)
    auc_recon_test = roc_auc_score(gt_test,-1*rec_all_test)
    auc_score_test = roc_auc_score(gt_test,-1*z_score_test)
    auc_rank_test = roc_auc_score(gt_test,-1*rank_score_test)
    # auc_zr_score_test = roc_auc_score(gt_test,-1*zr_score_test)
    
    eva_dic = {}
    eva_dic['val_recon'] = auc_recon_val
    eva_dic['val_zs'] = auc_score_val
    eva_dic['val_rank'] = auc_rank_val
    # eva_dic['val_zr'] = auc_zr_score_val
    eva_dic['test_recon'] = auc_recon_test
    eva_dic['test_zs'] = auc_score_test
    eva_dic['test_rank'] = auc_rank_test
    # eva_dic['test_zr'] = auc_zr_score_test
    
    eva_dic['epoch'] = epoch
    return eva_dic
