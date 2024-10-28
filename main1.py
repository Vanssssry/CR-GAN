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
import seaborn as sns

np.set_printoptions(threshold=np.inf)



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--normal_digit", type=int, default=0, help="noraml class")
parser.add_argument("--auxiliary_digit", type=int, default=1, help="abnormal aviliable during training process")
parser.add_argument("--gpu", type=str, default='3', help="gpu_num")
parser.add_argument("--dataset", type=str, default='MNIST', help="choice of dataset(CIFAR,F-MNIST,MNIST)")
parser.add_argument("--dir", type=str, default='/summary//', help="save dir")
parser.add_argument("--name", type=str, default='result', help="file name")
parser.add_argument("--gamma_l", type=float, default=0.2, help="ratio of auxiliary anomaly data")
parser.add_argument("--gamma_a", type=float, default=0, help="ratio of auxiliary normal data")
parser.add_argument("--gamma_p", type=float, default=0, help="ratio of pollution data")
parser.add_argument("--k", type=float, default=1, help="the number of categories of the anomalous data")
parser.add_argument("--nk", type=float, default=1, help="the number of categories of the known normal data")

opt = parser.parse_args()

a = 1
b = 0
c = 0.75
bn = 0
alpha_a = 1
#weight of auxiliary data
alpha = 10
beta = 10
#threshold for label flipping
th = 0.8
prob_pz = 1
prob_px = 1

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}
if(opt.k<=1):
    seed = 12
else:
    seed = opt.auxiliary_digit
    #seed = 12
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
cuda = True if torch.cuda.is_available() else False

if(opt.dataset=='CIFAR'):
    import model.arch_cifar as arch
else:
    import model.arch_mnist1 as arch


adversarial_loss = torch.nn.MSELoss()

generator = arch.Generator()
discriminator = arch.Discriminator()
encoder = arch.Encoder()

if(opt.dataset == 'CIFAR'):
    from dataset.cifarset2 import create_loader 
    train_pos, train_neg, train_tpos, val_loader, test_loader = create_loader(opt,kwargs)
elif(opt.dataset == 'F-MNIST'):
    from dataset.fmnistset2 import create_loader
    train_pos, train_neg, train_tpos, val_loader, test_loader = create_loader(opt,kwargs)
else:
    from dataset.mnistset2 import create_loader
    train_pos, train_neg, train_tpos, val_loader, test_loader = create_loader(opt,kwargs)

if cuda:
    generator = generator.cuda('cuda')
    encoder = encoder.cuda('cuda')
    discriminator = discriminator.cuda('cuda')
    adversarial_loss = adversarial_loss.cuda('cuda')
    
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.000025, betas=(0.5,0.9))
optimizer_E = torch.optim.Adam(encoder.parameters(),lr=0.0001,betas=(0.5,0.9))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
StepLR_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.98)
StepLR_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.98)
StepLR_E = torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=100, gamma=0.98)

from testing1 import test_eva
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
auc_re = pd.DataFrame()
best_val_recon = 0
best_test_recon = 0
best_val_zs = 0
best_test_zs = 0
best_test_rank = 0
best_val_rank = 0
best_seen_rank = 0
best_unseen_rank = 0
import time 

for epoch in range(opt.n_epochs):
    start = time.time()
    i = 0
    StepLR_G.step()
    StepLR_E.step()
    StepLR_D.step()
    dxx_list = []
    dxz_list = []
    for (batch_pos,batch_neg, batch_tpos) in zip(train_pos,cycle(train_neg), cycle(train_tpos)):
        discriminator.train()
        generator.train()
        encoder.train()
        
    
        i+=1
        
        img_pos = batch_pos[0]
        img_neg = batch_neg[0]

        
        alpha_img_pos = batch_tpos[0]

        alpha_img_pos = alpha_img_pos.to(device)
        alpha_valid = alpha_a * torch.ones([alpha_img_pos.size(0), 1])
        alpha_valid = alpha_valid.to(device)
        alpha_z = encoder(alpha_img_pos)
        

        target_pos = batch_pos[1]
        target_neg = batch_neg[1]        
        optimizer_D.zero_grad()   
        valid = torch.ones([img_pos.size(0), 1])
        fake = torch.zeros([img_pos.size(0), 1])       
        img_pos = img_pos.to(device)  
        img_neg = img_neg.to(device)
        valid = valid.to(device)
        fake = fake.to(device)
    
        pos_imgs = img_pos.type(Tensor)
        neg_imgs = img_neg.type(Tensor)
    
        z_out_fake = Variable(Tensor(np.random.normal(0, 1, (img_pos.shape[0], opt.latent_dim))))
        z_out_fake = z_out_fake.to(device)
        
        
        
        
        img = torch.cat([img_pos,img_neg])
        z_out = encoder(img)


        z_out_real = z_out[:img_pos.shape[0]]
        z_out_neg = z_out[img_pos.shape[0]:]
        
        z = torch.cat([z_out_real,z_out_fake])
        gen = generator(z)
        gen_imgs_real = gen[:img_pos.shape[0]]

        
        gen_imgs_fake = gen[img_pos.shape[0]:]
        
        prob = np.random.uniform(0,1)

        D_pos_xz = adversarial_loss(discriminator(pos_imgs,z_out_real,'xz')[0], a*valid)
        D_fake_xz = adversarial_loss(discriminator(gen_imgs_fake,z_out_fake,'xz')[0], b*valid)

        res_dxz_tpos = discriminator(alpha_img_pos,alpha_z,'xz')[0]
        res_dxz_neg = discriminator(neg_imgs,z_out_neg,'xz')[0]
        if prob <= prob_pz:
            alpha_D_pos_xz = alpha * adversarial_loss(res_dxz_tpos, alpha_valid)
            D_neg_xz = beta * adversarial_loss(res_dxz_neg, bn*(torch.ones([img_neg.size(0), 1])).to(device))
        else:
            alpha_D_pos_xz = alpha * adversarial_loss(res_dxz_tpos, b*alpha_valid)
            D_neg_xz = beta * adversarial_loss(res_dxz_neg, a*(torch.ones([img_neg.size(0), 1])).to(device))
        
        d_loss_xz = D_pos_xz+D_fake_xz+D_neg_xz+alpha_D_pos_xz
        
        lambda_z = torch.sum(res_dxz_tpos) / alpha_img_pos.shape[0] - torch.sum(res_dxz_neg) / neg_imgs.shape[0]
        lambda_z = lambda_z.item()
        if lambda_z > th and prob_pz > 0.05:
            prob_pz -= 0.05
            if prob_pz < 0:
                prob_pz = 0
        elif lambda_z <= th and prob_pz < 1:
            prob_pz += 0.05
            if prob_pz > 1:
                prob_pz = 1
       
        dxz_list.append(d_loss_xz.data.cpu().numpy())
        
        
        D_pos_xx = adversarial_loss(discriminator(img_pos,img_pos,'xx')[0], a*valid)
        D_fake_xx = adversarial_loss(discriminator(pos_imgs,gen_imgs_real,'xx')[0], b*valid)


        res_dxx_tpos = discriminator(alpha_img_pos,alpha_img_pos,'xx')[0]
        res_dxx_neg = discriminator(neg_imgs,neg_imgs,'xx')[0]

        prob = np.random.uniform(0,1)
        if prob <= prob_px:
            alpha_D_pos_xx = alpha * adversarial_loss(res_dxx_tpos, alpha_valid)
            D_neg_xx = beta * adversarial_loss(res_dxx_neg, bn*(torch.ones([img_neg.size(0), 1])).to(device))
        else:
            alpha_D_pos_xx = alpha * adversarial_loss(res_dxx_tpos, b*alpha_valid)
            D_neg_xx = beta * adversarial_loss(res_dxx_neg, a*(torch.ones([img_neg.size(0), 1])).to(device))
        
        
        d_loss_xx = D_pos_xx+D_fake_xx+D_neg_xx+alpha_D_pos_xx
        
        lambda_x = torch.sum(res_dxx_tpos) / alpha_img_pos.shape[0] - torch.sum(res_dxx_neg) / neg_imgs.shape[0]
        lambda_x = lambda_x.item()
        if lambda_x > th and prob_px > 0.05:
            prob_px -= 0.05
            if prob_px < 0:
                prob_px = 0
        elif lambda_x <= th and prob_px < 1:
            prob_px += 0.05
            if prob_px > 1:
                prob_px = 1
        
        dxx_list.append(d_loss_xx.data.cpu().numpy())
        
        d_loss = d_loss_xz+d_loss_xx
        
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
        
        
        cycle_loss = adversarial_loss(discriminator(img_pos,img_pos,'xx')[0],c*valid)+adversarial_loss(discriminator(img_pos,gen_imgs_real,'xx')[0],c*valid)+adversarial_loss(discriminator(neg_imgs,neg_imgs,'xx')[0],c*(torch.ones([img_neg.size(0), 1])).to(device)) + adversarial_loss(discriminator(alpha_img_pos,alpha_img_pos,'xx')[0],c*alpha_valid)
       
        g_loss = adversarial_loss(discriminator(gen_imgs_fake,z_out_fake,'xz')[0], c*valid) + (1/4)*cycle_loss
  
            
        optimizer_G.zero_grad()
        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        
        optimizer_E.zero_grad()
               
        e_loss = adversarial_loss(discriminator(pos_imgs,z_out_real,'xz')[0],c*valid) + adversarial_loss(discriminator(neg_imgs,z_out_neg,'xz')[0], c*(torch.ones([img_neg.size(0), 1])).to(device)) + adversarial_loss(discriminator(alpha_img_pos, alpha_z, 'xz')[0],c*alpha_valid) #+ (1/3)*cycle_loss_2
        
        e_loss.backward()
        optimizer_E.step()
        
        discriminator.eval()
        generator.eval()
        encoder.eval()
        recon_pos = torch.mean(torch.sum((generator(encoder(img_pos))-img_pos)**2,dim=(1,2,3)))
        
        recon_neg = torch.mean(torch.sum((generator(encoder(img_neg))-img_neg)**2,dim=(1,2,3)))
        
        print(
                "[Epoch %d/%d] [Batch %d/%d] [recon_pos:%.3f][reconneg:%.3f][prob_z:%f][prob_x:%f]"
                % (epoch, opt.n_epochs, i, len(train_pos), recon_pos.item(),recon_neg.item(), prob_pz, prob_px)
            )
   

    if((np.mean(dxx_list)<0.015 or np.mean(dxz_list)<0.015) and epoch>300):
        break
    eva_dic = test_eva(generator,encoder,discriminator,epoch,val_loader,test_loader,device,opt)
    auc_re = auc_re.append(eva_dic,ignore_index=True)
    end = time.time()
    time_epoch = end-start
    
    if(eva_dic['val_recon']>=best_val_recon):
        if(eva_dic['test_recon'] >= best_test_recon):
            best_test_recon = eva_dic['test_recon']
            best_val_recon = eva_dic['val_recon']
    if(eva_dic['val_zs']>=best_val_zs):
        if(eva_dic['test_zs'] >= best_test_zs):
            best_test_zs = eva_dic['test_zs']
            best_val_zs = eva_dic['val_zs']
    if(eva_dic['val_rank']>=best_val_rank):
        if(eva_dic['test_rank'] >= best_test_rank):
            best_test_rank = eva_dic['test_rank']
            best_val_rank = eva_dic['val_rank']
            best_seen_rank = eva_dic['seen_rank']
            best_unseen_rank = eva_dic['unseen_rank']
    print(
                "[Epoch %d/%d] [val_recon:%.3f][test_recon:%.3f] [val_zs:%.3f][test_zs:%.3f][val_rank:%.3f][test_rank:%.3f] [best_recon:%.3f][best_zs:%.3f][best_rank:%.3f][epoch_time:%.3f]"
                % (epoch, opt.n_epochs,eva_dic['val_recon'],eva_dic['test_recon'],eva_dic['val_zs'],eva_dic['test_zs'],eva_dic['val_rank'],eva_dic['test_rank'],best_test_recon,best_test_zs, best_test_rank,time_epoch)
            )

file_best_auroc = open('result.txt',mode = 'a')
file_best_auroc.write('\n' + str(opt.normal_digit) + '/' + str(opt.auxiliary_digit) + ' ' + str(opt.gamma_l) + ' ' + str(opt.gamma_a) + ": " + str(best_test_recon) + ' ' + str(best_test_zs) + ' ' + str(best_test_rank) + ' ' + str(best_seen_rank)+ ' ' + str(best_unseen_rank))
file_best_auroc.close()  

    
