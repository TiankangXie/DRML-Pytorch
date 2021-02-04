from torch.autograd.variable import Variable
from torch.nn import parameter
from data_loader import image_Loader
from DRML_model import DRML_net
from utils import *
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import logging
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from lr_scheduler import step_lr_scheduler

optim_dict = {'SGD':optim.SGD, 'Adam':optim.Adam}

cfg_lr = 0.001
master_file_path = "F:\\FaceExprDecode\\outs\\true_marked.csv"
n_epochs = 800
config_lr_decay_rate = 0.9
config_class_num = 12
config_train_batch = 24
config_test_batch = 12
config_test_every_epoch = 100
config_start_epoch = 80
config_optimizer_type = "SGD"
#config_lr_type = ""
config_gamma = 0.3
config_stepsize = 24
config_init_lr = 0.001

def adjust_lr(optimizer, decay_rate = 0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.9

use_gpu = torch.cuda.is_available()

master_file = pd.read_csv(master_file_path)

use_gpu = True

au_weight = calculate_AU_weight(master_file)
au_weight = torch.from_numpy(au_weight.astype('float'))
if use_gpu:
    au_weight = au_weight.cuda()

dsets = image_Loader(csv_dir=master_file_path, img_dir="F:\\FaceExprDecode\\aligned\\")


lengths = master_file.shape[0]
len_train = int(lengths*0.67)
len_test = lengths-len_train

train_set, test_set = random_split(dsets, [len_train,len_test])
train_set = DataLoader(train_set,batch_size=config_train_batch,shuffle=False)
test_set = DataLoader(test_set,batch_size=config_test_batch,shuffle=False)

net = DRML_net(config_class_num)
counter = 0

opt = optim_dict[config_optimizer_type](net.parameters(), lr = cfg_lr, momentum = 0.9, weight_decay = 0.9, nesterov = True)
param_lr = []
for param_group in opt.param_groups:
    param_lr.append(param_group['lr'])

if use_gpu:
    net = net.cuda()

for epoch_idx in range(n_epochs):
    
    if epoch_idx > config_start_epoch and epoch_idx % config_test_every_epoch == 0:
        print("testing:")
        net.train(False)
        f1score, accuracies = AU_detection_evalv2(test_set,net,use_gpu=use_gpu)
        print("F1 Score:",f1score, "accuracies: ", accuracies)

    for batch_index, (img,label) in enumerate(train_set):
        if counter > 0 and batch_index % 50 == 0:
            print('the number of training iterations is %d' % (counter))
            print('[epoch = %d][iter = %d][loss = %f][loss_au_dice = %f][loss_au_softmax = %f]' % (epoch_idx, batch_index,
                        loss.data.cpu().numpy(), loss_au_dice.data.cpu().numpy(), loss_au_softmax.data.cpu().numpy()))
        img = Variable(img)
        label = Variable(label)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        
        opt = step_lr_scheduler(param_lr, opt, epoch_idx, config_gamma, config_stepsize, config_init_lr)
        opt.zero_grad()
        pred = net(img)
        loss_au_softmax = au_softmax_loss(pred, label, weight=au_weight)
        loss_au_dice = au_dice_loss(pred, label, weight=au_weight)
        loss = loss_au_dice+loss_au_softmax
        loss.backward()
        opt.step()
        counter += 1


# %%
# %%
