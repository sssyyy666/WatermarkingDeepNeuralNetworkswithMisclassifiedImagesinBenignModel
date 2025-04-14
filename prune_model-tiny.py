from cv2 import imwrite
import torch
import torch.nn as nn
from torch.nn import modules
from torch.nn.modules import module
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import torch.nn.utils.prune as prune
from models import *
import models
import cv2
import numpy as np 
from data_loader import TinyImageNet
from pruning import prune_attack
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from data.ori_dataset import ori_folder
from data.wm_dataset import wm_folder
from torch.utils.data import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
targets = [1]
total_number = 30

# Data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(int(256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = ori_folder('../../dataset/tiny-imagenet-200/train',transform_train)
#train_dataset = ori_folder('../../dataset/cifar100/train',transform_train)
#train_dataset = ori_folder('../cifar10/train',transform_train)
train_dataset_wm = wm_folder('../wm_tinyimagenet200/',transform_train)
#train_dataset_wm = wm_folder('../wm_cifar100/',transform_train)

val_dataset = ori_folder('../../dataset/tiny-imagenet-200/val',transform_test)
#val_dataset = ori_folder('../../dataset/cifar100/test',transform_test)
#val_dataset = ori_folder('../cifar10/test',transform_test)
val_dataset_wm = wm_folder('../wm_tinyimagenet200/',transform_test)
#val_dataset_wm = wm_folder('../wm_cifar100/',transform_test)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, )
testloader = DataLoader(val_dataset, batch_size=64, shuffle=False, )

wm_trainloader = DataLoader(train_dataset_wm, batch_size=2, shuffle=True, )
wm_testloader = DataLoader(val_dataset_wm, batch_size=64, shuffle=False, )


# ratios = [0.25, 0.5, 0.75, 0.85]
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.85, 0.9]
# ratios = list()
# init = 0.0
# for i in range(20):
#     ratios.append(init)
#     init += 0.05

print(ratios)

net = VGG('VGG16')
#net = resnet2.resnet18()
net = resnet3.resnet50()
#print(net)

#net.load_state_dict(torch.load('chk/cifar10_wm200/cifar10_wm200_epoch179.pth'))
#wm_acc =92.81
#net.load_state_dict(torch.load('chk/cifar10_drop_prob=0.05, block_size=5/cifar10_dropblock_epoch189.pth'))
#wm_acc+dropblock =93.18
#net.load_state_dict(torch.load('chk/cifar10_drop_prob=0.05,_block_size=7/cifar10_dropblock_epoch66.pth'))
#wm_acc+dropblock =93.06
#net.load_state_dict(torch.load('chk/cifar10_drop_prob=0.1, block_size=5/cifar10_dropblock_epoch180.pth'))
#wm_acc+dropblock =92.24
net.load_state_dict(torch.load('checkpoint/checkpoint-tiny200-wm/ckpt2.pth'))
#net.load_state_dict(torch.load('chk/tiny200-wm/tiny200_wm_epoch56.pth'))
#wm_acc+dropblock =91.98
net = net.to(device)
#print(net)
for ratio in ratios:
    ratio = float(ratio)
    net.load_state_dict(torch.load('checkpoint/checkpoint-tiny200-wm/ckpt2.pth'))
    net = net.to(device)
    '''module = net.features[0]
    prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)
    # print(list(module.named_buffers()))'''

    prune_attack(net, "resnet50", ratio)

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(wm_trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('prune ratio: %.2f | TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (ratio, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_loss1 = 0
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss1 += loss.item()
            _, predicted = outputs.max(1)
            total1 += targets.size(0)
            correct1 += predicted.eq(targets).sum().item()

        print('prune ratio: %.2f | TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (
        ratio, test_loss1 / (batch_idx + 1), 100. * correct1 / total1, correct1, total1))