from numpy.core.defchararray import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

from data_loader import TinyImageNet
import os
import time
import cv2
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from models import *

from models import resnet3
from data.ori_dataset import ori_folder
from data.wm_dataset import wm_folder

from torch.utils.data import DataLoader



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
"""
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
"""
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=2)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, )
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#testloader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=2)
#testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, )

train_dataset = ori_folder('../../dataset/cifar10/train',transform_train)
#train_dataset = ori_folder('../../dataset/tiny-imagenet-200/train',transform_train)
train_dataset_wm = wm_folder('../wm_cifar10/',transform_train)

val_dataset = ori_folder('../../dataset/cifar10/test',transform_test)
#val_dataset = ori_folder('../../dataset/tiny-imagenet-200/val',transform_test)
val_dataset_wm = wm_folder('../wm_cifar10/',transform_test)

trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True, )
testloader = DataLoader(val_dataset, batch_size=256, shuffle=False, )

wm_trainloader = DataLoader(train_dataset_wm, batch_size=10, shuffle=True, )
wm_testloader = DataLoader(val_dataset_wm, batch_size=32, shuffle=False, )


# Model
print('==> Building model..')
#net = VGG('VGG16')
net = MobileNetV2()
#net = resnet3.resnet50()
net = net.to(device)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
#scheduler = MultiStepLR(optimizer, milestones=[16, 30, 50], gamma=0.1)
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#scheduler = MultiStepLR(optimizer, milestones=[15,25,35], gamma=0.1)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
#scheduler = MultiStepLR(optimizer, milestones=[20, 60, 100, 160, 180], gamma=0.1)
#scheduler = MultiStepLR(optimizer, milestones=[40, 80, 120, 160, 180], gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 150], gamma=0.1)
# Training
def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 200))
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end_time = time.time()
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # if batch_idx == 78:
            #     inputs = mix_data
            #     targets = mix_data_label
            #     inputs, targets = inputs.to(device), targets.to(device)
            #     outputs = net(inputs)
            #     loss = criterion(outputs, targets)
            #     test_loss += loss.item()
            #     _, predicted = outputs.max(1)
            #     total += targets.size(0)
            #     correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')

        if not os.path.isdir('checkpoint/checkpoint-cifar10-new-clean-MobileNetV2'):
            os.mkdir('checkpoint/checkpoint-cifar10-new-clean-MobileNetV2')
        torch.save(net.state_dict(), './checkpoint/checkpoint-cifar10-new-clean-MobileNetV2/ckpt.pth')
        best_acc = acc
#cifar100 vgg16 72.02
#tinyimagenet200  resnet 50 63.63
for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)
    scheduler.step()
print(best_acc)
