from numpy.core.defchararray import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 

import torchvision.transforms as transforms

import os
import time
import cv2
import random
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from models import *
from models import resnet2
from data.ori_dataset import ori_folder
from data.wm_dataset import wm_folder
from torch.utils.data import DataLoader

random.seed(6)
np.random.seed(6)
torch.manual_seed(6)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(6)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
'''
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
'''
#train_dataset = ori_folder('../../dataset/cifar100_10/train',transform_train)
#train_dataset = ori_folder('../wm_cifar10_overwrite-1/',transform_train)

train_dataset = ori_folder('../../dataset/cifar100-2000/',transform_train)
#train_dataset = ori_folder('../../dataset/tiny-imagenet-200-2000',transform_train)
#train_dataset = ori_folder('../wm_tiny200_overwrite-1',transform_train)
#train_dataset_wm = wm_folder('../wm_cifar100/',transform_train)
#train_dataset_wm = wm_folder('../wm_tinyimagenet200/',transform_train)
train_dataset_wm = wm_folder('../wm_cifar100/',transform_train)

val_dataset = ori_folder('../../dataset/cifar100/test',transform_test)
#val_dataset = ori_folder('../../dataset/tiny-imagenet-200/val',transform_test)
val_dataset_wm = wm_folder('../wm_tinyimagenet200/',transform_test)

trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, )
testloader = DataLoader(val_dataset, batch_size=128, shuffle=False, )

wm_trainloader = DataLoader(train_dataset_wm, batch_size=64, shuffle=True, )
wm_testloader = DataLoader(val_dataset_wm, batch_size=64, shuffle=False, )


# Model
print('==> Building model..')
#net = resnet2.resnet18()
net = VGG('VGG16')
#net = resnet3.resnet50()

#net.load_state_dict(torch.load('chk/cifar10_wm200/cifar10_wm200_epoch179.pth'))
#wm_acc =92.81
#net.load_state_dict(torch.load('chk/cifar10_drop_prob=0.05, block_size=5/cifar10_dropblock_epoch189.pth'))
#wm_acc+dropblock =93.18
#net.load_state_dict(torch.load('chk/cifar10_drop_prob=0.05,_block_size=7/cifar10_dropblock_epoch66.pth'))
#wm_acc+dropblock =93.06
#net.load_state_dict(torch.load('chk/cifar10_drop_prob=0.1, block_size=5/cifar10_dropblock_epoch180.pth'))
#wm_acc+dropblock =92.24
#net.load_state_dict(torch.load('chk/cifar10_drop_prob=0.1, block_size=7/cifar10_dropblock_epoch175.pth'))
#wm_acc+dropblock =91.98

net.load_state_dict(torch.load('checkpoint/checkpoint-cifar100-wm/ckpt.pth'))

net = net.to(device)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#scheduler = MultiStepLR(optimizer, milestones=[2, 4, 6, 8, 100, 120, 140, 160, 180], gamma=0.1)
#scheduler = MultiStepLR(optimizer, milestones=[20, 40, 90, 120, 150, 180], gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)
# Training
def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 200))
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
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
    #torch.save(net.state_dict(), 'chk/transfer/cifar10-sketch_wm_epoch{}.pth'.format(epoch + 1))  # 添加保存
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

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
   #torch.save(net.state_dict(), 'chk/transfer/cifar10_wm_epoch{}.pth'.format(epoch + 1))  # 添加保存
    '''
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        checkpoint_file_result = 'checkpoint-wm-transfer'+str(round+1)
        if not os.path.isdir('./checkpoint/'+checkpoint_file_result):
            os.mkdir('./checkpoint/'+checkpoint_file_result)
        torch.save(net.state_dict(), './checkpoint/'+checkpoint_file_result+'/ckpt.pth')
        best_acc = acc
    '''

def wm(epoch):
    global best_acc
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

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def wm2(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    wm(epoch)
    #wm2(epoch)
#print(best_acc)

'''
#------------------------------------------------------------------
# Loading weight files to the model and testing them.
net_test = DenseNet121()
net_test = net_test.to(device)
net_test = torch.nn.DataParallel(net_test)

net_test.load_state_dict(torch.load('./checkpoint/DenseNet121_93_51.pth'))

net_test.eval()

test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net_test(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

'''
