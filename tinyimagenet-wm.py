from numpy.core.defchararray import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 
import torchvision
import torchvision.transforms as transforms

import os
import time
import cv2
import random
from tqdm import tqdm
from pruning import prune_model
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from data_loader import TinyImageNet
from models import *
from models import resnet2
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

#train_dataset = ori_folder('../../dataset/tiny-imagenet-200/train',transform_train)
#train_dataset = ori_folder('../../dataset/cifar100/train',transform_train)
train_dataset = ori_folder('../cifar10/train',transform_train)
#train_dataset_wm = wm_folder('../wm_tinyimagenet200/',transform_train)
train_dataset_wm = wm_folder('../wm_cifar10_newori/',transform_train)

#val_dataset = ori_folder('../../dataset/tiny-imagenet-200/val',transform_test)
#val_dataset = ori_folder('../../dataset/cifar100/test',transform_test)
val_dataset = ori_folder('../cifar10/test',transform_test)
#val_dataset_wm = wm_folder('../wm_tinyimagenet200/',transform_test)
val_dataset_wm = wm_folder('../wm_cifar10_newori/',transform_test)

trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True, )
testloader = DataLoader(val_dataset, batch_size=256, shuffle=False, )

wm_trainloader = DataLoader(train_dataset_wm, batch_size=10, shuffle=True, )
wm_testloader = DataLoader(val_dataset_wm, batch_size=256, shuffle=False, )


# Model
print('==> Building model..')
net = resnet2.resnet18()
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
#net = VGG('VGG16')
#net = resnet3.resnet50()
net = net.to(device)
# print(net)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# net_test = torch.nn.DataParallel(net_test, device_ids=[0])
#model_name = './checkpoint/checkpoint-tiny200-wm-61.85/ckpt.pth'
#model_name = './checkpoint/checkpoint-tiny-imagenet-200-64-clean/ckpt.pth'
#model_name = './checkpoint/checkpoint-tiny-imagenet-200-clean/ckpt.pth'
#model_name = 'cifar10-resnet34_8x.pt'
model_name = './checkpoint/checkpoint-clean/ckpt.pth'
# model_name = './checkpoint/checkpoint-clean/ckpt.pth'
print("test model: ", model_name)
net.load_state_dict(torch.load(model_name, map_location=device))

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
#scheduler = MultiStepLR(optimizer, milestones=[16, 30, 50], gamma=0.1)

# optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
#scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 150, 180], gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[20, 40, 90, 120, 150, 180], gamma=0.1)

#scheduler = MultiStepLR(optimizer, milestones=[15,25,35,45], gamma=0.1)

# Training

def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 200))
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    current_lr = 0.0
    # idx = random.randint(1,100)

    wminputs, wmtargets = [], []
    if wm_trainloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wm_trainloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    # randomly select 10 batches
    batch_idx_for_ft = random.sample(range(0, int(100000/128)), 10)

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        if wm_trainloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()

        # masking strategy
       ##     prune_model(net, "resnet18", ratio)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # # recover learning rate
        # if batch_idx in batch_idx_for_ft:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = current_lr

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

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    torch.save(net.state_dict(), 'chk/wm_cifar10_newori/cifar10_wm_epoch{}.pth'.format(epoch + 1))  # 添加保存模型参数的代码
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        #92.9素描
       # if not os.path.isdir('checkpoint/checkpoint-cifar100-wm-wm'):
       #     os.mkdir('checkpoint/checkpoint-cifar100-wm')
        torch.save(net.state_dict(), 'checkpoint/checkpoint-cifar10-wm-newori/ckpt.pth')
        best_acc = acc


def wm_test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(wm_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('WmLoss: %.3f | WmAcc: %.3f%% (%d/%d)' % (
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

#cifar100 vgg16 72.02
#tinyimagenet200  resnet 50 31epoch 63.81  60epoch 63.68
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    wm_test(epoch)
    scheduler.step()
print(best_acc)
