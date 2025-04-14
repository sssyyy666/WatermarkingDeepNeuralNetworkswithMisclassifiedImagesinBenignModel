import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 

import torchvision
import torchvision.transforms as transforms
import cv2

from torch.utils.data import DataLoader
from models import *
from models import resnet3
from models import resnet2
from data.ori_dataset import ori_folder
from data.wm_dataset import wm_folder
from torchvision.transforms import Compose, Normalize, ToTensor
from data_loader import TinyImageNet
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
"""
transform_test = transforms.Compose([
    transforms.Resize(int(256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
"""
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

#train_dataset = ori_folder('../../dataset/cifar10/train',transform_train)
train_dataset = ori_folder('../cifar10/train',transform_train)
train_dataset_wm = wm_folder('../wm_tinyimagenet200/',transform_test)
#testset = ori_folder('../../dataset/tiny-imagenet-200/val',transform_test)
#testset = ori_folder('../../dataset/cifar100/test',transform_train)
testset = ori_folder('../../dataset/cifar10/test',transform_test)


#val_dataset = ori_folder('../../dataset/cifar10/test',transform_test)
val_dataset = ori_folder('../../dataset/cifar10/test',transform_test)
val_dataset_wm = wm_folder('../wm_cifar10/',transform_test)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, )
testloader = DataLoader(testset, batch_size=256, shuffle=False, )

wm_trainloader = DataLoader(train_dataset_wm, batch_size=64, shuffle=True, )
wm_testloader = DataLoader(val_dataset, batch_size=32, shuffle=False, )


# Model
print('==> Building model..')
#net = ResNet18()
#net = resnet3.resnet50()

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
# net = EfficientNetB0()
#net =resnet3.resnet50()
net =resnet2.resnet18()
#net = VGG('VGG16')
# net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

#------------------------------------------------------------------
# Loading weight files to the model and testing them.

methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

net_test = VGG('VGG16')
#net_test = MobileNetV2()
#net_test = resnet2.resnet18()
net_test = resnet3.resnet50()
# print(net_test)

net_test = net_test.to(device)

# net_test = torch.nn.DataParallel(net_test, device_ids=[0])
#model_name = './chk/cifar100-wm/cifar100_wm_epoch40.pth'
#model_name = './checkpoint/checkpoint-cifar10-wm-200/ckpt.pth'
#model_name = './checkpoint/checkpoint-tiny-imagenet-200-64-clean/ckpt.pth'
#model_name = "../../attack/NAD-main/NAD-main/chk/tiny200epoch20.pth"
#model_name = './checkpoint/checkpoint-cifar100-clean/ckpt.pth'
#model_name = 'cifar10-resnet34_8x.pt'
model_name = "./content/sketch_dfme.pt"
model_name = "./checkpoint/checkpoint-tiny-imagenet-200-clean/ckpt.pth"
# model_name = './checkpoint/checkpoint-clean/ckpt.pth'
#model_name = './checkpoint/checkpoint-cifar10-new-clean-vgg/ckpt.pth'
print("test model: ", model_name)
net_test.load_state_dict(torch.load(model_name, map_location=device))
net_test.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(wm_trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net_test(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
# wmcifar10 在 moblenetv2 11% vgg19 是 10%