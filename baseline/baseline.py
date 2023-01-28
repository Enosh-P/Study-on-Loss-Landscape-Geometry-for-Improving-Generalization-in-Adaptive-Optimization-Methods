#imports
from __future__ import print_function
import os
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms


use_cuda = torch.cuda.is_available()

def get_data_loaders():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    kwargs = {'num_workers': 2, 'pin_memory': True}


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, **kwargs)

    return trainloader, testloader

#basicblock and resnet class
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ImageNet models
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2], 10)

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num):
        super(VGG, self).__init__()
        self.input_size = 32
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        self.fc = self._make_fc_layers()
        self.classifier = nn.Linear(self.n_maps, num)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out

    def _make_fc_layers(self):
        layers = []
        layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, self.n_maps),
                   nn.BatchNorm1d(self.n_maps),
                   nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG16():
    return VGG('VGG16', 10)


# map between model name and function
models = {
    'resnet18': ResNet18,
    'vgg16': VGG16,
}

def load(model_name):
    net = models[model_name]()
    net.eval()
    return net

#initialize params
def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Training
def train(trainloader, net, criterion, optimizer, use_cuda=True):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
                     
            if optimizer_val=='sgd' or optimizer_val=='adam':
              outputs = net(inputs)
              loss = criterion(outputs, targets)
              loss.backward()
              optimizer.step()     

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    return train_loss/total, 100 - 100.*correct/total,  100.*correct/total

def test(testloader, net, criterion, use_cuda=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    return test_loss/total, 100 - 100.*correct/total,  100.*correct/total


def main(model_name, rand_seed, optimizer_val):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(rand_seed)
        cudnn.benchmark = True

    trainloader, testloader = get_data_loaders()
    net = ResNet18() if model_name == 'ResNet18' else VGG16()
    init_params(net)

    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()

    if optimizer_val == 'sgd':
        lr_val = 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr_val, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer_val == 'adam':
        lr_val = 0.001
        optimizer = optim.Adam(net.parameters(), lr=lr_val, weight_decay=weight_decay)
   
    train_losses = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)

    for epoch in range(epochs):
        loss, train_err, train_acc = train(trainloader, net, criterion, optimizer, use_cuda)
        train_losses[epoch] = loss
        train_accuracy[epoch] = train_acc
        test_loss, test_err, test_acc = test(testloader, net, criterion, use_cuda)
        val_losses[epoch] = test_loss
        val_accuracy[epoch] = test_acc
        status = 'epoch: %d loss: %.5f train_err: %.3f train_acc: %.3f test_top1: %.3f test_loss %.5f test_acc: %.3f\n' % (epoch, loss, train_err, train_acc, test_err, test_loss, test_acc)
        print(status)
       
        if int(epoch) == 0.3*epochs or int(epoch) == 0.6*epochs or int(epoch) == 0.8*epochs:
            lr_val *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

    return train_losses, train_accuracy, val_losses, val_accuracy, net



if __name__ == "__main__":
    models = ['ResNet18', 'VGG16']
    seeds = [0, 1, 2]
    opts = ['sgd', 'adam']

    batch_size = 128
    start_epoch = 1
    lr_decay = 0.2
    weight_decay = 0.0005
    momentum = 0.9
    epochs = 200
    loss_name = 'crossentropy'

    for model_name in models:
        for rand_seed in seeds:
            for optimizer_val in opts:
                train_losses, train_accuracy, val_losses, val_accuracy, net = main(model_name, rand_seed, optimizer_val)

                torch.save(net.state_dict(), f'{optimizer_val}_baseline_{model_name}_seed{rand_seed}.pt')
                with open(f'{optimizer_val}_baseline_{model_name}_seed{rand_seed}.npy', 'wb') as numpy_file:
                    np.save(numpy_file, train_losses)
                    np.save(numpy_file, train_accuracy)
                    np.save(numpy_file, val_losses)
                    np.save(numpy_file, val_accuracy)
