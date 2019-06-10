# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:03:46 2018

@author: Chen
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .ternarized_modules import  TernaryLinear, TernaryConv2d

tau = 0.5

class VGG_Cifar10(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG_Cifar10, self).__init__()
        self.infl_ratio = 1.
        self.features = nn.Sequential(
            TernaryConv2d(3, int(128*self.infl_ratio), kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(int(128*self.infl_ratio)),
            nn.Hardtanh(inplace=True),

            TernaryConv2d(int(128*self.infl_ratio), int(128*self.infl_ratio), kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(128*self.infl_ratio)),
            nn.Hardtanh(inplace=True),


            TernaryConv2d(int(128*self.infl_ratio), int(256*self.infl_ratio), kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(int(256*self.infl_ratio)),
            nn.Hardtanh(inplace=True),


            TernaryConv2d(int(256*self.infl_ratio), int(256*self.infl_ratio), kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(256*self.infl_ratio)),
            nn.Hardtanh(inplace=True),


            TernaryConv2d(int(256*self.infl_ratio), int(512*self.infl_ratio), kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(int(512*self.infl_ratio)),
            nn.Hardtanh(inplace=True),


            TernaryConv2d(int(512*self.infl_ratio), int(512*self.infl_ratio), kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(512*self.infl_ratio)),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            TernaryLinear(int(512*self.infl_ratio) * 4 * 4, int(1024*self.infl_ratio), bias=True),
            nn.BatchNorm1d(int(1024*self.infl_ratio)),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            TernaryLinear(int(1024*self.infl_ratio), int(1024*self.infl_ratio), bias=True),
            nn.BatchNorm1d(int(1024*self.infl_ratio)),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            TernaryLinear(int(1024*self.infl_ratio), num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            5: {'lr': 5e-2},
            10: {'lr': 1e-2},
            60: {'lr': 5e-3},
            90: {'lr': 1e-3},
            120: {'lr': 5e-4},
            150: {'lr': 1e-3},
            170: {'lr': 5e-4},
            180: {'lr': 1e-4}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, int(512*self.infl_ratio) * 4 * 4)
        x = self.classifier(x)
        return x


def vgg_cifar10_ternary(**kwargs):
    num_classes = getattr(kwargs,'num_classes', 10)
    # num_classes = getattr(kwargs, 'num_classes', 100)
    return VGG_Cifar10(num_classes)
